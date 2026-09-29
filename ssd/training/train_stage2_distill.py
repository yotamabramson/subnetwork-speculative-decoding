"""Stage 2: end-to-end logit distillation.

Runs the full draft (frozen base layers + bridges) and minimises
``kd_weight * T^2 * KL(target_T || draft_T) + ce_weight * CE(draft, next_token)``
against the frozen target's logits, computed online. Bridges start from the
Stage 1 checkpoint when one exists.

This matches inference, where the draft shares the target's KV cache. At each
position, the draft's layers attend to the *target's* keys/values for earlier
positions and use their own bridged state only for the current one
(``true_layer_inputs``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from ssd.config import SSDConfig, profile_name
from ssd.models.subnetwork_draft import SubnetworkDraftModel
from ssd.models.target_wrapper import TargetWrapper
from ssd.runtime import checkpoint_path, input_device
from ssd.training.batches import training_steps
from ssd.training.common import MetricLogger, make_optimizer
from ssd.training.losses import chunked_distill_loss

log = logging.getLogger("ssd")


def train_stage2(
    cfg: SSDConfig,
    base: nn.Module,
    tokenizer,
    init_from: Optional[str | Path] = "stage1",
    max_steps: Optional[int] = None,
    accelerator=None,
) -> Optional[Path]:
    s2 = cfg.training.stage2
    torch.manual_seed(cfg.training.seed + 1)
    layers = cfg.draft_layers
    name = profile_name(layers)
    draft = SubnetworkDraftModel(base, layers, cfg.bridge)
    if not draft.bridge_specs:
        log.info("[stage2 %s] no bridges (layers are contiguous); nothing to train", name)
        return None
    if init_from == "stage1":
        p = checkpoint_path(cfg, 1)
        init_from = p if p.exists() else None
        if init_from is None and draft.bridge_specs:
            log.warning("[stage2 %s] no stage1 checkpoint; distilling from untrained bridges", name)
    if init_from is not None:
        draft.load_bridges(str(init_from))

    steps = max_steps or s2.steps
    module: nn.Module = draft
    opt, sched = make_optimizer(draft.parameters(), s2.lr, s2.weight_decay, s2.warmup_steps, steps)
    rank, world, is_main = 0, 1, True
    if accelerator is not None:
        module, opt = accelerator.prepare(module, opt)
        rank, world, is_main = accelerator.process_index, accelerator.num_processes, accelerator.is_main_process

    dev = input_device(base)
    tw = TargetWrapper(base)
    L = base.config.num_hidden_layers
    batches = training_steps(cfg, tokenizer, s2.batch_size, s2.micro_batch, rank, world)
    if is_main:
        log.info("[stage2 %s] %.1fM params, T=%.1f, %d steps", name, sum(p.numel() for p in draft.parameters()) / 1e6, s2.temperature, steps)
    metrics = MetricLogger(f"[stage2 {name}]", cfg.training.log_every, is_main)
    out = checkpoint_path(cfg, 2)

    module.train()
    for step in range(steps):
        groups = [g.to(dev) for g in next(batches)]
        n_total = max(1, sum(g.n_pred for g in groups))
        opt.zero_grad(set_to_none=True)
        stats: dict[str, float] = {}
        tokens = 0
        for g in groups:
            if g.n_pred == 0:
                continue
            share = g.n_pred / n_total  # weight groups by their number of loss positions
            tokens += g.n_pred
            taps = tw(g.ids, boundaries=[*layers, L], compute_logits=False).boundaries
            with torch.no_grad():
                target_hidden = base.model.norm(taps[L])
            labels = torch.full_like(g.ids, -100)
            labels[:, :-1] = g.ids[:, 1:]
            true_inputs = {i: taps[i] for i in layers}
            if s2.ttt_steps > 1:
                # Multi-step training: loss at every unrolled draft step; step j is
                # only defined from position j-1 on (its chain starts at t-j+1 >= 0).
                outs = draft.forward_unrolled(g.ids, true_inputs, s2.ttt_steps)
                weights = [s2.ttt_decay**j for j in range(s2.ttt_steps)]
                loss = 0.0
                for j, (h, w) in enumerate(zip(outs, weights)):
                    valid = g.pred.clone()
                    valid[:, :j] = False
                    if not bool(valid.any()):
                        continue
                    lj, sj = chunked_distill_loss(
                        base.lm_head, h[valid], target_hidden[valid], labels[valid],
                        s2.temperature, s2.kd_weight, s2.ce_weight,
                    )
                    loss = loss + w * lj
                    stats[f"top1_d{j + 1}"] = stats.get(f"top1_d{j + 1}", 0.0) + sj["top1_agree"] * share
                loss = loss / sum(weights)
            else:
                draft_hidden = module(g.ids, compute_logits=False, true_layer_inputs=true_inputs).hidden_states
                loss, sj = chunked_distill_loss(
                    base.lm_head, draft_hidden[g.pred], target_hidden[g.pred], labels[g.pred],
                    s2.temperature, s2.kd_weight, s2.ce_weight,
                )
                for k, v in sj.items():
                    stats[k] = stats.get(k, 0.0) + v * share
            stats["loss"] = stats.get("loss", 0.0) + loss.item() * share
            if accelerator is not None:
                accelerator.backward(loss * share)
            else:
                (loss * share).backward()
        if draft.bridge_specs:
            torch.nn.utils.clip_grad_norm_(draft.parameters(), s2.grad_clip)
            opt.step()
        sched.step()
        metrics.update(step, steps, stats, tokens * world, sched.get_last_lr()[0])

        if is_main and ((step + 1) % cfg.training.save_every == 0 or step + 1 == steps):
            draft.save_bridges(str(out), stage=2, step=step + 1)
    if is_main:
        log.info("[stage2 %s] saved %s", name, out)
    return out
