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
from ssd.data.text_stream import batched, packed_rows
from ssd.models.subnetwork_draft import SubnetworkDraftModel
from ssd.models.target_wrapper import TargetWrapper
from ssd.runtime import checkpoint_path, input_device
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
    batches = batched(packed_rows(cfg.data, tokenizer, rank, world), s2.batch_size)
    if is_main:
        log.info("[stage2 %s] %.1fM params, T=%.1f, %d steps", name, sum(p.numel() for p in draft.parameters()) / 1e6, s2.temperature, steps)
    metrics = MetricLogger(f"[stage2 {name}]", cfg.training.log_every, is_main)
    out = checkpoint_path(cfg, 2)

    module.train()
    for step in range(steps):
        ids = next(batches).to(dev)
        taps = tw(ids, boundaries=[*layers, L], compute_logits=False).boundaries
        with torch.no_grad():
            target_hidden = base.model.norm(taps[L])
        labels = torch.full_like(ids, -100)
        labels[:, :-1] = ids[:, 1:]
        true_inputs = {i: taps[i] for i in layers}
        draft_hidden = module(ids, compute_logits=False, true_layer_inputs=true_inputs).hidden_states
        loss, stats = chunked_distill_loss(
            base.lm_head,
            draft_hidden,
            target_hidden,
            labels,
            s2.temperature,
            s2.kd_weight,
            s2.ce_weight,
        )
        stats = {"loss": loss.item(), **stats}

        opt.zero_grad(set_to_none=True)
        if accelerator is not None:
            accelerator.backward(loss)
        else:
            loss.backward()
        if draft.bridge_specs:
            torch.nn.utils.clip_grad_norm_(draft.parameters(), s2.grad_clip)
            opt.step()
        sched.step()
        metrics.update(step, steps, stats, ids.numel() * world, sched.get_last_lr()[0])

        if is_main and ((step + 1) % cfg.training.save_every == 0 or step + 1 == steps):
            draft.save_bridges(str(out), stage=2, step=step + 1)
    if is_main:
        log.info("[stage2 %s] saved %s", name, out)
    return out
