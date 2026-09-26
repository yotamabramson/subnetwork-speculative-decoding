"""Stage 1: bridge feature regression.

For every bridge ``src -> tgt`` in a profile, minimise
``alpha * relMSE(pred, h_tgt) + beta * (1 - cos(pred, h_tgt))`` where
``h_*`` are target-model boundary activations.

* ``mode: teacher`` — the bridge input is the target's own ``h_src``. The
  bridges train independently and see clean inputs.
* ``mode: chained`` — the bridge input comes from running the draft itself,
  so upstream errors propagate, as they will at inference.

Activations come from the frozen target, computed online, or from a cache
written by ``ssd.data.extract_activations``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn as nn

from ssd.config import SSDConfig, profile_name
from ssd.data.text_stream import batched, packed_rows
from ssd.models.subnetwork_draft import SubnetworkDraftModel
from ssd.models.target_wrapper import TargetWrapper
from ssd.runtime import checkpoint_path, input_device
from ssd.training.common import MetricLogger, make_optimizer
from ssd.training.losses import feature_loss

log = logging.getLogger("ssd")


class _TeacherForced(nn.Module):
    def __init__(self, draft: SubnetworkDraftModel):
        super().__init__()
        self.draft = draft

    def forward(self, taps: dict[int, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {s.name: self.draft.bridges[s.name](taps[s.src_boundary]) for s in self.draft.bridge_specs}


class _Chained(nn.Module):
    def __init__(self, draft: SubnetworkDraftModel):
        super().__init__()
        self.draft = draft

    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.draft(input_ids, compute_logits=False, output_bridge_states=True).bridge_states


def _activation_batches(cfg, base, tokenizer, boundaries, batch_size, rank, world) -> Iterator[tuple[torch.Tensor, dict]]:
    s1 = cfg.training.stage1
    if s1.activation_cache:
        from ssd.data.extract_activations import cached_batches

        yield from cached_batches(s1.activation_cache, batch_size, boundaries)
        return
    tw = TargetWrapper(base)
    dev = input_device(base)
    for ids in batched(packed_rows(cfg.data, tokenizer, rank, world), batch_size):
        ids = ids.to(dev)
        yield ids, tw(ids, boundaries=boundaries, compute_logits=False).boundaries


def train_stage1(
    cfg: SSDConfig,
    base: nn.Module,
    tokenizer,
    max_steps: Optional[int] = None,
    accelerator=None,
) -> Optional[Path]:
    s1 = cfg.training.stage1
    torch.manual_seed(cfg.training.seed)
    layers = cfg.draft_layers
    name = profile_name(layers)
    draft = SubnetworkDraftModel(base, layers, cfg.bridge)
    if not draft.bridge_specs:
        log.info("[stage1 %s] no bridges (layers are contiguous); nothing to train", name)
        return None
    if s1.mode not in ("teacher", "chained"):
        raise ValueError(f"stage1.mode must be teacher|chained, got {s1.mode!r}")

    module: nn.Module = _TeacherForced(draft) if s1.mode == "teacher" else _Chained(draft)
    steps = max_steps or s1.steps
    opt, sched = make_optimizer(draft.parameters(), s1.lr, s1.weight_decay, s1.warmup_steps, steps)
    rank, world, is_main = 0, 1, True
    if accelerator is not None:
        module, opt = accelerator.prepare(module, opt)
        rank, world, is_main = accelerator.process_index, accelerator.num_processes, accelerator.is_main_process

    boundaries = sorted({b for s in draft.bridge_specs for b in (s.src_boundary, s.tgt_boundary)})
    batches = _activation_batches(cfg, base, tokenizer, boundaries, s1.batch_size, rank, world)
    n_params = sum(p.numel() for p in draft.parameters())
    if is_main:
        log.info("[stage1 %s] %d bridges, %.1fM params, mode=%s, %d steps", name, len(draft.bridge_specs), n_params / 1e6, s1.mode, steps)
    metrics = MetricLogger(f"[stage1 {name}]", cfg.training.log_every, is_main)
    out = checkpoint_path(cfg, 1)

    module.train()
    for step in range(steps):
        ids, taps = next(batches)
        preds = module(taps) if s1.mode == "teacher" else module(ids.to(input_device(base)))
        total = 0.0
        stats: dict[str, float] = {}
        for spec in draft.bridge_specs:
            pred = preds[spec.name]
            tgt = taps[spec.tgt_boundary].to(pred.device)
            mask = torch.ones(pred.shape[:2], dtype=torch.bool, device=pred.device)
            mask[:, : s1.skip_first_positions] = False
            loss, st = feature_loss(pred, tgt, s1.alpha_mse, s1.beta_cos, mask)
            total = total + loss
            stats[f"{spec.name}.cos"] = st["cos"]
            stats[f"{spec.name}.rel_mse"] = st["rel_mse"]
        stats = {"loss": total.item(), **stats}

        opt.zero_grad(set_to_none=True)
        if accelerator is not None:
            accelerator.backward(total)
        else:
            total.backward()
        torch.nn.utils.clip_grad_norm_(draft.parameters(), s1.grad_clip)
        opt.step()
        sched.step()
        metrics.update(step, steps, stats, ids.numel() * world, sched.get_last_lr()[0])

        if is_main and ((step + 1) % cfg.training.save_every == 0 or step + 1 == steps):
            draft.save_bridges(str(out), stage=1, step=step + 1)
    if is_main:
        log.info("[stage1 %s] saved %s", name, out)
    return out
