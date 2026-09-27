"""Online self-distillation: train the bridges on the target's own endless output.

The target samples ``streams`` parallel continuations of one fixed prompt, and
keeps going forever. When a stream ends its turn, the same prompt is inserted
again as a new user turn and the target answers it again, in the same
conversation. (Left to free-run past end-of-turn, streams fall into an
absorbing loop of role-header tokens.) A stream stuck in any other loop (low
distinct-token ratio) gets the same treatment. Inserted tokens aren't target
samples, so there's no loss on predicting them. While generating, the target
records at every position what training needs: its input to each draft layer (the keys/values the draft sees at
inference) and its final hidden state (-> KD target logits). So there is no
second target pass and no stored dataset.

Every ``chunk`` generated tokens, the bridges train on those new positions
(loss), with up to ``context`` preceding positions supplying the target's
keys/values, which is exactly the inference situation. When a stream reaches
``max_context``, it slides: the last ``keep_on_slide`` tokens are re-prefilled
at position 0, and generation continues.

    ssd-train --config ... --stage online --hours 10 --init outputs/.../stage2.pt
"""

from __future__ import annotations

import logging
import signal
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from ssd.config import SSDConfig, profile_name
from ssd.data.generate_selfdistill import sample_next
from ssd.engine.speculative import TargetRunner, _sync
from ssd.models.subnetwork_draft import SubnetworkDraftModel
from ssd.runtime import checkpoint_path, device_memory_gb, free_device_memory, input_device
from ssd.training.common import MetricLogger
from ssd.training.losses import chunked_distill_loss

log = logging.getLogger("ssd")


class _Streams:
    """Preallocated per-stream buffers of tokens and target activations."""

    def __init__(self, B: int, max_len: int, d: int, layers: list[int], dtype, device):
        self.tokens = torch.zeros(B, max_len, dtype=torch.long, device=device)
        self.final = torch.zeros(B, max_len, d, dtype=dtype, device=device)  # post-norm hidden
        self.taps = {i: torch.zeros(B, max_len, d, dtype=dtype, device=device) for i in layers}
        self.sampled = torch.zeros(B, max_len, dtype=torch.bool, device=device)  # token came from the target's sampling
        self.len = 0

    def write(self, tokens: torch.Tensor, out, sampled: torch.Tensor) -> None:
        n = tokens.shape[1]
        sl = slice(self.len, self.len + n)
        self.tokens[:, sl] = tokens
        self.sampled[:, sl] = sampled
        self.final[:, sl] = out.hidden_states
        for i, t in out.layer_inputs.items():
            self.taps[i][:, sl] = t
        self.len += n


def train_online(
    cfg: SSDConfig,
    base: nn.Module,
    tokenizer,
    init_from: Optional[str | Path] = None,
    hours: Optional[float] = None,
    max_steps: Optional[int] = None,
    seed: int = 0,
) -> Path:
    oc = cfg.training.online
    torch.manual_seed(seed)
    layers = cfg.draft_layers
    name = profile_name(layers)
    dev = input_device(base)
    dtype = base.model.embed_tokens.weight.dtype
    draft = SubnetworkDraftModel(base, layers, cfg.bridge)
    if init_from:
        draft.load_bridges(str(init_from))
        log.info("[online %s] warm start from %s", name, init_from)
    target = TargetRunner(base)
    out_path = checkpoint_path(cfg, "online")

    total = max_steps  # None: open-ended (stop on --hours or SIGINT/SIGTERM)
    opt = torch.optim.AdamW(draft.parameters(), lr=oc.lr, weight_decay=oc.weight_decay, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, oc.warmup_steps)))  # then constant
    metrics = MetricLogger(f"[online {name}]", cfg.training.log_every)

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": oc.prompt}], add_generation_prompt=True, return_dict=True
    )["input_ids"]
    B = oc.streams
    streams = _Streams(B, oc.max_context + 1, base.config.hidden_size, layers, dtype, dev)
    want = set(layers)
    eos = base.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    reprompt = tokenizer(
        f"<|start_header_id|>user<|end_header_id|>\n\n{oc.prompt}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
        add_special_tokens=False,
    )["input_ids"]
    forced: list[list[int]] = [[] for _ in range(B)]  # tokens to insert next, per stream
    n_reprompts = n_loop_resets = 0

    def next_tokens(sampled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply pending insertions; queue a re-prompt after end-of-turn."""
        nonlocal n_reprompts
        toks = sampled[:, 0].tolist()
        flags = [True] * B
        for b in range(B):
            if forced[b]:
                toks[b], flags[b] = forced[b].pop(0), False
            elif toks[b] in eos_ids:
                forced[b] = list(reprompt)
                n_reprompts += 1
        return (torch.tensor(toks, device=dev).unsqueeze(1),
                torch.tensor(flags, device=dev).unsqueeze(1))

    stop = {"flag": False}
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.__setitem__("flag", True))

    @torch.no_grad()
    def prefill(ids: torch.Tensor, sampled: torch.Tensor):
        cache = target.new_cache(oc.max_context + 1)
        streams.len = 0
        out = target(ids, past_key_values=cache, output_layer_inputs=want)
        streams.write(ids, out, sampled)
        return cache, sample_next(out.logits[:, -1], oc.temperature, oc.top_p)

    prompt_ids = torch.tensor([prompt] * B, device=dev)
    cache, first = prefill(prompt_ids, torch.zeros_like(prompt_ids, dtype=torch.bool))
    pending, pending_sampled = next_tokens(first)
    trained_upto = streams.len  # no loss on the prompt itself
    step, gen_tokens, t_start = 0, 0, time.perf_counter()
    gen_time = 0.0
    log.info("[online %s] %d streams, chunk=%d, context=%d, prompt=%r", name, B, oc.chunk, oc.context, oc.prompt)

    def running() -> bool:
        return (not stop["flag"] and (total is None or step < total)
                and (hours is None or time.perf_counter() - t_start < hours * 3600))

    while running():
        # ---- generate one chunk per stream
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(oc.chunk):
                if streams.len >= oc.max_context:  # slide: keep the tail, re-prefill at position 0
                    tail = streams.tokens[:, streams.len - oc.keep_on_slide : streams.len].clone()
                    tail_sampled = streams.sampled[:, streams.len - oc.keep_on_slide : streams.len].clone()
                    cache = None  # drop the old cache before allocating the new one
                    free_device_memory()
                    cache, _ = prefill(tail, tail_sampled)
                    free_device_memory()  # the prefill's large temporaries would otherwise stay cached
                    trained_upto = streams.len
                out = target(pending, past_key_values=cache, output_layer_inputs=want)
                streams.write(pending, out, pending_sampled)
                pending, pending_sampled = next_tokens(sample_next(out.logits[:, -1], oc.temperature, oc.top_p))
                gen_tokens += B
            _sync(dev)  # the GPU runs async: settle before timing
        gen_time += time.perf_counter() - t0

        # ---- loop detector: a stream whose recent tokens are mostly repeats is re-prompted
        if streams.len >= oc.loop_window:
            recent = streams.tokens[:, streams.len - oc.loop_window : streams.len].tolist()
            for b in range(B):
                if not forced[b] and len(set(recent[b])) / oc.loop_window < oc.loop_min_distinct:
                    forced[b] = [eot] + list(reprompt)
                    n_loop_resets += 1

        # ---- train on the new positions [trained_upto, len)
        L = streams.len
        q0 = max(trained_upto, L - oc.chunk)
        if L - q0 < 2:
            continue
        s0 = max(0, q0 - oc.context)
        labels_all = torch.cat([streams.tokens[:, 1:L], pending], 1)  # label[t] = token t+1
        # Loss only where the next token was sampled by the target (not an inserted prompt).
        valid_all = torch.cat([streams.sampled[:, 1:L], pending_sampled], 1)
        chunk_tokens = streams.tokens[:, q0:L].tolist()
        distinct = sum(len(set(r)) / len(r) for r in chunk_tokens) / B  # per-stream, then averaged
        draft.train()
        for r0 in range(0, B, oc.micro_batch):
            rows = slice(r0, r0 + oc.micro_batch)
            ids = streams.tokens[rows, q0:L]
            pos = torch.arange(q0, L, device=dev).unsqueeze(0).expand(ids.shape[0], -1)
            true_in = {i: streams.taps[i][rows, s0:L] for i in layers}
            valid = valid_all[rows, q0:L]
            if not bool(valid.any()):
                continue
            d_hidden = draft(ids, position_ids=pos, compute_logits=False, true_layer_inputs=true_in).hidden_states
            loss, stats = chunked_distill_loss(
                base.lm_head,
                d_hidden[valid],
                streams.final[rows, q0:L][valid],
                labels_all[rows, q0:L][valid],
                oc.temperature_kd,
                oc.kd_weight,
                oc.ce_weight,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(draft.parameters(), oc.grad_clip)
            opt.step()
            sched.step()
            stats = {"loss": loss.item(), **stats, "distinct": distinct,
                     "masked": 1 - valid.float().mean().item(), "reprompts": n_reprompts,
                     "loop_resets": n_loop_resets,
                     "gen_tok_s": gen_tokens / max(gen_time, 1e-6), "mem_gb": device_memory_gb(dev)}
            metrics.update(step, total, stats, ids.numel(), sched.get_last_lr()[0])
            step += 1
            if step % oc.save_every == 0:
                draft.save_bridges(str(out_path), stage="online", step=step, tokens=gen_tokens)
            if step % (cfg.training.log_every * 10) == 0:
                snippet = tokenizer.decode(streams.tokens[0, max(0, L - 60) : L].tolist())
                log.info("[online %s] stream 0 (pos %d): ...%r", name, L, snippet)
            if not running():
                break
        trained_upto = L
        free_device_memory()  # row shapes vary pass to pass; don't let the allocator cache pile up

    draft.save_bridges(str(out_path), stage="online", step=step, tokens=gen_tokens)
    log.info("[online %s] stopped after %d steps, %d generated tokens; saved %s", name, step, gen_tokens, out_path)
    return out_path
