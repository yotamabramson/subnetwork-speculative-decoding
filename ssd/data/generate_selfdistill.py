"""Self-distillation data: the target answers dataset prompts itself.

Training the draft on the target's *own* responses (instead of the dataset's
human-written ones) teaches it the target's style, which is what verification
rewards (Medusa and EAGLE both report higher acceptance). Output is a
.jsonl of ``{"messages": [user, assistant]}`` that ``data.local_path`` can
point at.

Generation uses our own batched sampler (``TargetRunner`` + ``KVCache``),
which on MPS is several times faster than HF ``generate``. Prompts are
left-padded, so each row gets its own positions and an attention mask that
hides the padding.

    python -m ssd.data.generate_selfdistill --config configs/llama32_1b_subnetwork.yaml \\
        --out data/selfdistill_1b.jsonl --tokens 3000000

Resumable: rerunning skips the prompts already written to ``--out``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Iterator, Optional

import torch

from ssd.config import SSDConfig
from ssd.engine.speculative import TargetRunner
from ssd.engine.tree_drafter import warp_probs

log = logging.getLogger("ssd")


def iter_prompts(cfg: SSDConfig, tokenizer, max_prompt_tokens: int) -> Iterator[tuple[str, list[int]]]:
    """(user text, chat-formatted prompt ids) for each dataset example's first user turn."""
    from datasets import load_dataset

    for ex in load_dataset(cfg.data.dataset, split=cfg.data.split, streaming=True):
        msgs = ex.get(cfg.data.messages_field) or []
        if not msgs or msgs[0].get("role") != "user":
            continue
        ids = tokenizer.apply_chat_template([msgs[0]], add_generation_prompt=True, return_dict=True)["input_ids"]
        if len(ids) <= max_prompt_tokens:
            yield msgs[0]["content"], ids


def sample_next(logits: torch.Tensor, temperature: float, top_p: float, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """[B, V] logits -> [B, 1] tokens. Greedy at temperature 0; otherwise top-k=64,
    then temperature + top-p within those (a full-vocab nucleus sort over 128k
    tokens is needlessly expensive)."""
    if temperature == 0:
        return logits.argmax(-1, keepdim=True)
    val, idx = logits.float().topk(64, dim=-1)
    probs = warp_probs(val, temperature, top_p)
    choice = torch.multinomial(probs.cpu() if generator is not None else probs, 1, generator=generator)
    return idx.gather(-1, choice.to(logits.device))


@torch.no_grad()
def generate_batch(
    target: TargetRunner,
    prompts: list[list[int]],
    max_new_tokens: int,
    eos_ids: set[int],
    temperature: float,
    top_p: float,
    pad_id: int,
    generator: Optional[torch.Generator] = None,
) -> list[list[int]]:
    """Sample continuations for a batch of prompts (left-padded). Returns new
    tokens per row, cut after the first EOS (inclusive)."""
    dev = target.base.embed_tokens.weight.device
    B, T = len(prompts), max(len(p) for p in prompts)
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    valid = torch.zeros(B, T, dtype=torch.bool)
    for b, p in enumerate(prompts):
        ids[b, T - len(p):] = torch.tensor(p)
        valid[b, T - len(p):] = True
    ids, valid = ids.to(dev), valid.to(dev)
    pos = (valid.long().cumsum(-1) - 1).clamp_min(0)
    causal = torch.ones(T, T, dtype=torch.bool, device=dev).tril()
    mask = (causal[None] & valid[:, None, :]) | torch.eye(T, dtype=torch.bool, device=dev)[None]  # pad rows see themselves (no NaN)

    cache = target.new_cache(T + max_new_tokens)
    logits = target(ids, position_ids=pos, past_key_values=cache, tree_attention_mask=mask, logits_to_keep=1).logits[:, -1]
    next_pos = pos[:, -1:] + 1
    out = torch.empty(B, 0, dtype=torch.long, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    eos = torch.tensor(sorted(eos_ids), device=dev)
    for step in range(max_new_tokens):
        tok = sample_next(logits, temperature, top_p, generator)
        out = torch.cat([out, tok], 1)
        done |= torch.isin(tok[:, 0], eos)
        if bool(done.all()) or step == max_new_tokens - 1:
            break
        valid = torch.cat([valid, torch.ones(B, 1, dtype=torch.bool, device=dev)], 1)
        logits = target(tok, position_ids=next_pos, past_key_values=cache, tree_attention_mask=valid[:, None, :]).logits[:, -1]
        next_pos = next_pos + 1

    rows = []
    for r in out.tolist():
        cut = next((i + 1 for i, t in enumerate(r) if t in eos_ids), len(r))
        rows.append(r[:cut])
    return rows


def main():
    from ssd.runtime import load_base_model, load_tokenizer, setup_env

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tokens", type=int, default=3_000_000, help="stop after this many generated tokens (total, incl. resumed)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--max-prompt-tokens", type=int, default=384)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9, help="applied within the top 64 tokens")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    setup_env()
    cfg = SSDConfig.from_yaml(args.config)
    base = load_base_model(cfg)
    tokenizer = load_tokenizer(cfg)
    target = TargetRunner(base)
    eos = base.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    pad_id = tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_rows, done_tokens = 0, 0
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                done_rows += 1
                done_tokens += json.loads(line)["n_tokens"]
        log.info("resuming: %d rows / %d tokens already in %s", done_rows, done_tokens, out_path)

    torch.manual_seed(args.seed + done_rows)
    prompts = iter_prompts(cfg, tokenizer, args.max_prompt_tokens)
    for _ in range(done_rows):
        next(prompts)

    t0, new_tokens = time.perf_counter(), 0
    with open(out_path, "a") as f:
        while done_tokens < args.tokens:
            # Pull a chunk and batch prompts of similar length to limit padding.
            chunk = [x for _, x in zip(range(args.batch_size * 8), prompts)]
            if not chunk:
                break
            order = sorted(range(len(chunk)), key=lambda i: len(chunk[i][1]))
            results: dict[int, list[int]] = {}
            for s in range(0, len(order), args.batch_size):
                idx = order[s : s + args.batch_size]
                rows = generate_batch(target, [chunk[i][1] for i in idx], args.max_new_tokens, eos_ids,
                                      args.temperature, args.top_p, pad_id)
                results.update(zip(idx, rows))
            for i, (text, _) in enumerate(chunk):  # write in dataset order (keeps resume exact)
                toks = results[i]
                reply = tokenizer.decode(toks, skip_special_tokens=True)
                f.write(json.dumps({"messages": [{"role": "user", "content": text},
                                                 {"role": "assistant", "content": reply}],
                                    "n_tokens": len(toks)}) + "\n")
                done_tokens += len(toks)
                new_tokens += len(toks)
            f.flush()
            dt = time.perf_counter() - t0
            log.info("%d / %d tokens (%.0f tok/s)", done_tokens, args.tokens, new_tokens / dt)


if __name__ == "__main__":
    main()
