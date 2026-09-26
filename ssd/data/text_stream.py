"""Packed token rows for training.

Conversations are rendered with the tokenizer's chat template, concatenated,
and cut into fixed ``seq_len`` rows. No padding, so the draft never needs a
padding mask.
"""

from __future__ import annotations

import json
from typing import Iterator, Optional

import torch

from ssd.config import DataConfig


def _iter_raw(cfg: DataConfig) -> Iterator[dict]:
    if cfg.local_path:
        with open(cfg.local_path) as f:
            if cfg.local_path.endswith(".jsonl"):
                for line in f:
                    if line.strip():
                        yield json.loads(line)
            else:
                for para in f.read().split("\n\n"):
                    if para.strip():
                        yield {"text": para}
        return
    from datasets import load_dataset

    yield from load_dataset(cfg.dataset, split=cfg.split, streaming=True)


def _render(example: dict, cfg: DataConfig, tokenizer) -> Optional[list[int]]:
    msgs = example.get(cfg.messages_field)
    if msgs:
        text = tokenizer.apply_chat_template(msgs, tokenize=False)
        return tokenizer(text, add_special_tokens=False)["input_ids"]
    if example.get("text"):
        return tokenizer(example["text"], add_special_tokens=True)["input_ids"]
    return None


def packed_rows(
    cfg: DataConfig,
    tokenizer,
    rank: int = 0,
    world_size: int = 1,
    repeat: bool = True,
) -> Iterator[torch.Tensor]:
    """Yield LongTensor rows of length ``cfg.seq_len``; rows are sharded round-robin
    across ranks. With ``repeat`` the source is cycled indefinitely."""
    buf: list[int] = []
    row_idx = 0
    while True:
        n_rows_this_pass = 0
        for i, ex in enumerate(_iter_raw(cfg)):
            if i < cfg.skip_samples:
                continue
            ids = _render(ex, cfg, tokenizer)
            if not ids:
                continue
            buf.extend(ids)
            while len(buf) >= cfg.seq_len:
                row, buf = buf[: cfg.seq_len], buf[cfg.seq_len :]
                if row_idx % world_size == rank:
                    yield torch.tensor(row, dtype=torch.long)
                row_idx += 1
                n_rows_this_pass += 1
        if not repeat:
            return
        if n_rows_this_pass == 0:
            raise ValueError(f"data source yields fewer than seq_len={cfg.seq_len} tokens")


def batched(rows: Iterator[torch.Tensor], batch_size: int) -> Iterator[torch.Tensor]:
    batch: list[torch.Tensor] = []
    for r in rows:
        batch.append(r)
        if len(batch) == batch_size:
            yield torch.stack(batch)
            batch = []
