"""Training batches for both data formats.

Each optimizer step gets a list of *groups* (micro-batches, processed one at a
time with gradient accumulation). Each group has:

* ``ids``   [b, T] tokens
* ``pred``  [b, T] bool: position t trains next-token prediction (label = ids[t+1])
* ``feat``  [b, T] bool: position t's hidden state trains feature regression

Formats (``data.format``):

* ``packed``: our original packed text rows (every position counts, except the
  last one, which has no label).
* ``eagle3``: EAGLE-3's data, one conversation per row (no padding, variable T).
  Only assistant tokens count: ``pred[t] = loss_mask[t+1]`` (the predicted token
  is an assistant token) and ``feat[t] = loss_mask[t]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import torch

from ssd.config import SSDConfig


@dataclass
class Group:
    ids: torch.Tensor
    pred: torch.Tensor
    feat: torch.Tensor

    def to(self, device) -> "Group":
        return Group(self.ids.to(device), self.pred.to(device), self.feat.to(device))

    @property
    def n_pred(self) -> int:
        return int(self.pred.sum())


def training_steps(
    cfg: SSDConfig,
    tokenizer,
    batch_size: int,
    micro_batch: Optional[int],
    rank: int = 0,
    world: int = 1,
) -> Iterator[list[Group]]:
    """Yield, per optimizer step, the list of micro-batch groups."""
    if cfg.data.format == "eagle3":
        from ssd.data.eagle3_data import iter_examples

        if not cfg.data.local_path:
            raise ValueError("data.format=eagle3 needs data.local_path (regenerated EAGLE-3 jsonl)")
        rows = iter_examples(cfg.data.local_path, tokenizer, cfg.data.max_len, rank=rank, world=world,
                             skip=cfg.data.skip_samples)
        while True:
            step = []
            for _ in range(batch_size):
                ids, mask = next(rows)
                pred = torch.zeros_like(mask, dtype=torch.bool)
                pred[:, :-1] = mask[:, 1:].bool()
                step.append(Group(ids, pred, mask.bool()))
            yield step
    elif cfg.data.format == "packed":
        from ssd.data.text_stream import batched, packed_rows

        mb = micro_batch or batch_size
        for ids in batched(packed_rows(cfg.data, tokenizer, rank, world), batch_size):
            groups = []
            for r0 in range(0, ids.shape[0], mb):
                g = ids[r0 : r0 + mb]
                pred = torch.ones_like(g, dtype=torch.bool)
                pred[:, -1] = False
                groups.append(Group(g, pred, torch.ones_like(g, dtype=torch.bool)))
            yield groups
    else:
        raise ValueError(f"unknown data.format {cfg.data.format!r} (packed | eagle3)")
