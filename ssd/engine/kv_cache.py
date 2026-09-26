"""Preallocated KV cache used by the draft sub-network and, during tree
verification, by the target (run through the same layer implementation).

One preallocated [B, n_kv_heads, max_len, head_dim] buffer pair per draft
layer slot (slot = position in ``layer_indices``, not the base layer index).
Buffers are allocated lazily on the first write so each slot lands on the
device of the base layer it serves (works with multi-GPU ``device_map``).

Write protocol for one forward over ``q`` new tokens:
    for each slot: k, v = cache.update(slot, k_new, v_new)   # writes [len, len+q)
    cache.advance(q)                                          # commit
"""

from __future__ import annotations

from typing import Optional

import torch


class KVCache:
    def __init__(self, num_slots: int, max_length: int):
        self.num_slots = num_slots
        self.max_length = max_length
        self.seq_len = 0
        self.k: list[Optional[torch.Tensor]] = [None] * num_slots
        self.v: list[Optional[torch.Tensor]] = [None] * num_slots

    def get_seq_length(self) -> int:
        return self.seq_len

    def update(self, slot: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, n_kv, q_len, head_dim = k.shape
        end = self.seq_len + q_len
        if end > self.max_length:
            raise RuntimeError(f"KVCache overflow: {end} > max_length={self.max_length}")
        if self.k[slot] is None:
            shape = (bsz, n_kv, self.max_length, head_dim)
            self.k[slot] = torch.empty(shape, dtype=k.dtype, device=k.device)
            self.v[slot] = torch.empty(shape, dtype=v.dtype, device=v.device)
        self.k[slot][:, :, self.seq_len : end] = k
        self.v[slot][:, :, self.seq_len : end] = v
        return self.k[slot][:, :, :end], self.v[slot][:, :, :end]

    def advance(self, n: int) -> None:
        self.seq_len += n

    def crop(self, length: int) -> None:
        """Discard everything at positions >= ``length``."""
        if length > self.seq_len:
            raise ValueError(f"cannot crop to {length} > current length {self.seq_len}")
        self.seq_len = length

    def keep_positions(self, start: int, positions: torch.Tensor) -> None:
        """Compact the tail after tree verification.

        Keeps cache entries ``positions`` (absolute indices >= ``start``, e.g.
        the accepted tree path) and moves them to ``[start, start + len)``.
        """
        n = positions.numel()
        for slot in range(self.num_slots):
            if self.k[slot] is None:
                continue
            idx = positions.to(self.k[slot].device)
            self.k[slot][:, :, start : start + n] = self.k[slot].index_select(2, idx)
            self.v[slot][:, :, start : start + n] = self.v[slot].index_select(2, idx)
        self.seq_len = start + n

    def reset(self) -> None:
        self.seq_len = 0
