"""Wraps the frozen target model and taps residual-stream "boundaries".

Boundary ``i`` (0 <= i < L) is the hidden state entering decoder layer ``i``;
boundary ``L`` is the pre-norm state entering the final norm. (HF's
``output_hidden_states`` would give the *post*-norm state for the last entry,
which is the wrong regression target, so we use forward pre-hooks instead and
only keep the boundaries asked for.)
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn as nn


@dataclass
class TargetOutput:
    logits: Optional[torch.Tensor]
    boundaries: dict[int, torch.Tensor]


class TargetWrapper(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.num_layers = base_model.config.num_hidden_layers

    def _boundary_module(self, i: int) -> nn.Module:
        inner = self.base_model.model
        return inner.layers[i] if i < self.num_layers else inner.norm

    @contextmanager
    def _tap(self, boundaries: Iterable[int], store: dict[int, torch.Tensor]):
        handles = []
        for i in sorted(set(boundaries)):
            if not 0 <= i <= self.num_layers:
                raise ValueError(f"boundary {i} out of range [0, {self.num_layers}]")

            def hook(_mod, args, kwargs, i=i):
                store[i] = args[0] if args else kwargs["hidden_states"]

            handles.append(self._boundary_module(i).register_forward_pre_hook(hook, with_kwargs=True))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        boundaries: Iterable[int] = (),
        attention_mask: Optional[torch.Tensor] = None,
        compute_logits: bool = True,
    ) -> TargetOutput:
        store: dict[int, torch.Tensor] = {}
        with self._tap(boundaries, store):
            if compute_logits:
                out = self.base_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                logits = out.logits
            else:
                self.base_model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                logits = None
        return TargetOutput(logits=logits, boundaries=store)
