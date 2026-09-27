"""Optimizer / schedule / logging helpers shared by both training stages."""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from typing import Optional

import torch

log = logging.getLogger("ssd")


def make_optimizer(params, lr: float, weight_decay: float, warmup: int, total: int):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))

    def lr_lambda(step: int) -> float:  # linear warmup, cosine decay to 10%
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


class MetricLogger:
    def __init__(self, prefix: str, every: int, is_main: bool = True):
        self.prefix, self.every, self.is_main = prefix, every, is_main
        self.sums: dict[str, float] = defaultdict(float)
        self.n = 0
        self.tokens = 0
        self.t0 = time.perf_counter()

    def update(self, step: int, total: Optional[int], stats: dict[str, float], tokens: int, lr: float) -> None:
        for k, v in stats.items():
            self.sums[k] += v
        self.n += 1
        self.tokens += tokens
        if (step + 1) % self.every == 0 or (total is not None and step + 1 == total):
            dt = time.perf_counter() - self.t0
            if self.is_main:
                body = " ".join(f"{k}={v / self.n:.4f}" for k, v in self.sums.items())
                of = f"/{total}" if total is not None else ""
                log.info("%s step %d%s %s lr=%.2e tok/s=%.0f", self.prefix, step + 1, of, body, lr, self.tokens / dt)
            self.sums.clear()
            self.n = self.tokens = 0
            self.t0 = time.perf_counter()
