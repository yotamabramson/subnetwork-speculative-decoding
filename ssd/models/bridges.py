"""Learnable transition bridges that map the output of one frozen base layer
onto the input distribution expected by the next (non-adjacent) one.

    out = x + MLP(RMSNorm(x))

The output projection is initialised with a tiny std (default 1e-3), so an
untrained bridge is ~identity and the draft initially behaves like plain
layer-skipping. Inner projections use a standard fan-in init: initialising
*every* layer at 1e-3 would make gradients into the first projection scale with
the product of the downstream weights (~1e-3 per layer) and stall training.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ssd.config import BridgeConfig

_ACTIVATIONS = {"silu": nn.SiLU, "gelu": nn.GELU, "relu": nn.ReLU}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


class _PlainMLP(nn.Module):
    def __init__(self, d_model: int, hidden: int, num_layers: int, activation: str, bias: bool):
        super().__init__()
        if num_layers not in (2, 3):
            raise ValueError(f"mlp bridge supports 2 or 3 layers, got {num_layers}")
        act = _ACTIVATIONS[activation]
        dims = [d_model] + [hidden] * (num_layers - 1)
        layers: list[nn.Module] = []
        for i in range(num_layers - 1):
            layers += [nn.Linear(dims[i], dims[i + 1], bias=bias), act()]
        self.body = nn.Sequential(*layers)
        self.out_proj = nn.Linear(hidden, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.body(x))


class _SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int, bias: bool):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden, bias=bias)
        self.up_proj = nn.Linear(d_model, hidden, bias=bias)
        self.out_proj = nn.Linear(hidden, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransitionBridge(nn.Module):
    """Residual MLP bridge: ``out = x + MLP(RMSNorm(x))``.

    Parameters are kept in ``dtype`` (fp32 by default) regardless of the base
    model's dtype; inputs are cast in and the result cast back.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        mlp_type: str = "mlp",
        activation: str = "silu",
        bias: bool = True,
        init_std: float = 1e-3,
        norm_eps: float = 1e-5,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        hidden = hidden_dim or d_model
        self.d_model = d_model
        self.norm = RMSNorm(d_model, eps=norm_eps)
        if mlp_type == "mlp":
            if activation not in _ACTIVATIONS:
                raise ValueError(f"unknown activation {activation!r}")
            self.mlp = _PlainMLP(d_model, hidden, num_layers, activation, bias)
        elif mlp_type == "swiglu":
            if num_layers != 2:
                raise ValueError("swiglu bridge is a fixed gate/up + down block; set num_layers=2")
            self.mlp = _SwiGLU(d_model, hidden, bias)
        else:
            raise ValueError(f"unknown mlp_type {mlp_type!r}")
        self.reset_parameters(init_std)
        self.to(dtype)

    @classmethod
    def from_config(cls, d_model: int, cfg: BridgeConfig) -> "TransitionBridge":
        return cls(
            d_model=d_model,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            mlp_type=cfg.mlp_type,
            activation=cfg.activation,
            bias=cfg.bias,
            init_std=cfg.init_std,
            norm_eps=cfg.norm_eps,
            dtype=getattr(torch, cfg.dtype),
        )

    @torch.no_grad()
    def reset_parameters(self, init_std: float = 1e-3) -> None:
        for name, module in self.mlp.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if name == "out_proj":
                nn.init.normal_(module.weight, std=init_std) if init_std > 0 else module.weight.zero_()
            else:
                nn.init.normal_(module.weight, std=1.0 / math.sqrt(module.in_features))
            if module.bias is not None:
                module.bias.zero_()
        self.norm.weight.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        p = self.norm.weight
        x = x.to(device=p.device, dtype=p.dtype)
        return (x + self.mlp(self.norm(x))).to(in_dtype)
