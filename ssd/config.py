"""YAML-backed configuration for SSD.

Only the ``model`` and ``bridge`` sections are typed; ``training`` and
``drafting`` are passed through as dicts and consumed by their respective
entry points.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class BridgeConfig:
    mlp_type: str = "mlp"  # "mlp" | "swiglu"
    num_layers: int = 2  # linear layers in the MLP ("mlp" supports 2-3; "swiglu" is fixed at 2)
    hidden_dim: Optional[int] = None  # None -> d_model; smaller = bottleneck, larger = expansion
    activation: str = "silu"  # "silu" | "gelu" | "relu" (ignored for swiglu, which always uses silu)
    bias: bool = True
    init_std: float = 1e-3  # std of the output projection -> bridge starts as ~identity
    norm_eps: float = 1e-5
    dtype: str = "float32"  # bridge parameter dtype; inputs are cast in/out


@dataclass
class ModelConfig:
    base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    # 0-indexed decoder layer indices, strictly increasing.
    layer_indices: list[int] = field(default_factory=lambda: [0, 15, 31])
    torch_dtype: str = "bfloat16"
    device_map: Optional[str] = None
    attn_implementation: str = "sdpa"


@dataclass
class SSDConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    bridge: BridgeConfig = field(default_factory=BridgeConfig)
    training: dict[str, Any] = field(default_factory=dict)
    drafting: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SSDConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        unknown = set(raw) - {"model", "bridge", "training", "drafting"}
        if unknown:
            raise ValueError(f"Unknown top-level config sections: {sorted(unknown)}")
        return cls(
            model=ModelConfig(**raw.get("model", {})),
            bridge=BridgeConfig(**raw.get("bridge", {})),
            training=raw.get("training", {}) or {},
            drafting=raw.get("drafting", {}) or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_layer_indices(layer_indices: list[int], num_hidden_layers: int) -> list[int]:
    idx = [int(i) for i in layer_indices]
    if not idx:
        raise ValueError("layer_indices must be non-empty")
    if any(b <= a for a, b in zip(idx, idx[1:])):
        raise ValueError(f"layer_indices must be strictly increasing, got {idx}")
    if idx[0] < 0 or idx[-1] >= num_hidden_layers:
        raise ValueError(
            f"layer_indices {idx} out of range for a model with {num_hidden_layers} layers "
            f"(valid: 0..{num_hidden_layers - 1}; indices are 0-based)"
        )
    return idx
