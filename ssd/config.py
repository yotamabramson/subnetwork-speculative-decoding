"""YAML-backed configuration for SSD.

One YAML per target model. ``draft_layers`` is the 0-based subset of base
layers that forms the draft sub-network (used for every draft step), and
``drafting`` describes the speculation tree.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional, Union

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
    torch_dtype: str = "bfloat16"
    device_map: Optional[str] = None  # e.g. "auto" to shard across GPUs; None -> single `device`
    device: str = "auto"  # "auto" -> cuda > mps > cpu
    attn_implementation: str = "sdpa"


@dataclass
class DataConfig:
    dataset: str = "HuggingFaceH4/ultrachat_200k"
    split: str = "train_sft"
    messages_field: str = "messages"  # chat-format column; rendered with the tokenizer's chat template
    local_path: Optional[str] = None  # .jsonl with a messages/text field, or .txt; overrides `dataset`
    seq_len: int = 1024  # conversations are packed into fixed-length rows
    skip_samples: int = 0  # skip this many conversations first (e.g. to hold out an eval slice)


@dataclass
class Stage1Config:
    steps: int = 2000
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_steps: int = 100
    grad_clip: float = 1.0
    alpha_mse: float = 1.0
    beta_cos: float = 1.0
    # "teacher": each bridge sees the target's own activation at its source boundary.
    # "chained": bridges run inside the draft, so errors compound as they will at inference.
    mode: str = "teacher"
    skip_first_positions: int = 1  # position 0 is the attention sink (massive activations)
    activation_cache: Optional[str] = None  # dir written by ssd.data.extract_activations


@dataclass
class Stage2Config:
    steps: int = 2000
    batch_size: int = 4
    lr: float = 3e-5
    weight_decay: float = 0.0
    warmup_steps: int = 100
    grad_clip: float = 1.0
    temperature: float = 2.0
    kd_weight: float = 1.0
    ce_weight: float = 0.1


IntOrList = Union[int, list[int]]


@dataclass
class DraftingConfig:
    """Speculation tree. Per-depth values accept an int (same at every depth)
    or a list of length ``depth``.

    Each round the tree grows ``depth`` levels. At level d, every kept node from
    level d-1 proposes ``branch[d]`` children (its top tokens under the draft,
    or samples without replacement when temperature > 0). Of those, the
    ``width[d]`` with the highest cumulative draft log-prob are kept. Finally
    the best ``max_nodes`` nodes overall go to the target for verification.
    ``branch: 1`` is a plain chain of ``depth`` tokens.
    """

    depth: int = 4  # K
    branch: IntOrList = 1
    width: Optional[IntOrList] = None  # None -> no per-level cap
    max_nodes: Optional[int] = None  # None -> verify every node
    temperature: float = 0.0  # 0 -> greedy (exact-match verification)
    top_p: float = 1.0
    # Draft scores only the N most frequent training tokens (None = full vocab).
    # Verification always uses the full vocab, so outputs are unaffected.
    draft_vocab: Optional[int] = None

    def per_depth(self, value: Optional[IntOrList], name: str) -> list[Optional[int]]:
        if value is None or isinstance(value, int):
            return [value] * self.depth
        if len(value) != self.depth:
            raise ValueError(f"drafting.{name} has {len(value)} entries but depth={self.depth}")
        return list(value)

    def validate(self) -> None:
        if self.depth < 1:
            raise ValueError("drafting.depth must be >= 1")
        if any(b is None or b < 1 for b in self.per_depth(self.branch, "branch")):
            raise ValueError("drafting.branch entries must be >= 1")
        if any(w is not None and w < 1 for w in self.per_depth(self.width, "width")):
            raise ValueError("drafting.width entries must be >= 1")
        if self.max_nodes is not None and self.max_nodes < 1:
            raise ValueError("drafting.max_nodes must be >= 1")
        if self.draft_vocab is not None and self.draft_vocab < 1:
            raise ValueError("drafting.draft_vocab must be >= 1 (or null)")
        if not 0.0 < self.top_p <= 1.0 or self.temperature < 0:
            raise ValueError("need temperature >= 0 and 0 < top_p <= 1")


@dataclass
class OnlineConfig:
    """Online self-distillation (``ssd-train --stage online``): the target generates
    an endless stream from one fixed prompt and the bridges train on it as it is
    produced. Nothing is stored."""

    prompt: str = "Tell me about something you find fascinating, in detail."
    streams: int = 32  # parallel sampled streams (all from the same prompt)
    temperature: float = 0.8
    top_p: float = 0.95  # applied within the top 64 tokens
    chunk: int = 128  # tokens generated per stream between training passes (= loss positions)
    context: int = 384  # extra preceding positions per training row (keys only, no loss)
    max_context: int = 1024  # stream length that triggers a slide
    keep_on_slide: int = 256  # tokens kept (re-prefilled at position 0) when sliding
    loop_window: int = 64  # loop detector: look at a stream's last N tokens...
    loop_min_distinct: float = 0.25  # ...and re-prompt it if fewer than this fraction are distinct
    micro_batch: int = 8  # stream rows per optimizer step
    lr: float = 5e-5
    weight_decay: float = 0.0
    warmup_steps: int = 50
    grad_clip: float = 1.0
    temperature_kd: float = 2.0
    kd_weight: float = 1.0
    ce_weight: float = 0.1
    save_every: int = 200  # optimizer steps


@dataclass
class TrainingConfig:
    output_dir: str = "outputs/default"
    seed: int = 0
    log_every: int = 20
    save_every: int = 500
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    online: OnlineConfig = field(default_factory=OnlineConfig)


def _build(cls, raw: Optional[dict], where: str):
    raw = raw or {}
    names = {f.name for f in fields(cls)}
    unknown = set(raw) - names
    if unknown:
        raise ValueError(f"unknown keys in {where}: {sorted(unknown)}")
    return cls(**raw)


def profile_name(layers: list[int]) -> str:
    return "L" + "-".join(str(i) for i in layers)


@dataclass
class SSDConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    draft_layers: list[int] = field(default_factory=lambda: [0, 15, 31])
    bridge: BridgeConfig = field(default_factory=BridgeConfig)
    drafting: DraftingConfig = field(default_factory=DraftingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SSDConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown top-level config sections: {sorted(unknown)}")
        layers = raw.get("draft_layers")
        if not isinstance(layers, list) or not all(isinstance(i, int) for i in layers):
            raise ValueError("draft_layers must be a list of 0-based layer indices, e.g. [0, 7, 15]")
        tr = dict(raw.get("training") or {})
        training = _build(
            TrainingConfig,
            {
                **{k: v for k, v in tr.items() if k not in ("stage1", "stage2", "online")},
                "stage1": _build(Stage1Config, tr.get("stage1"), "training.stage1"),
                "stage2": _build(Stage2Config, tr.get("stage2"), "training.stage2"),
                "online": _build(OnlineConfig, tr.get("online"), "training.online"),
            },
            "training",
        )
        drafting = _build(DraftingConfig, raw.get("drafting"), "drafting")
        drafting.validate()
        return cls(
            model=_build(ModelConfig, raw.get("model"), "model"),
            draft_layers=layers,
            bridge=_build(BridgeConfig, raw.get("bridge"), "bridge"),
            drafting=drafting,
            data=_build(DataConfig, raw.get("data"), "data"),
            training=training,
        )

    def validate_against(self, num_hidden_layers: int) -> None:
        validate_layer_indices(self.draft_layers, num_hidden_layers)

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
