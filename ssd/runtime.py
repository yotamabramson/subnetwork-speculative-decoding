"""Model loading, device selection and bridge-checkpoint layout shared by the
``ssd-train`` and ``ssd-run`` entry points.

Checkpoints live at ``{training.output_dir}/{profile}/stage{1,2}.pt`` where
``profile`` names the layer subset, e.g. ``L0-7-15``, so bridges trained for
one subset are never loaded into another.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from ssd.config import SSDConfig, profile_name
from ssd.models.subnetwork_draft import SubnetworkDraftModel

log = logging.getLogger("ssd")


def setup_env(level: int = logging.INFO) -> None:
    """Load ``.env`` from the working directory (e.g. HF_TOKEN) and configure logging."""
    from dotenv import load_dotenv

    load_dotenv(Path.cwd() / ".env")
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def free_device_memory() -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def device_memory_gb(device: torch.device) -> float:
    """Memory the framework holds on the device (incl. its allocator cache), in GB."""
    if device.type == "cuda":
        return torch.cuda.memory_reserved(device) / 2**30
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 2**30
    return 0.0


def load_tokenizer(cfg: SSDConfig):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(cfg.model.base_model)


def load_base_model(cfg: SSDConfig, device: Optional[torch.device] = None) -> nn.Module:
    from transformers import AutoModelForCausalLM

    kwargs = dict(dtype=getattr(torch, cfg.model.torch_dtype), attn_implementation=cfg.model.attn_implementation)
    if cfg.model.device_map is not None:
        base = AutoModelForCausalLM.from_pretrained(cfg.model.base_model, device_map=cfg.model.device_map, **kwargs)
    else:
        base = AutoModelForCausalLM.from_pretrained(cfg.model.base_model, **kwargs)
        base.to(device or resolve_device(cfg.model.device))
    base.eval().requires_grad_(False)
    cfg.validate_against(base.config.num_hidden_layers)
    return base


def input_device(base: nn.Module) -> torch.device:
    return base.model.embed_tokens.weight.device


def checkpoint_path(cfg: SSDConfig, stage: int | str) -> Path:
    """stage: 1, 2 or "online"."""
    name = f"stage{stage}.pt" if isinstance(stage, int) else f"{stage}.pt"
    return Path(cfg.training.output_dir) / profile_name(cfg.draft_layers) / name


def latest_checkpoint(cfg: SSDConfig) -> Optional[Path]:
    for stage in ("online", 2, 1):
        p = checkpoint_path(cfg, stage)
        if p.exists():
            return p
    return None


def build_draft(
    cfg: SSDConfig,
    base: nn.Module,
    checkpoint: Optional[str | Path] = "latest",
    inference: bool = False,
) -> SubnetworkDraftModel:
    """Build the draft for ``cfg.draft_layers``. ``checkpoint``: a path, "latest"
    (online > stage2 > stage1 under output_dir, else untrained with a warning), or None.
    ``inference``: cast bridges to the base dtype (they train in fp32) and apply
    ``drafting.draft_vocab``."""
    draft = SubnetworkDraftModel(base, cfg.draft_layers, cfg.bridge)
    name = profile_name(cfg.draft_layers)
    if checkpoint == "latest":
        checkpoint = latest_checkpoint(cfg)
        if checkpoint is None and draft.bridge_specs:
            log.warning("no trained bridges for %s under %s; using untrained bridges", name, cfg.training.output_dir)
    if checkpoint is not None:
        draft.load_bridges(str(checkpoint))
        log.info("loaded bridges for %s from %s", name, checkpoint)
    if inference:
        draft.bridges.to(base.model.embed_tokens.weight.dtype)
    if inference and cfg.drafting.draft_vocab:
        from ssd.data.token_freq import token_freq_path, top_vocab

        path = token_freq_path(cfg)
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run ssd-train or `python -m ssd.data.token_freq --config ...`")
        eos = base.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        draft.set_vocab_subset(top_vocab(torch.load(path), cfg.drafting.draft_vocab, eos))
        log.info("draft vocab: %d most frequent tokens", cfg.drafting.draft_vocab)
    return draft
