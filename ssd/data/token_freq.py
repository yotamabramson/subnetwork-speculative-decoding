"""Token frequencies over the training data, for the draft's reduced vocabulary.

The draft's lm_head is a large share of each draft step (128k x d for
Llama-3). Scoring only the most frequent tokens (``drafting.draft_vocab``) cuts
that cost, as EAGLE-3 does. Verification still uses the full vocabulary, so
outputs are unchanged; tokens outside the subset simply can't be drafted.

    python -m ssd.data.token_freq --config configs/llama32_1b_subnetwork.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from ssd.config import SSDConfig
from ssd.data.text_stream import packed_rows

log = logging.getLogger("ssd")


def token_freq_path(cfg: SSDConfig) -> Path:
    return Path(cfg.training.output_dir) / "token_freq.pt"


def count_token_freq(cfg: SSDConfig, tokenizer, num_tokens: int = 2_000_000) -> torch.Tensor:
    counts = torch.zeros(len(tokenizer), dtype=torch.long)
    seen = 0
    for row in packed_rows(cfg.data, tokenizer, repeat=False):
        counts += torch.bincount(row, minlength=counts.numel())[: counts.numel()]
        seen += row.numel()
        if seen >= num_tokens:
            break
    return counts


def ensure_token_freq(cfg: SSDConfig, tokenizer, num_tokens: int = 2_000_000) -> Path:
    path = token_freq_path(cfg)
    if not path.exists():
        log.info("counting token frequencies over %d tokens -> %s", num_tokens, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(count_token_freq(cfg, tokenizer, num_tokens), path)
    return path


def top_vocab(counts: torch.Tensor, size: int, always: set[int] = frozenset()) -> torch.Tensor:
    """The ``size`` most frequent token ids (always including ``always``), sorted ascending."""
    order = counts.argsort(descending=True, stable=True)
    forced = [t for t in always if 0 <= t < counts.numel()]
    keep = torch.tensor(forced, dtype=torch.long)
    rest = order[~torch.isin(order, keep)][: max(0, size - len(forced))]
    return torch.cat([keep, rest]).sort().values


def main():
    from ssd.runtime import load_tokenizer, setup_env

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--tokens", type=int, default=2_000_000)
    args = p.parse_args()
    setup_env()
    cfg = SSDConfig.from_yaml(args.config)
    path = token_freq_path(cfg)
    path.unlink(missing_ok=True)
    ensure_token_freq(cfg, load_tokenizer(cfg), args.tokens)
    counts = torch.load(path)
    for size in (8192, 16384, 32768, 65536):
        cover = counts[counts.argsort(descending=True)[:size]].sum() / counts.sum()
        log.info("top %6d tokens cover %.2f%% of training tokens", size, 100 * cover.item())


if __name__ == "__main__":
    main()
