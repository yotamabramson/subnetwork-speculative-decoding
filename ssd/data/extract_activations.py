"""Cache target-model boundary activations to disk for Stage 1.

Writes ``{out_dir}/meta.json`` and ``shard_XXXXX.safetensors`` files with
``input_ids`` [n, T] and ``b{i}`` [n, T, d] for each boundary ``i`` the
config's bridges read or regress onto.

Size: 2 bytes x d_model x #boundaries per token. For the 1B config that is
~24 KB/token, so on a laptop, online extraction during training (the default)
is usually the better choice. Use a cache when running several Stage-1 epochs
over a fixed calibration set.

    python -m ssd.data.extract_activations --config configs/llama32_1b_subnetwork.yaml \\
        --out activations/llama32_1b --rows 512
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterator

import torch
from safetensors.torch import load_file, save_file

from ssd.config import SSDConfig
from ssd.data.text_stream import batched, packed_rows
from ssd.models.subnetwork_draft import required_boundaries
from ssd.models.target_wrapper import TargetWrapper

log = logging.getLogger("ssd")


@torch.no_grad()
def extract(cfg: SSDConfig, base, tokenizer, out_dir: str, num_rows: int, rows_per_shard: int = 64, batch_size: int = 4):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    L = base.config.num_hidden_layers
    boundaries = sorted(required_boundaries(cfg.draft_layers, L))
    tw = TargetWrapper(base)
    device = base.model.embed_tokens.weight.device

    shard_ids: list[torch.Tensor] = []
    shard_acts: dict[int, list[torch.Tensor]] = {b: [] for b in boundaries}
    n_done, n_shards = 0, 0

    def flush():
        nonlocal n_shards, shard_ids, shard_acts
        if not shard_ids:
            return
        tensors = {"input_ids": torch.cat(shard_ids)}
        tensors.update({f"b{b}": torch.cat(v).to(torch.bfloat16) for b, v in shard_acts.items()})
        save_file(tensors, str(out / f"shard_{n_shards:05d}.safetensors"))
        n_shards += 1
        shard_ids, shard_acts = [], {b: [] for b in boundaries}

    rows = packed_rows(cfg.data, tokenizer, repeat=False)
    for ids in batched(rows, batch_size):
        ids = ids[: num_rows - n_done]
        taps = tw(ids.to(device), boundaries=boundaries, compute_logits=False).boundaries
        shard_ids.append(ids)
        for b in boundaries:
            shard_acts[b].append(taps[b].cpu())
        n_done += ids.shape[0]
        if sum(x.shape[0] for x in shard_ids) >= rows_per_shard:
            flush()
        log.info("extracted %d/%d rows", n_done, num_rows)
        if n_done >= num_rows:
            break
    flush()
    meta = {
        "base_model": cfg.model.base_model,
        "boundaries": boundaries,
        "seq_len": cfg.data.seq_len,
        "d_model": base.config.hidden_size,
        "num_rows": n_done,
        "num_shards": n_shards,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def cached_batches(cache_dir: str, batch_size: int, boundaries: list[int]) -> Iterator[tuple[torch.Tensor, dict[int, torch.Tensor]]]:
    """Cycle forever over cached shards, yielding (input_ids, {boundary: acts})."""
    root = Path(cache_dir)
    meta = json.loads((root / "meta.json").read_text())
    missing = set(boundaries) - set(meta["boundaries"])
    if missing:
        raise ValueError(f"activation cache {cache_dir} lacks boundaries {sorted(missing)}; re-run extraction")
    shards = sorted(root.glob("shard_*.safetensors"))
    while True:
        for path in shards:
            data = load_file(str(path))
            n = data["input_ids"].shape[0]
            for i in range(0, n - batch_size + 1, batch_size):
                yield data["input_ids"][i : i + batch_size], {b: data[f"b{b}"][i : i + batch_size] for b in boundaries}


def main():
    from ssd.runtime import load_base_model, load_tokenizer, setup_env

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rows", type=int, default=512, help="number of seq_len rows to extract")
    p.add_argument("--rows-per-shard", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()
    setup_env()
    cfg = SSDConfig.from_yaml(args.config)
    base = load_base_model(cfg)
    meta = extract(cfg, base, load_tokenizer(cfg), args.out, args.rows, args.rows_per_shard, args.batch_size)
    log.info("wrote %s", meta)


if __name__ == "__main__":
    main()
