"""ssd-train: train the transition bridges for the config's draft layer subset.

    ssd-train --config configs/llama32_1b_subnetwork.yaml                 # stage 1 then stage 2
    ssd-train --config ... --stage 1                                      # one stage only
    ssd-train --config ... --steps 20 --data-file notes.txt               # quick smoke run
    accelerate launch -m ssd.cli.train --config ...                       # multi-GPU (DDP)

Checkpoints: {training.output_dir}/{profile}/stage{1,2}.pt
"""

from __future__ import annotations

import argparse
import logging
import os

from ssd.config import SSDConfig, profile_name
from ssd.data.token_freq import ensure_token_freq
from ssd.runtime import free_device_memory, load_base_model, load_tokenizer, setup_env
from ssd.training.train_stage1_feature import train_stage1
from ssd.training.train_stage2_distill import train_stage2

log = logging.getLogger("ssd")


def main(argv=None):
    p = argparse.ArgumentParser(prog="ssd-train", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--stage", choices=["1", "2", "all"], default="all")
    p.add_argument("--steps", type=int, help="override steps for every stage (smoke tests)")
    p.add_argument("--output-dir", help="override training.output_dir")
    p.add_argument("--data-file", help="override data.local_path (.jsonl with messages/text, or .txt)")
    p.add_argument("--activation-cache", help="override training.stage1.activation_cache")
    args = p.parse_args(argv)

    setup_env()
    cfg = SSDConfig.from_yaml(args.config)
    if args.output_dir:
        cfg.training.output_dir = args.output_dir
    if args.data_file:
        cfg.data.local_path = args.data_file
    if args.activation_cache:
        cfg.training.stage1.activation_cache = args.activation_cache

    accelerator = None
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        from accelerate import Accelerator

        if cfg.model.device_map is not None:
            raise SystemExit("multi-process DDP and model.device_map sharding can't be combined; pick one")
        accelerator = Accelerator()

    base = load_base_model(cfg, accelerator.device if accelerator else None)
    tokenizer = load_tokenizer(cfg)
    log.info("draft layers %s", profile_name(cfg.draft_layers))
    if accelerator is None or accelerator.is_main_process:
        ensure_token_freq(cfg, tokenizer)  # for drafting.draft_vocab

    if args.stage in ("1", "all"):
        train_stage1(cfg, base, tokenizer, max_steps=args.steps, accelerator=accelerator)
        free_device_memory()
    if args.stage in ("2", "all"):
        train_stage2(cfg, base, tokenizer, max_steps=args.steps, accelerator=accelerator)
    log.info("done")


if __name__ == "__main__":
    main()
