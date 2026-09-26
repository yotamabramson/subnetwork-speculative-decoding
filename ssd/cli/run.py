"""ssd-run: run the model on a test prompt.

    ssd-run --config configs/llama32_1b_subnetwork.yaml --prompt "Explain KV caches."
    ssd-run --config ... --prompt-file test.txt --depth 6 --branch 4,2,2,1,1,1 --max-nodes 32
    ssd-run --config ... --prompt "..." --temperature 0.7 --seed 0

Modes:
  sd      (default) speculative decoding with the draft sub-network + tree from
          `drafting:` (overridable by flags), timed against plain autoregressive
          decoding of the target. Reports accepted tokens per round and speedup.
  target  autoregressive target only
  draft   the draft sub-network generating on its own (to eyeball bridge quality)

Bridges load from {output_dir}/{profile}/stage2.pt (else stage1.pt), unless
--bridges or --untrained is given.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace

import torch

from ssd.config import SSDConfig, profile_name
from ssd.engine.speculative import SpeculativeGenerator, TargetRunner, autoregressive_generate
from ssd.runtime import build_draft, input_device, load_base_model, load_tokenizer, setup_env

log = logging.getLogger("ssd")


def encode_prompt(tokenizer, prompt: str, raw: bool) -> torch.Tensor:
    if not raw and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt", return_dict=True
        )["input_ids"]
    return tokenizer(prompt, return_tensors="pt")["input_ids"]


def _int_or_list(s: str):
    parts = [int(x) for x in s.split(",")]
    return parts[0] if len(parts) == 1 else parts


def main(argv=None):
    p = argparse.ArgumentParser(prog="ssd-run", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompt")
    g.add_argument("--prompt-file")
    p.add_argument("--mode", choices=["sd", "target", "draft"], default="sd")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--raw", action="store_true", help="don't wrap the prompt in the chat template")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-warmup", action="store_true")
    t = p.add_argument_group("tree overrides (default: config `drafting:`)")
    t.add_argument("--depth", type=int)
    t.add_argument("--branch", type=_int_or_list, help="int or comma list per depth, e.g. 3,2,2,1")
    t.add_argument("--width", type=_int_or_list)
    t.add_argument("--max-nodes", type=int)
    t.add_argument("--temperature", type=float)
    t.add_argument("--top-p", type=float)
    b = p.add_mutually_exclusive_group()
    b.add_argument("--bridges", help="explicit bridge checkpoint path")
    b.add_argument("--untrained", action="store_true", help="use freshly initialised bridges")
    args = p.parse_args(argv)

    setup_env()
    cfg = SSDConfig.from_yaml(args.config)
    overrides = {k: getattr(args, k) for k in ("depth", "branch", "width", "max_nodes", "temperature", "top_p") if getattr(args, k) is not None}
    dcfg = replace(cfg.drafting, **overrides)
    dcfg.validate()
    prompt = args.prompt if args.prompt is not None else open(args.prompt_file).read()

    base = load_base_model(cfg)
    tokenizer = load_tokenizer(cfg)
    dev = input_device(base)
    ids = encode_prompt(tokenizer, prompt, args.raw).to(dev)
    eos = base.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    L = base.config.num_hidden_layers
    target = TargetRunner(base)
    draft = None
    if args.mode in ("sd", "draft"):
        draft = build_draft(cfg, base, None if args.untrained else (args.bridges or "latest"))

    print(f"\nprompt: {ids.shape[1]} tokens | device: {dev} | draft layers {profile_name(cfg.draft_layers)} ({len(cfg.draft_layers)}/{L})")
    print(f"tree: depth={dcfg.depth} branch={dcfg.branch} width={dcfg.width} max_nodes={dcfg.max_nodes} "
          f"temperature={dcfg.temperature} top_p={dcfg.top_p}")

    def run_ar(model, n, seed):
        return autoregressive_generate(model, ids, n, eos_ids, dcfg.temperature, dcfg.top_p, seed)

    if args.mode == "draft":
        # Autoregressive generation with the draft alone.
        res = run_ar(draft, args.max_new_tokens, args.seed)
        print(f"\n=== draft only: {len(res.tokens)} tokens, {len(res.tokens) / res.decode_time:.1f} tok/s ===")
        print(tokenizer.decode(res.tokens, skip_special_tokens=True))
        return

    if not args.no_warmup:  # first MPS/CUDA calls include kernel compilation
        run_ar(target, 8, args.seed)
        if args.mode == "sd":
            SpeculativeGenerator(base, draft, dcfg, target).generate(ids, 8, eos_ids, seed=args.seed)

    ar = run_ar(target, args.max_new_tokens, args.seed)
    ar_tps = len(ar.tokens) / ar.decode_time
    if args.mode == "target":
        print(f"\n=== target: {len(ar.tokens)} tokens, {ar_tps:.1f} tok/s ===")
        print(tokenizer.decode(ar.tokens, skip_special_tokens=True))
        return

    sd = SpeculativeGenerator(base, draft, dcfg, target).generate(ids, args.max_new_tokens, eos_ids, seed=args.seed)
    sd_tps = len(sd.tokens) / sd.decode_time
    rounds = len(sd.accepted_per_round)

    print(f"\n=== speculative: {len(sd.tokens)} tokens ===")
    print(tokenizer.decode(sd.tokens, skip_special_tokens=True))
    print("\n--- stats (decode phase; prefill excluded) ---")
    print(f"rounds:                {rounds}")
    print(f"accepted tokens/round: {sd.mean_accepted:.2f}   (incl. bonus; max {dcfg.depth + 1})")
    print(f"mean tree size:        {sum(sd.tree_sizes) / max(1, rounds):.1f} nodes")
    per = {k: 1000 * v / max(1, rounds) for k, v in sd.times.items() if k not in ("prefill", "draft_prefill")}
    print("per round:             " + "  ".join(f"{k}={v:.1f}ms" for k, v in per.items()))
    print(f"target step (AR):      {1000 * ar.decode_time / max(1, len(ar.tokens) - 1):.1f}ms/token")
    print(f"throughput:            AR {ar_tps:.1f} tok/s  |  SD {sd_tps:.1f} tok/s  |  speedup {sd_tps / ar_tps:.2f}x")
    if dcfg.temperature == 0:
        n = min(len(ar.tokens), len(sd.tokens))
        same = ar.tokens[:n] == sd.tokens[:n]
        print(f"greedy output identical to AR: {same}" + ("" if same else "  (bf16 batched-vs-single numerics can flip near-ties)"))


if __name__ == "__main__":
    main()
