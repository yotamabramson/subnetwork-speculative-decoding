"""Summarize EAGLE-format answer files for SSD, EAGLE-3 and the baselines.

For each (system, benchmark) answer jsonl it reports:
* τ = Σ new_tokens / Σ (idx + 1), identical for every system
* tok/s = Σ new_tokens / Σ wall_time
* speedup vs a given baseline file, two ways:
  - ``ratio_tokps``: tok/s ratio (throughput, same token counting on both sides);
  - ``ratio_eagle``: EAGLE's ``speed.py`` definition (mean of per-question
    speeds; baseline tokens counted by re-tokenizing its answer text).

    python -m ssd.benchmark.eagle_report --tokenizer meta-llama/Llama-3.1-8B-Instruct \\
        --pair ssd=results/{bench}/ssd-t0.jsonl:results/{bench}/ssd-baseline-t0.jsonl \\
        --pair eagle3=results/{bench}/eagle3-t0.jsonl:results/{bench}/eagle-baseline-t0.jsonl
"""

from __future__ import annotations

import argparse
import json

import numpy as np

BENCHES = ("mt_bench", "humaneval", "gsm8k", "alpaca", "sum")


def load(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def stats(rows: list[dict]) -> dict:
    new = sum(sum(r["choices"][0]["new_tokens"]) for r in rows)
    rounds = sum(sum(i + 1 for i in r["choices"][0]["idxs"]) for r in rows)
    t = sum(sum(r["choices"][0]["wall_time"]) for r in rows)
    per_q = [sum(r["choices"][0]["new_tokens"]) / sum(r["choices"][0]["wall_time"]) for r in rows]
    return {"tau": new / max(1, rounds), "tokps": new / max(t, 1e-9), "mean_q_speed": float(np.mean(per_q)), "n": len(rows)}


def eagle_speed_ratio(model_rows, base_rows, tokenizer) -> float:
    """EAGLE's speed.py: mean per-question speed; baseline tokens re-tokenized from text."""
    speeds = [sum(r["choices"][0]["new_tokens"]) / sum(r["choices"][0]["wall_time"]) for r in model_rows]
    speeds0 = []
    for r in base_rows:
        toks = sum(len(tokenizer(a).input_ids) - 1 for a in r["choices"][0]["turns"])
        speeds0.append(toks / sum(r["choices"][0]["wall_time"]))
    return float(np.mean(speeds) / np.mean(speeds0))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--pair", action="append", required=True,
                   help="name=model_pattern:baseline_pattern, patterns contain {bench}")
    p.add_argument("--benches", nargs="+", default=list(BENCHES))
    args = p.parse_args(argv)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    for pair in args.pair:
        name, pats = pair.split("=", 1)
        mpat, bpat = pats.split(":", 1)
        print(f"\n== {name}")
        print(f"{'bench':10s} {'tau':>6s} {'tok/s':>8s} {'base tok/s':>10s} {'ratio_tokps':>11s} {'ratio_eagle':>11s}")
        taus, rs, re_ = [], [], []
        for b in args.benches:
            try:
                m, base = load(mpat.format(bench=b)), load(bpat.format(bench=b))
            except FileNotFoundError:
                print(f"{b:10s} (missing)")
                continue
            sm, sb = stats(m), stats(base)
            r1, r2 = sm["tokps"] / sb["tokps"], eagle_speed_ratio(m, base, tok)
            taus.append(sm["tau"]), rs.append(r1), re_.append(r2)
            print(f"{b:10s} {sm['tau']:6.2f} {sm['tokps']:8.1f} {sb['tokps']:10.1f} {r1:10.2f}x {r2:10.2f}x")
        if taus:
            print(f"{'mean':10s} {np.mean(taus):6.2f} {'':8s} {'':10s} {np.mean(rs):10.2f}x {np.mean(re_):10.2f}x")


if __name__ == "__main__":
    main()
