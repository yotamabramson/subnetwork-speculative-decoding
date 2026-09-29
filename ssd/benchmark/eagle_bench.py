"""SSD on EAGLE-3's benchmarks, protocol-identical to EAGLE's evaluation.

Mirrors ``eagle/evaluation/gen_ea_answer_llama3chat.py`` (SafeAILab/EAGLE @
cb7e084): the same question files, fixed system prompt, chat template, 3
warmup runs on the first question, seeds, stop tokens, output clean-up and
answer-jsonl format (``choices[{index, turns, idxs, new_tokens, wall_time}]``).
MT-bench turn 2 includes the model's own turn-1 answer. Only the model
changes: SSD speculative decoding, or (``--mode baseline``) plain
autoregressive decoding through the same code path.

Parity notes:
* EAGLE calls ``eagenerate`` without ``max_new_tokens``, so its default of 512
  applies (with ``max_length`` 2048). We use the same limits.
* ``idxs`` holds the last round index, as in EAGLE, so a turn has ``idx + 1``
  rounds and τ = Σ new_tokens / Σ (idx + 1), computed identically for both
  systems by ``ssd.benchmark.eagle_report``.

    python -m ssd.benchmark.eagle_bench --config configs/llama31_8b_subnetwork.yaml \\
        --bench-name mt_bench --temperature 0 --answer-file results/mt_bench/ssd-t0.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace

import torch

from ssd.config import SSDConfig
from ssd.data.eagle3_data import EAGLE_SYSTEM_PROMPT
from ssd.engine.speculative import SpeculativeGenerator, TargetRunner, _sync, autoregressive_generate
from ssd.runtime import build_draft, input_device, load_base_model, load_tokenizer, setup_env

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_QUESTION_DIR = os.path.join(REPO, "third_party/EAGLE/eagle/data")
BENCHES = ("mt_bench", "humaneval", "gsm8k", "alpaca", "sum")


def load_questions(path: str, begin=None, end=None) -> list[dict]:
    with open(path) as f:
        qs = [json.loads(line) for line in f if line.strip()]
    return qs[begin:end]


def clean_output(tokenizer, output_ids: list[int]) -> str:
    """EAGLE's post-processing: cut at the first stop token, decode, strip special tokens."""
    stop = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")}
    cut = next((i for i, t in enumerate(output_ids) if t in stop), None)
    if cut is not None:
        output_ids = output_ids[:cut]
    out = tokenizer.decode(output_ids, spaces_between_special_tokens=False)
    for special in tokenizer.special_tokens_map.values():
        for s in special if isinstance(special, list) else [special]:
            out = out.replace(s, "")
    return out.strip()


def answer_question(question, tokenizer, generate, device, seed: int) -> dict:
    torch.manual_seed(seed)
    messages = [{"role": "system", "content": EAGLE_SYSTEM_PROMPT}]
    turns, idxs, new_tokens, wall_time = [], [], [], []
    for qs in question["turns"]:
        messages.append({"role": "user", "content": qs})
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = torch.tensor(tokenizer([prompt], add_special_tokens=False).input_ids, device=device)
        _sync(device)
        t0 = time.time()
        tokens, rounds = generate(input_ids, seed)
        _sync(device)
        wall_time.append(time.time() - t0)
        output = clean_output(tokenizer, tokens)
        turns.append(output)
        idxs.append(int(rounds - 1))
        new_tokens.append(int(len(tokens)))
        messages.append({"role": "assistant", "content": output})
    return {"index": seed, "turns": turns, "idxs": idxs, "new_tokens": new_tokens, "wall_time": wall_time}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--mode", choices=["ssd", "baseline"], default="ssd")
    p.add_argument("--bench-name", default="mt_bench", choices=BENCHES + ("qa",))
    p.add_argument("--question-dir", default=DEFAULT_QUESTION_DIR)
    p.add_argument("--question-begin", type=int)
    p.add_argument("--question-end", type=int)
    p.add_argument("--answer-file", required=True)
    p.add_argument("--model-id", default="ssd")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-new-tokens", type=int, default=512)  # EAGLE's effective default
    p.add_argument("--num-choices", type=int, default=1)
    p.add_argument("--dtype", default="float16", help="EAGLE evaluates in fp16")
    p.add_argument("--bridges", help="bridge checkpoint (default: latest under training.output_dir)")
    t = p.add_argument_group("tree (EAGLE-3: total_token 60 -> max_nodes 59, top_k 10 -> branch/width 10)")
    t.add_argument("--depth", type=int)
    t.add_argument("--branch", type=int)
    t.add_argument("--width", type=int)
    t.add_argument("--max-nodes", type=int)
    t.add_argument("--draft-vocab", type=int)
    args = p.parse_args(argv)

    setup_env()
    from accelerate.utils import set_seed

    set_seed(0)
    cfg = SSDConfig.from_yaml(args.config)
    cfg.model.torch_dtype = args.dtype
    over = {k: getattr(args, k) for k in ("depth", "branch", "width", "max_nodes", "draft_vocab") if getattr(args, k) is not None}
    cfg.drafting = replace(cfg.drafting, temperature=args.temperature, top_p=1.0, **over)
    cfg.drafting.validate()

    base = load_base_model(cfg)
    tokenizer = load_tokenizer(cfg)
    device = input_device(base)
    target = TargetRunner(base)
    stop = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")}

    if args.mode == "ssd":
        draft = build_draft(cfg, base, args.bridges or "latest", inference=True)
        sd = SpeculativeGenerator(base, draft, cfg.drafting, target)

        def generate(ids, seed):
            res = sd.generate(ids, args.max_new_tokens, stop, seed=seed if args.temperature > 0 else None)
            return res.tokens, max(1, len(res.accepted_per_round))
    else:
        def generate(ids, seed):
            res = autoregressive_generate(target, ids, args.max_new_tokens, stop, args.temperature, 1.0,
                                          seed if args.temperature > 0 else None)
            return res.tokens, max(1, len(res.tokens))

    qfile = os.path.join(args.question_dir, args.bench_name, "question.jsonl")
    questions = load_questions(qfile, args.question_begin, args.question_end)
    model_id = f"{args.model_id}-temperature-{args.temperature}"
    with torch.inference_mode():
        for _ in range(3):  # warmup, as EAGLE
            answer_question(questions[0], tokenizer, generate, device, 0)
        os.makedirs(os.path.dirname(os.path.abspath(args.answer_file)), exist_ok=True)
        for q in questions:
            choices = [answer_question(q, tokenizer, generate, device, i) for i in range(args.num_choices)]
            with open(args.answer_file, "a") as f:
                f.write(json.dumps({"question_id": q["question_id"], "answer_id": f"{model_id}-{q['question_id']}",
                                    "model_id": model_id, "choices": choices, "tstamp": time.time()}) + "\n")
            print(f"{args.bench_name} q{q['question_id']}: "
                  f"tau={sum(choices[0]['new_tokens']) / sum(i + 1 for i in choices[0]['idxs']):.2f} "
                  f"tok/s={sum(choices[0]['new_tokens']) / sum(choices[0]['wall_time']):.1f}", flush=True)


if __name__ == "__main__":
    main()
