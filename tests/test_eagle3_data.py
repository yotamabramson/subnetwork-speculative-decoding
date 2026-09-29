"""Our port of EAGLE-3's preprocessing must match theirs exactly.

Runs EAGLE's original ``preprocess_function`` (extracted from
``third_party/EAGLE/eagle/traineagle3/main.py``) and ours on the same
conversations with the real Llama-3 tokenizer. Skipped when the EAGLE clone or
the tokenizer isn't available locally.
"""

import ast
import json
import os

import pytest
import torch

from ssd.data.eagle3_data import EAGLE_SYSTEM_PROMPT, eagle3_preprocess, regenerate

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EAGLE_MAIN = os.path.join(REPO, "third_party/EAGLE/eagle/traineagle3/main.py")

CONVS = [
    [{"from": "human", "value": "What is 2+2?"}, {"from": "gpt", "value": "2+2 equals 4."}],
    [
        {"from": "human", "value": "Name three colors."},
        {"from": "gpt", "value": "Red, green and blue."},
        {"from": "human", "value": "Which one is the sky?"},
        {"from": "gpt", "value": "The sky is usually blue.\n\nAt sunset it can look orange."},
        {"from": "human", "value": "Thanks!"},
        {"from": "gpt", "value": "You're welcome."},
    ],
    [{"from": "gpt", "value": "leading assistant turn is skipped"}, {"from": "human", "value": "Hi"},
     {"from": "gpt", "value": "Hello! How can I help?"}],
]


@pytest.fixture(scope="module")
def tokenizer():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", local_files_only=True)
    except Exception:
        pytest.skip("Llama-3 tokenizer not cached locally")


@pytest.fixture(scope="module")
def eagle_preprocess():
    if not os.path.exists(EAGLE_MAIN):
        pytest.skip("third_party/EAGLE not cloned")
    src = open(EAGLE_MAIN).read()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "build_dataset_rank")
    inner = next(n for n in fn.body if isinstance(n, ast.FunctionDef) and n.name == "preprocess_function")
    ns = {"torch": torch, "train_config": {"max_len": 2048}}
    exec(compile(ast.Module(body=[inner], type_ignores=[]), EAGLE_MAIN, "exec"), ns)
    return ns["preprocess_function"]


def test_preprocess_matches_eagle(tokenizer, eagle_preprocess):
    ns_tok = tokenizer
    eagle_preprocess.__globals__["tokenizer"] = ns_tok
    theirs = eagle_preprocess({"id": list(range(len(CONVS))), "conversations": CONVS})
    for i, conv in enumerate(CONVS):
        ids, mask = eagle3_preprocess(tokenizer, conv)
        assert torch.equal(ids, theirs["input_ids"][i][0])
        assert torch.equal(mask, theirs["loss_mask"][i][0])


def test_loss_mask_covers_assistant_text_only(tokenizer):
    """Note EAGLE's own quirk, kept for parity: in every assistant turn except
    the last, its hard-coded offsets also mask the turn's final token (here the
    closing '.')."""
    ids, mask = eagle3_preprocess(tokenizer, CONVS[1])
    trained = tokenizer.decode(ids[mask.bool()])
    for a in ("Red, green and blue", "The sky is usually blue.", "You're welcome."):
        assert a in trained
    for q in ("Name three colors.", "Which one is the sky?", EAGLE_SYSTEM_PROMPT[:40]):
        assert q not in trained


def test_long_conversations_are_dropped(tokenizer):
    long = [{"from": "human", "value": "x " * 3000}, {"from": "gpt", "value": "ok"}]
    assert eagle3_preprocess(tokenizer, long, max_len=2048) is None


def test_regenerate_turn_by_turn(tmp_path):
    """Each assistant turn is regenerated from the original user turns plus the
    model's own earlier answers; resumable."""
    inp = tmp_path / "in.jsonl"
    rows = [{"id": "a", "conversations": CONVS[1]}, {"id": "b", "conversations": CONVS[0]}]
    inp.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    seen = []

    def fake(msgs):
        seen.extend(msgs)
        return [f"answer{len(m)}" for m in msgs]  # depends on the history length

    out = tmp_path / "out.jsonl"
    assert regenerate(str(inp), str(out), fake, batch=10) == 2
    got = {json.loads(line)["id"]: json.loads(line)["conversations"] for line in out.read_text().splitlines()}
    assert [t["value"] for t in got["a"]] == ["Name three colors.", "answer2", "Which one is the sky?",
                                               "answer4", "Thanks!", "answer6"]
    turn2 = next(m for m in seen if len(m) == 4)
    assert turn2[0]["content"] == EAGLE_SYSTEM_PROMPT and turn2[2] == {"role": "assistant", "content": "answer2"}
    assert regenerate(str(inp), str(out), fake) == 0  # resume: nothing left to do


def test_training_on_eagle3_format(tokenizer, tmp_path):
    """Stage 1 + Stage 2 (multi-step) run on EAGLE-format conversations, and
    only assistant-token positions carry loss."""
    from transformers import LlamaConfig, LlamaForCausalLM

    from ssd.config import SSDConfig
    from ssd.training.batches import training_steps
    from ssd.training.train_stage1_feature import train_stage1
    from ssd.training.train_stage2_distill import train_stage2

    data = tmp_path / "regen.jsonl"
    data.write_text("\n".join(json.dumps({"id": str(i), "conversations": c}) for i, c in enumerate(CONVS)) + "\n")
    torch.manual_seed(0)
    base = LlamaForCausalLM(LlamaConfig(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128,
                                        num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2)).eval()
    cfg = SSDConfig()
    cfg.draft_layers = [0, 3, 5]
    cfg.data.format, cfg.data.local_path = "eagle3", str(data)
    cfg.training.output_dir = str(tmp_path / "out")
    cfg.training.stage1.batch_size = cfg.training.stage2.batch_size = 2
    cfg.training.stage2.ttt_steps = 3

    groups = next(training_steps(cfg, tokenizer, 3, None))
    assert len(groups) == 3 and all(g.ids.shape[0] == 1 for g in groups)
    for g in groups:
        ids, mask = eagle3_preprocess(tokenizer, next(c for c in CONVS if eagle3_preprocess(tokenizer, c)[0].numel() == g.ids.shape[1]))
        assert torch.equal(g.feat[0], mask.bool())
        assert torch.equal(g.pred[0, :-1], mask[1:].bool())

    assert train_stage1(cfg, base, tokenizer, max_steps=3).exists()
    assert train_stage2(cfg, base, tokenizer, max_steps=3).exists()


def test_regenerate_drops_overlong_conversations(tokenizer, tmp_path):
    """A conversation whose regenerated length passes max_len stops being
    regenerated and is dropped (EAGLE drops it at training anyway)."""
    inp = tmp_path / "in.jsonl"
    rows = [{"id": "short", "conversations": CONVS[0]}, {"id": "long", "conversations": CONVS[1]}]
    inp.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    calls = []

    def fake(msgs):
        calls.append(len(msgs))
        return ["word " * (5 if len(m) == 2 else 400) for m in msgs]  # long answers after the first turn

    out = tmp_path / "out.jsonl"
    n = regenerate(str(inp), str(out), fake, tokenizer=tokenizer, max_len=300)
    assert n == 1 and json.loads(out.read_text())["id"] == "short"
    assert (tmp_path / "out.jsonl.dropped").read_text().count("long") == 1
    assert calls == [2, 1]  # the long conversation got no third call after crossing max_len
    assert regenerate(str(inp), str(out), fake, tokenizer=tokenizer, max_len=300) == 0  # resume skips both
