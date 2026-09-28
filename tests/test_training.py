"""Config parsing, losses, and short end-to-end training runs on a tiny Llama."""

import os

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from ssd.config import SSDConfig
from ssd.data.extract_activations import cached_batches, extract
from ssd.runtime import build_draft, checkpoint_path
from ssd.training.losses import chunked_distill_loss, distill_loss, feature_loss
from ssd.training.train_stage1_feature import train_stage1
from ssd.training.train_stage2_distill import train_stage2

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CharTokenizer:
    chat_template = None

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [ord(c) % 128 for c in text]}


@pytest.fixture
def tiny_setup(tmp_path):
    torch.manual_seed(0)
    base = LlamaForCausalLM(
        LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=6,
                    num_attention_heads=4, num_key_value_heads=2)
    ).eval()
    text = tmp_path / "corpus.txt"
    text.write_text("\n\n".join(f"sample {i}: the quick brown fox jumps over the lazy dog." for i in range(50)))
    cfg = SSDConfig()
    cfg.draft_layers = [0, 3, 5]
    cfg.data.local_path = str(text)
    cfg.data.seq_len = 32
    cfg.training.output_dir = str(tmp_path / "out")
    cfg.training.log_every = 5
    for s in (cfg.training.stage1, cfg.training.stage2):
        s.batch_size, s.warmup_steps = 4, 2
    cfg.training.stage1.lr = 3e-3
    cfg.training.stage2.lr = 1e-3
    return cfg, base


# --------------------------------------------------------------------- config
@pytest.mark.parametrize("name,layers", [("llama32_1b", [0, 7, 15]), ("llama3_8b", [0, 15, 31]), ("llama3_70b", [0, 39, 79])])
def test_shipped_configs_load(name, layers):
    cfg = SSDConfig.from_yaml(os.path.join(REPO_ROOT, "configs", f"{name}_subnetwork.yaml"))
    assert cfg.draft_layers == layers
    assert cfg.training.stage2.temperature == 2.0
    assert len(cfg.drafting.per_depth(cfg.drafting.branch, "branch")) == cfg.drafting.depth


@pytest.mark.parametrize("yaml_text", [
    "draft_layers: {1: [0, 1]}\n",  # the old per-K map is no longer accepted
    "draft_layers: [0, 1]\ndrafting: {depth: 3, branch: [2, 2]}\n",  # branch list length != depth
    "draft_layers: [0, 1]\ndrafting: {depth: 0}\n",
])
def test_bad_configs_rejected(tmp_path, yaml_text):
    p = tmp_path / "c.yaml"
    p.write_text(yaml_text)
    with pytest.raises(ValueError):
        SSDConfig.from_yaml(p)


def test_unknown_config_keys_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("draft_layers: [0, 1]\ntraining: {stage1: {lrr: 1.0}}\n")
    with pytest.raises(ValueError, match="lrr"):
        SSDConfig.from_yaml(p)


# --------------------------------------------------------------------- losses
def test_feature_loss_zero_at_match_and_mask():
    x = torch.randn(2, 5, 16)
    loss, st = feature_loss(x, x)
    assert loss.item() < 1e-6 and st["cos"] > 0.9999
    y = x.clone()
    y[:, 0] = -y[:, 0]  # only position 0 is wrong
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[:, 0] = False
    assert feature_loss(y, x, mask=mask)[0].item() < 1e-6
    assert feature_loss(y, x)[0].item() > 0.1


def test_distill_loss_zero_kd_when_equal():
    logits = torch.randn(2, 4, 10)
    loss, st = distill_loss(logits, logits, None, temperature=2.0, ce_weight=0.0)
    assert abs(st["kd"]) < 1e-5 and st["top1_agree"] == 1.0


def test_chunked_distill_matches_full():
    torch.manual_seed(0)
    head = torch.nn.Linear(16, 50, bias=False)
    d = torch.randn(3, 7, 16, requires_grad=True)
    t = torch.randn(3, 7, 16)
    labels = torch.randint(0, 50, (3, 7))
    labels[:, -1] = -100
    ref, ref_st = distill_loss(head(d), head(t), labels, 2.0, 1.0, 0.3)
    (g_ref,) = torch.autograd.grad(ref, d)
    out, st = chunked_distill_loss(head, d, t, labels, 2.0, 1.0, 0.3, chunk_tokens=4)
    (g,) = torch.autograd.grad(out, d)
    torch.testing.assert_close(out, ref)
    torch.testing.assert_close(g, g_ref)
    assert st["top1_agree"] == pytest.approx(ref_st["top1_agree"])


# ------------------------------------------------------------------- training
@pytest.mark.parametrize("mode", ["teacher", "chained"])
def test_stage1_reduces_loss_and_saves(tiny_setup, mode):
    cfg, base = tiny_setup
    cfg.training.stage1.mode = mode
    draft0 = build_draft(cfg, base, checkpoint=None)
    path = train_stage1(cfg, base, CharTokenizer(), max_steps=40)
    assert path == checkpoint_path(cfg, 1) and path.exists()

    trained = build_draft(cfg, base)  # "latest" -> stage1
    from ssd.models.target_wrapper import TargetWrapper

    ids = torch.tensor([[ord(c) % 128 for c in "sample 7: the quick brown fox jumps over"]])
    taps = TargetWrapper(base)(ids, boundaries=[1, 3, 4, 5], compute_logits=False).boundaries

    def err(d):
        return sum(feature_loss(d.bridges[s.name](taps[s.src_boundary]), taps[s.tgt_boundary])[0].item() for s in d.bridge_specs)

    assert err(trained) < 0.7 * err(draft0)


def test_stage2_improves_agreement(tiny_setup):
    cfg, base = tiny_setup
    cfg.training.stage2.ce_weight = 0.0  # the random target isn't a language model; isolate the KD term
    cfg.draft_layers = [0, 1, 3, 4, 5]
    train_stage1(cfg, base, CharTokenizer(), max_steps=30)
    ids = torch.tensor([[ord(c) % 128 for c in "sample 3: the quick brown fox jumps over the lazy dog."]])
    with torch.no_grad():
        tgt = base(ids).logits

    def kd(d):
        with torch.no_grad():
            return distill_loss(d(ids).logits, tgt, None, ce_weight=0.0)[1]["kd"]

    before = kd(build_draft(cfg, base))
    path = train_stage2(cfg, base, CharTokenizer(), max_steps=40)
    assert path.name == "stage2.pt"
    assert kd(build_draft(cfg, base)) < before  # "latest" now resolves to stage2


def test_stage2_multistep_training_runs_and_improves_deep_steps(tiny_setup):
    """ttt_steps=3 trains all unrolled steps; step-2/3 KD should drop too."""
    cfg, base = tiny_setup
    cfg.training.stage2.ce_weight = 0.0
    cfg.training.stage2.ttt_steps = 3
    cfg.training.stage2.micro_batch = 2
    from ssd.models.target_wrapper import TargetWrapper

    ids = torch.tensor([[ord(c) % 128 for c in "sample 3: the quick brown fox jumps over the lazy dog."]])
    layers = cfg.draft_layers

    def deep_kd(d):
        with torch.no_grad():
            taps = TargetWrapper(base)(ids, boundaries=[*layers, 6], compute_logits=False).boundaries
            outs = d.forward_unrolled(ids, {i: taps[i] for i in layers}, 3)
            tgt = base.lm_head(base.model.norm(taps[6]))
            return distill_loss(base.lm_head(outs[2])[:, 2:], tgt[:, 2:], None, ce_weight=0.0)[1]["kd"]

    before = deep_kd(build_draft(cfg, base, checkpoint=None))
    path = train_stage2(cfg, base, CharTokenizer(), init_from=None, max_steps=30)
    assert deep_kd(build_draft(cfg, base, checkpoint=path)) < before


def test_activation_cache_roundtrip_and_stage1_from_cache(tiny_setup, tmp_path):
    cfg, base = tiny_setup
    meta = extract(cfg, base, CharTokenizer(), str(tmp_path / "acts"), num_rows=12, rows_per_shard=8)
    assert meta["num_rows"] == 12 and meta["num_shards"] == 2
    assert meta["boundaries"] == [0, 1, 3, 4, 5]  # bridge endpoints + selected-layer inputs
    ids, acts = next(cached_batches(str(tmp_path / "acts"), 4, [1, 3]))
    assert ids.shape == (4, 32) and acts[3].shape == (4, 32, 64)

    cfg.training.stage1.activation_cache = str(tmp_path / "acts")
    assert train_stage1(cfg, base, CharTokenizer(), max_steps=5).exists()
