"""Forward-path tests for the SSD draft model.

The fast tests use a tiny randomly initialised Llama so they run on CPU in
seconds. ``test_llama3_8b_*`` loads the real Meta-Llama-3-8B-Instruct and only
runs with ``SSD_TEST_LLAMA3_8B=1`` (needs HF access to the gated repo and
~16 GB of accelerator memory).
"""

import copy
import os

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from ssd.config import BridgeConfig, SSDConfig
from ssd.models import SubnetworkDraftModel, TargetWrapper, TransitionBridge

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def tiny_llama():
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,  # exercise GQA
        max_position_embeddings=256,
        rope_theta=500000.0,
        attn_implementation="eager",
    )
    return LlamaForCausalLM(cfg).eval()


def identity_bridges(**kw) -> BridgeConfig:
    return BridgeConfig(init_std=0.0, **kw)


# --------------------------------------------------------------------- bridges
@pytest.mark.parametrize("mlp_type,num_layers,hidden", [("mlp", 2, None), ("mlp", 3, 32), ("swiglu", 2, 256)])
def test_bridge_shapes_and_near_identity(mlp_type, num_layers, hidden):
    bridge = TransitionBridge(64, hidden_dim=hidden, num_layers=num_layers, mlp_type=mlp_type, init_std=1e-3)
    x = torch.randn(2, 7, 64, dtype=torch.bfloat16)
    y = bridge(x)
    assert y.shape == x.shape and y.dtype == x.dtype
    rel = (y.float() - x.float()).norm() / x.float().norm()
    assert rel < 0.05, f"untrained bridge should be ~identity, rel diff {rel:.3g}"


def test_bridge_zero_init_is_exact_identity_and_trainable():
    bridge = TransitionBridge(64, init_std=0.0)
    x = torch.randn(3, 5, 64)
    torch.testing.assert_close(bridge(x), x)
    bridge(x).pow(2).sum().backward()
    assert bridge.mlp.out_proj.weight.grad.abs().sum() > 0


# ------------------------------------------------------------ slicing / sharing
def test_pointer_sharing_and_frozen_base(tiny_llama):
    draft = SubnetworkDraftModel(tiny_llama, [0, 2, 5])
    inner = tiny_llama.model
    for slot, idx in enumerate([0, 2, 5]):
        assert draft.base.layers[slot] is inner.layers[idx]
        assert draft.base.layers[slot].self_attn.q_proj.weight.data_ptr() == inner.layers[idx].self_attn.q_proj.weight.data_ptr()
    assert draft.base.embed_tokens is inner.embed_tokens
    assert draft.base.norm is inner.norm
    assert draft.base.lm_head is tiny_llama.lm_head

    # Only bridges are owned / trainable / serialised.
    assert all(k.startswith("bridges.") for k in draft.state_dict())
    assert all(p.requires_grad for p in draft.parameters())
    assert not any(p.requires_grad for p in tiny_llama.parameters())


def test_bridge_placement(tiny_llama):
    draft = SubnetworkDraftModel(tiny_llama, [0, 2, 5])  # gaps 0->2 and 2->5, last layer kept
    assert [(s.name, s.src_boundary, s.tgt_boundary) for s in draft.bridge_specs] == [
        ("into_2", 1, 2),
        ("into_5", 3, 5),
    ]
    draft = SubnetworkDraftModel(tiny_llama, [1, 2, 4])  # skips layer 0 and the last layer
    assert [s.name for s in draft.bridge_specs] == ["into_1", "into_4", "into_norm"]
    assert draft.bridge_specs[-1].tgt_boundary == 6


@pytest.mark.parametrize("bad", [[0, 6], [3, 3], [4, 2], []])
def test_invalid_layer_indices(tiny_llama, bad):
    with pytest.raises(ValueError):
        SubnetworkDraftModel(tiny_llama, bad)


# -------------------------------------------------------------- forward path
def test_output_shapes_untrained_bridges(tiny_llama):
    draft = SubnetworkDraftModel(tiny_llama, [0, 2, 5])
    ids = torch.randint(0, 128, (2, 11))
    out = draft(ids, output_bridge_states=True)
    assert out.logits.shape == (2, 11, 128)
    assert out.hidden_states.shape == (2, 11, 64)
    assert set(out.bridge_states) == {"into_2", "into_5"}
    assert torch.isfinite(out.logits).all()
    assert draft(ids, logits_to_keep=3).logits.shape == (2, 3, 128)
    assert draft(ids, compute_logits=False).logits is None


def test_full_layer_selection_matches_base_model(tiny_llama):
    """With every layer selected there are no bridges, so the re-implemented
    layer forward must reproduce the HF model exactly."""
    draft = SubnetworkDraftModel(tiny_llama, list(range(6)))
    assert len(draft.bridges) == 0
    ids = torch.randint(0, 128, (2, 13))
    with torch.no_grad():
        torch.testing.assert_close(draft(ids).logits, tiny_llama(ids).logits, atol=1e-5, rtol=1e-4)


def test_identity_bridges_equal_layer_skipping(tiny_llama):
    """Zero-init bridges == running the selected base layers back to back."""
    draft = SubnetworkDraftModel(tiny_llama, [0, 3, 5], identity_bridges())
    skip = LlamaForCausalLM(copy.deepcopy(tiny_llama.config)).eval()
    skip.load_state_dict(tiny_llama.state_dict())
    skip.model.layers = torch.nn.ModuleList([skip.model.layers[i] for i in (0, 3, 5)])
    for new_idx, layer in enumerate(skip.model.layers):
        layer.self_attn.layer_idx = new_idx
    skip.config.num_hidden_layers = 3
    ids = torch.randint(0, 128, (1, 9))
    with torch.no_grad():
        torch.testing.assert_close(draft(ids).logits, skip(ids).logits, atol=1e-5, rtol=1e-4)


def test_kv_cache_incremental_matches_full(tiny_llama):
    draft = SubnetworkDraftModel(tiny_llama, [0, 2, 5])
    ids = torch.randint(0, 128, (2, 12))
    with torch.no_grad():
        full = draft(ids).logits
        cache = draft.new_cache(32)
        chunks = [draft(ids[:, :7], past_key_values=cache).logits]
        for t in range(7, 12):
            chunks.append(draft(ids[:, t : t + 1], past_key_values=cache).logits)
    assert cache.get_seq_length() == 12
    torch.testing.assert_close(torch.cat(chunks, 1), full, atol=1e-5, rtol=1e-4)


def test_tree_attention_mask_matches_per_branch_decoding(tiny_llama):
    """Tree:      prefix -> a -> {b, c};  b -> d
    Each tree node's logits must equal decoding its root-to-node path alone."""
    draft = SubnetworkDraftModel(tiny_llama, [0, 2, 5])
    torch.manual_seed(1)
    prefix = torch.randint(0, 128, (1, 6))
    a, b, c, d = 10, 20, 30, 40
    tree_tokens = torch.tensor([[a, b, c, d]])
    parents = [-1, 0, 0, 1]
    depth = [0, 1, 1, 2]
    n = len(parents)
    mask = torch.zeros(n, n, dtype=torch.bool)
    for i in range(n):
        j = i
        while j != -1:
            mask[i, j] = True
            j = parents[j]
    pos = torch.tensor([[6 + dd for dd in depth]])

    with torch.no_grad():
        cache = draft.new_cache(32)
        draft(prefix, past_key_values=cache)
        tree_logits = draft(tree_tokens, position_ids=pos, past_key_values=cache, tree_attention_mask=mask).logits

        paths = {0: [a], 1: [a, b], 2: [a, c], 3: [a, b, d]}
        for node, path in paths.items():
            seq = torch.cat([prefix, torch.tensor([path])], 1)
            ref = draft(seq).logits[:, -1]
            torch.testing.assert_close(tree_logits[:, node], ref, atol=1e-5, rtol=1e-4)

        # Compact the cache to the accepted path a -> b -> d and keep decoding.
        cache.keep_positions(6, torch.tensor([6, 7, 9]))
        nxt = draft(torch.tensor([[50]]), past_key_values=cache).logits[:, -1]
        ref = draft(torch.tensor([[*prefix[0].tolist(), a, b, d, 50]])).logits[:, -1]
        torch.testing.assert_close(nxt, ref, atol=1e-5, rtol=1e-4)


def test_gradients_flow_through_frozen_layers(tiny_llama):
    draft = SubnetworkDraftModel(tiny_llama, [0, 2, 5])
    ids = torch.randint(0, 128, (2, 8))
    draft(ids).logits.logsumexp(-1).mean().backward()
    for name, bridge in draft.bridges.items():
        assert bridge.mlp.out_proj.weight.grad is not None, name
        assert bridge.mlp.out_proj.weight.grad.abs().sum() > 0, name
    assert all(p.grad is None for p in tiny_llama.parameters())
    draft.zero_grad(set_to_none=True)


# -------------------------------------------------------------- target taps
def test_target_wrapper_boundaries(tiny_llama):
    tw = TargetWrapper(tiny_llama)
    ids = torch.randint(0, 128, (2, 10))
    out = tw(ids, boundaries=[0, 2, 5, 6])
    hf = tiny_llama(ids, output_hidden_states=True)
    for i in (0, 2, 5):
        torch.testing.assert_close(out.boundaries[i], hf.hidden_states[i])
    # Boundary L is pre-norm; HF's last hidden state is post-norm.
    torch.testing.assert_close(tiny_llama.model.norm(out.boundaries[6]), hf.hidden_states[6])
    torch.testing.assert_close(out.logits, hf.logits)
    # Hooks are removed after the call.
    assert all(not layer._forward_pre_hooks for layer in tiny_llama.model.layers)


def test_bridge_specs_line_up_with_target_boundaries(tiny_llama):
    """A bridge fed the target's src boundary is compared against its tgt boundary."""
    draft = SubnetworkDraftModel(tiny_llama, [0, 2, 5])
    tw = TargetWrapper(tiny_llama)
    ids = torch.randint(0, 128, (1, 8))
    needed = {b for s in draft.bridge_specs for b in (s.src_boundary, s.tgt_boundary)}
    taps = tw(ids, boundaries=needed, compute_logits=False).boundaries
    for s in draft.bridge_specs:
        pred = draft.bridges[s.name](taps[s.src_boundary])
        assert pred.shape == taps[s.tgt_boundary].shape


# ------------------------------------------------------------------- configs
@pytest.mark.parametrize("name,indices", [("llama3_8b", [0, 15, 31]), ("llama3_70b", [0, 39, 79])])
def test_configs_load(name, indices):
    cfg = SSDConfig.from_yaml(os.path.join(REPO_ROOT, "configs", f"{name}_subnetwork.yaml"))
    assert cfg.model.layer_indices == indices
    assert cfg.training["stage2"]["temperature"] == 2.0


# ------------------------------------------------------------ real Llama-3-8B
@pytest.mark.llama3_8b
@pytest.mark.skipif(os.environ.get("SSD_TEST_LLAMA3_8B") != "1", reason="set SSD_TEST_LLAMA3_8B=1")
def test_llama3_8b_subnetwork_forward():
    cfg = SSDConfig.from_yaml(os.path.join(REPO_ROOT, "configs", "llama3_8b_subnetwork.yaml"))
    if cfg.model.device_map is None:
        cfg.model.device_map = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    base, draft = SubnetworkDraftModel.from_config(cfg)

    for slot, idx in enumerate(cfg.model.layer_indices):
        assert draft.base.layers[slot] is base.model.layers[idx]
    assert [s.name for s in draft.bridge_specs] == ["into_15", "into_31"]
    n_bridge = sum(p.numel() for p in draft.parameters())
    assert n_bridge == 2 * (4096 + 2 * (4096 * 4096 + 4096))  # norm + 2 full-width linears, x2 bridges

    device = draft.base.embed_tokens.weight.device
    ids = torch.randint(0, base.config.vocab_size, (2, 32), device=device)
    with torch.no_grad():
        out = draft(ids, output_bridge_states=True)
        cache = draft.new_cache(64)
        draft(ids[:, :31], past_key_values=cache)
        step = draft(ids[:, 31:], past_key_values=cache)
    assert out.logits.shape == (2, 32, base.config.vocab_size)
    assert torch.isfinite(out.logits).all()
    torch.testing.assert_close(step.logits[:, -1].float(), out.logits[:, -1].float(), atol=5e-2, rtol=5e-2)
