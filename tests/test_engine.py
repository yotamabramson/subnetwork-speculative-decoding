"""Speculative decoding correctness on tiny random Llamas (CPU, fp32).

* Greedy: output must equal plain greedy decoding with the target, whatever
  the tree shape and however bad the draft.
* Sampling: the distribution of generated tokens must match the target's.
"""

import itertools

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from ssd.config import BridgeConfig, DraftingConfig
from ssd.engine.speculative import SpeculativeGenerator, TargetRunner, autoregressive_generate
from ssd.engine.tree_drafter import TreeDrafter
from ssd.models import SubnetworkDraftModel


def tiny(vocab=128, seed=0, scale=1.0):
    torch.manual_seed(seed)
    m = LlamaForCausalLM(
        LlamaConfig(vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=6,
                    num_attention_heads=4, num_key_value_heads=2)
    ).eval()
    with torch.no_grad():
        m.lm_head.weight.mul_(scale)
    return m


TREES = {
    "chain": DraftingConfig(depth=4, branch=1),
    "tree": DraftingConfig(depth=3, branch=[3, 2, 2]),
    "pruned": DraftingConfig(depth=5, branch=[4, 3, 2, 2, 1], width=[4, 6, 6, 4, 4], max_nodes=12),
    "deep_chain": DraftingConfig(depth=8, branch=1),
}


@pytest.fixture(scope="module")
def base():
    return tiny()


@pytest.mark.parametrize("tree_name", list(TREES))
@pytest.mark.parametrize("draft_kind", ["untrained", "perfect"])
def test_greedy_sd_matches_autoregressive(base, tree_name, draft_kind):
    if draft_kind == "perfect":  # all layers, no bridges -> draft == target, everything accepted
        draft = SubnetworkDraftModel(base, list(range(6)))
    else:
        draft = SubnetworkDraftModel(base, [0, 3, 5], BridgeConfig(init_std=0.02))
    target = TargetRunner(base)
    eos = {-1}
    for prompt_seed in range(3):
        g = torch.Generator().manual_seed(prompt_seed)
        ids = torch.randint(0, 128, (1, 5 + 4 * prompt_seed), generator=g)
        ref = autoregressive_generate(target, ids, 40, eos).tokens
        hf = base.generate(ids, max_new_tokens=40, do_sample=False, pad_token_id=0)[0, ids.shape[1]:].tolist()
        assert ref == hf
        res = SpeculativeGenerator(base, draft, TREES[tree_name], target).generate(ids, 40, eos)
        assert res.tokens == ref, f"{tree_name}/{draft_kind} diverged"
        assert sum(res.accepted_per_round) >= 39
        if draft_kind == "perfect" and TREES[tree_name].max_nodes is None:
            assert res.mean_accepted > TREES[tree_name].depth  # ~depth+1 per round


def test_greedy_sd_stops_at_eos(base):
    target = TargetRunner(base)
    ids = torch.randint(0, 128, (1, 6), generator=torch.Generator().manual_seed(0))
    free = autoregressive_generate(target, ids, 30, {-1}).tokens
    eos = {free[10]}
    ref = autoregressive_generate(target, ids, 30, eos).tokens
    draft = SubnetworkDraftModel(base, list(range(6)))
    res = SpeculativeGenerator(base, draft, TREES["chain"], target).generate(ids, 30, eos)
    assert res.tokens == ref and res.tokens[-1] in eos


def test_tree_shape_limits(base):
    draft = SubnetworkDraftModel(base, [0, 3, 5])
    cfg = TREES["pruned"]
    drafter = TreeDrafter(draft, cfg)
    cache = draft.new_cache(64)
    tree = drafter.build(cache, torch.tensor([[1, 2, 3]]), root_pos=2)
    assert len(tree) <= cfg.max_nodes
    for i in range(len(tree)):  # ancestor-closed, depths consistent
        p = tree.parents[i]
        assert p == -1 or (p < i and tree.depths[p] == tree.depths[i] - 1)
    assert cache.get_seq_length() == 3 + drafter.max_tree_cache()


@pytest.mark.parametrize("tree_cfg", [
    DraftingConfig(depth=2, branch=[2, 2], temperature=1.0),
    DraftingConfig(depth=2, branch=1, temperature=0.8, top_p=0.9),
])
def test_sampling_sd_matches_target_distribution(tree_cfg):
    """Tokens 2 and 3 come from one speculative round (token 1 from prefill).
    Compare their empirical joint with the target's exact joint."""
    from ssd.engine.tree_drafter import warp_probs

    V = 8
    base = tiny(vocab=V, seed=1, scale=8.0)  # sharper than uniform, still multi-modal
    draft = SubnetworkDraftModel(base, [0, 5], BridgeConfig(init_std=0.05))
    target = TargetRunner(base)
    prompt = torch.tensor([[1, 5, 2]])
    T, P = tree_cfg.temperature, tree_cfg.top_p

    with torch.no_grad():
        p1 = warp_probs(base(prompt).logits[0, -1], T, P)
        exact = torch.zeros(V, V, V)
        for a, b in itertools.product(range(V), range(V)):
            logits = base(torch.cat([prompt, torch.tensor([[a, b]])], 1)).logits[0]
            exact[a, b] = p1[a] * warp_probs(logits[-2], T, P)[b] * warp_probs(logits[-1], T, P)
    exact_23 = exact.sum(0)  # marginal over tokens 2, 3

    sd = SpeculativeGenerator(base, draft, tree_cfg, target)
    N = 3000
    counts = torch.zeros(V, V)
    for s in range(N):
        toks = sd.generate(prompt, 3, set(), seed=s).tokens
        counts[toks[1], toks[2]] += 1
    tv = 0.5 * (counts / N - exact_23).abs().sum().item()
    assert tv < 0.06, f"TV distance {tv:.3f}"
