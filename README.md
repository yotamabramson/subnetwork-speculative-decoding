# SSD: Subnetwork Speculative Decoding

A speculative-decoding draft model built from a **frozen subset of the target
model's own decoder layers** (e.g. first / middle / last), joined by small
learned **transition bridges**. The draft only references weights that are
already in memory for the target; the bridges are the only new parameters.

```
embed ─► layer a0 ─► bridge ─► layer a1 ─► bridge ─► layer a2 ─► norm ─► lm_head
```

Each bridge is a residual MLP, `out = x + MLP(RMSNorm(x))`. Its output
projection starts near zero, so an untrained draft behaves like plain layer
skipping.

Training has two stages:
1. **Feature regression.** Each bridge is fed the target's activation at the
   boundary it starts from and regressed onto the activation at the boundary
   it must reproduce, with loss `α·MSE + β·(1 − cos)`.
2. **Logit distillation.** The draft is trained end to end with KL divergence
   at temperature T=2 against the target, plus ground-truth cross-entropy.

## Layout

```
configs/                 llama3_8b_subnetwork.yaml, llama3_70b_subnetwork.yaml
ssd/
  config.py              SSDConfig (YAML) + layer-index validation
  models/
    bridges.py           TransitionBridge (mlp 2–3 layers / swiglu; bottleneck or expansion)
    subnetwork_draft.py  SubnetworkDraftModel: sliced frozen layers + bridges
    target_wrapper.py    TargetWrapper: taps residual-stream boundaries of the target
  engine/
    kv_cache.py          DraftKVCache: preallocated per-slot cache with tree compaction
    tree_drafter.py      (todo) dynamic tree expansion
    verify.py            (todo) greedy + stochastic verification
  data/extract_activations.py      (todo)
  training/train_stage1_feature.py (todo)
  training/train_stage2_distill.py (todo)
  benchmark/latency_eval.py        (todo)
  benchmark/profile_cuda.py        (todo)
tests/test_forward_pass.py
```

## Conventions

- **Layer indices are 0-based.** Llama-3-8B has layers 0..31, so first/middle/last
  is `[0, 15, 31]`. Llama-3-70B has layers 0..79, so it is `[0, 39, 79]`.
- **Boundary `i`** is the residual stream entering layer `i`, and **boundary `L`**
  is the stream entering the final norm. A bridge is placed wherever the
  draft skips layers. `draft.bridge_specs` lists each bridge's
  `(src_boundary, tgt_boundary)` pair, and `TargetWrapper` taps those same
  boundaries to produce the Stage 1 targets.
- The draft **re-implements the decoder-layer forward** using the layer's own
  submodules, so it can take arbitrary tree attention masks and its own KV
  cache. A test checks that it matches HF `LlamaForCausalLM` exactly when
  every layer is selected.
- `draft.parameters()` / `state_dict()` contain **only the bridges**, and the
  base model is frozen. Bridges are fp32 by default and cast activations in
  and out. With `device_map="auto"`, each bridge is placed on the device of
  the layer that consumes its output.

## Usage

```python
from ssd.config import SSDConfig
from ssd.models import SubnetworkDraftModel

cfg = SSDConfig.from_yaml("configs/llama3_8b_subnetwork.yaml")
base, draft = SubnetworkDraftModel.from_config(cfg)

cache = draft.new_cache(max_length=4096)
out = draft(input_ids, past_key_values=cache)                  # prefill
out = draft(tree_ids, position_ids=tree_pos, past_key_values=cache,
            tree_attention_mask=tree_mask, logits_to_keep=0)   # tree step
cache.keep_positions(start, accepted_positions)                # keep accepted path
```

## Tests

```bash
pip install -e ".[dev]"
pytest                                   # tiny random Llama, CPU, ~2s
SSD_TEST_LLAMA3_8B=1 pytest -m llama3_8b  # real Llama-3-8B-Instruct (gated HF repo)
```
