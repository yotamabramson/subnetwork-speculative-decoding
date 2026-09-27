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

## Quick start (Mac / MPS, Llama-3.2-1B)

```bash
python3.11 -m venv .venv && .venv/bin/pip install -e ".[dev]"
echo "HF_TOKEN=hf_..." > .env        # gated meta-llama repos; loaded at runtime, git-ignored

# 1) train the bridges (stage 1 then stage 2)
ssd-train --config configs/llama32_1b_subnetwork.yaml

# 2) speculative decoding on a test prompt, timed against plain autoregressive decoding
ssd-run --config configs/llama32_1b_subnetwork.yaml --prompt "Explain KV caches."
ssd-run --config ... --prompt "..." --depth 6 --branch 4,2,2,1,1,1 --max-nodes 32   # tree overrides
ssd-run --config ... --prompt "..." --mode target     # plain autoregressive target only
```

## Configuration

Each model has one YAML. The draft is a single subset of base layers, used for
every draft step (per-step subsets aren't supported):

```yaml
draft_layers: [0, 7, 15]     # 0-based

drafting:                    # the speculation tree
  depth: 4                   # K: tokens drafted ahead per round
  branch: [3, 2, 2, 1]       # children proposed per kept node, per depth (int = same everywhere)
  width: 8                   # nodes kept per depth, by cumulative draft log-prob (null = no cap)
  max_nodes: 24              # nodes sent to the target for verification (null = all)
  temperature: 0.0           # 0 = greedy
  top_p: 1.0
  draft_vocab: null          # e.g. 32768: draft scores only the most frequent tokens
```

`draft_vocab` targets the draft's biggest cost: the 128k-row `lm_head` is
~60% of a draft step on Llama-3.2-1B. EAGLE-3 uses the same trick. The
verification still uses the full vocabulary, so outputs are unchanged, and
tokens outside the subset simply can't be drafted. Frequencies come from the
training data (`ssd-train` writes `{output_dir}/token_freq.pt`; or run
`python -m ssd.data.token_freq --config ...`).

`branch: 1` gives a plain chain of `depth` tokens. Bridges are saved to
`{output_dir}/{profile}/stage{1,2}.pt`, where the profile is named after the
layers (e.g. `L0-7-15`), so changing `draft_layers` never loads mismatched
bridges.

## Speculative decoding

There is one KV cache, shared by the target and the draft. The draft's layers
(e.g. 0, 7, 15) are the target's own layers, so they read the target's real
keys/values for the entire committed context. There's no draft prefill and no
separate draft cache.

1. **Prefill:** the target processes the prompt.
2. **Draft:** starting from the last committed token, the draft grows the tree
   level by level, one batched forward per level. Each node attends to the
   committed context plus its own ancestors. The draft writes KV only for the
   speculative tokens, at its own layers, and those entries are discarded
   afterwards.
3. **Verify:** the target scores the root plus every tree node in one forward
   with a tree attention mask, writing real KV for them. The cache keeps the
   root and the accepted path.

Training matches this. Stage 2, and Stage 1 in `chained` mode, pass the
target's own layer inputs as `true_layer_inputs`, so position t attends to
the target's KV for positions < t and to the draft's own state only at t. A
test checks that the training-mode logits equal the shared-cache inference
logits.

- **Greedy** (`temperature: 0`): a child is accepted when it equals the
  target's argmax. The output is identical to plain greedy decoding with the
  target, and tests check this for chains, trees, pruned trees and bad drafts.
- **Sampling:** multi-candidate speculative sampling without replacement. It
  reproduces the target's distribution exactly for unpruned trees, and a test
  checks this statistically. `width`/`max_nodes` pruning makes it near-exact,
  the same trade-off EAGLE-2 makes.
- **Every round** commits the accepted tokens plus one bonus token from the
  target.

The target runs through the same layer implementation as the draft (all
layers, no bridges), which matches HF exactly. The autoregressive baseline
uses the same path, so speedups compare like with like.

## Training

- **Stage 1 (feature regression).** The frozen target runs on real text, and
  its residual stream is recorded at each bridge's endpoints. The input is the
  output of the selected layer before the gap. The target is the input the
  next selected layer normally sees. Loss: `α·relMSE + β·(1 − cos)`, with
  position 0 (the attention sink) excluded. Two modes:
  - `mode: teacher` (default): bridges see the target's clean activations.
  - `mode: chained`: bridges see the draft's own upstream output.

  Activations are computed online by default. Alternatively,
  `python -m ssd.data.extract_activations` writes a disk cache that
  `stage1.activation_cache` points at.
- **Stage 2 (distillation).** The full draft runs on real text, trained with
  KL divergence (T=2) against the target's logits plus next-token CE. Logs
  include `top1_agree`, which is the depth-1 greedy acceptance rate.
- **Online self-distillation** (`ssd-train --stage online [--hours H] [--init ckpt]`):
  the target samples endless parallel streams from one fixed prompt
  (`training.online`). It keeps going past end-of-turn, and slides when a
  stream reaches `max_context`. While generating, it records its inputs to the
  draft's layers and its final hidden states, so there is no second target pass
  and no stored dataset. Every `chunk` tokens, the bridges train on the new
  positions, with preceding positions supplying the target's keys/values as
  context. Checkpoints go to `{output_dir}/{profile}/online.pt`, which
  `ssd-run` prefers over stage 2 and stage 1 checkpoints.
- **Self-distillation dataset** (`python -m ssd.data.generate_selfdistill`): the
  stored alternative. The target answers dataset prompts, and training reads
  the resulting `.jsonl` through `data.local_path`.
- **Multi-GPU:** `accelerate launch -m ssd.cli.train --config ...` (DDP). For
  70B, use `model.device_map: auto` in a single process instead.

## Layout

```
configs/                 llama32_1b (dev), llama3_8b, llama3_70b
ssd/
  config.py              SSDConfig (YAML): draft_layers K->layers, bridge, data, training
  runtime.py             model loading, device selection, checkpoint layout
  cli/train.py           ssd-train entry point
  cli/run.py             ssd-run entry point
  models/
    bridges.py           TransitionBridge (mlp 2–3 layers / swiglu; bottleneck or expansion)
    subnetwork_draft.py  SubnetworkDraftModel: sliced frozen layers + bridges
    target_wrapper.py    TargetWrapper: taps residual-stream boundaries of the target
  engine/
    kv_cache.py          KVCache: preallocated per-layer cache with tree-path compaction
    tree_drafter.py      TreeDrafter: level-by-level tree growth (branch / width / max_nodes)
    verify.py            tree mask construction; greedy + speculative-sampling verification
    speculative.py       SpeculativeGenerator loop, TargetRunner, autoregressive baseline
  data/text_stream.py              chat-template rendering + packing into fixed rows
  data/extract_activations.py      optional on-disk activation cache for stage 1
  training/train_stage1_feature.py stage 1 (teacher | chained)
  training/train_stage2_distill.py stage 2 (KL T=2 + CE)
  benchmark/latency_eval.py        (todo)
  benchmark/profile_cuda.py        (todo)
tests/                   test_forward_pass.py, test_training.py, test_engine.py
```

## Conventions

- **Layer indices are 0-based.** Llama-3.2-1B has layers 0..15, Llama-3-8B has
  0..31, and Llama-3-70B has 0..79.
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

## Tests

```bash
pip install -e ".[dev]"
pytest                                   # tiny random Llamas, CPU, ~30s
SSD_TEST_LLAMA3_8B=1 pytest -m llama3_8b  # real Llama-3-8B-Instruct (gated HF repo)
```
