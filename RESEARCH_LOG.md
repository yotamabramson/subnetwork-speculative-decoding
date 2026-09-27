# SSD research log

An append-only record of how this project has evolved: decisions, reversals,
measurements and open questions. **New entries go at the end; earlier text is
never deleted or rewritten.** If an earlier statement turns out to be wrong, a
later entry says so. For how to use the code, see [README.md](README.md).

---

## Goal and idea

**Subnetwork Speculative Decoding (SSD):** use a frozen subset of the target
LLM's own decoder layers as the speculative draft model. Learned MLP
"transition bridges" join the kept layers where layers are skipped. Example:
Llama-3.2-1B layers `[0, 7, 15]`, with one bridge standing in for layers 1–6
and another for layers 8–14.

- **No new weights** apart from the bridges. The draft's layers are the target's
  layers in memory.
- **Trained in two stages:**
  1. **Feature regression:** each bridge learns to map the target's hidden state
     at one layer boundary to its hidden state at the next kept layer.
  2. **Logit distillation:** the full draft is trained end to end with KL
     divergence (T=2) against the target's logits, plus next-token
     cross-entropy.
- **Decoding:** tree speculative decoding, with greedy or stochastic verification.
- **Final target:** Llama-3-8B/70B on A100/H100.
- **Development:** Llama-3.2-1B-Instruct on a MacBook M3 Pro (18 GB). The 8B
  doesn't fit there, and the Mac's numbers are only indicative. The user asked
  explicitly for **no Mac-specific optimization**.

The project began from a written spec called "BRIDLE" (layers {1, 16, 32}
for 8B, {1, 40, 80} for 70B). It was renamed SSD on day 1.

---

## 2026-09-26 (evening): scaffold

- **Package layout:** `ssd/` (models, engine, training, data, cli), plus
  `configs/` and `tests/`, with a local `.venv` (Python 3.11, torch 2.14,
  transformers 5.17).
- **Layer indices are 0-based.** The spec's {1, 16, 32} don't exist for
  Llama-3-8B, whose layers are 0..31. They were read as 1-based first/middle/last
  and mapped to `[0, 15, 31]`; the 70B equivalent is `[0, 39, 79]`.
- **Bridges:** `out = x + MLP(RMSNorm(x))`. The MLP is 2–3 layers or SwiGLU, at
  full width by default.
  - Only the output projection is initialised small (std 1e-3), so an untrained
    bridge is about the identity, i.e. plain layer skipping.
  - Inner layers use standard fan-in initialisation. Initialising every layer at
    1e-3 would make the first layer's gradients about 1e-6.
- **The draft re-implements the decoder-layer forward** on top of the layer's
  own submodules instead of calling HF's `LlamaDecoderLayer`. This allows
  arbitrary tree masks and our own KV cache. A test shows it matches HF exactly
  when every layer is selected.
- **Base modules are referenced, not registered**, so `state_dict()` holds only
  the bridges.

## 2026-09-26: hardware decision

- **8B doesn't fit the Mac:** ~16 GB of bf16 weights against about 13 GB
  usable by the GPU.
- **The user's Windows GPU has only 12 GB.**
- **Decision:** develop on the Mac with Llama-3.2-1B-Instruct (same
  architecture and tokenizer, 16 layers). Rent an A100/H100 later for 8B.
- **Model access:** the gated `meta-llama` repos are reached through
  `HF_TOKEN` in `.env`, which is git-ignored and loaded at runtime.

## 2026-09-26: per-K layer subsets, tried and dropped

- **The idea:** the user specified a map from draft depth K to a layer subset,
  e.g. K=1–2 → `[0,7,15]` and K=3 → `[0,1,7,14,15]`. It was implemented as
  "profiles", each with its own bridges.
- **The problem**, spotted by the user: if later draft steps use a larger
  subset, the new layers (1 and 14) have no KV entries for the tokens drafted
  earlier. Even the shared layers (7 and 15) have *different* KV, because their
  inputs come from different bridges.
- **The possible fix**, not built: a separate cache per subset, with
  "catch-up" forwards.
- **The user's decision:** all options are cumbersome, so use **a single layer
  subset for every draft step**. The config became `draft_layers: [...]`, and
  per-step subsets are explicitly unsupported.

## 2026-09-26: training pipeline

- **Stage 1 (teacher mode):** the frozen target runs on real text; each bridge
  regresses from the target's activation at its source boundary to the one at
  its target boundary. Transformer layers take no part in the bridge's
  computation.
  - Loss: `α·relMSE + β·(1−cos)`, with position 0 (the attention sink) excluded.
  - Activations are computed online per batch by default; an on-disk cache is
    optional.
- **Stage 2:** end-to-end KL (T=2) plus CE against the target's logits.
- **Memory fix:** Stage 2 first thrashed memory on the Mac (74 tok/s), because
  it materialized several full 128k-vocab logit tensors. Computing the
  `lm_head` and loss in token chunks under activation checkpointing brought it
  back to 526 tok/s, with identical loss values.
- **Stage 1 plateaus early:** for `[0,7,15]` it settles by ~400 steps at
  cosine ≈ 0.70 for the bridge into layer 7 and ≈ 0.77 for the one into layer 15.

## 2026-09-26: discussion of bridge design

- **Why an additive skip?** It isn't needed for expressiveness, since `W·x` can
  be the identity. It matches the residual stream's additive structure, gives
  an identity starting point, and separates magnitude (carried by the skip)
  from direction (the MLP only sees the normalized input).
- **Bridges carry no information from other tokens:** they work per token.
  - I suggested a causal convolution, cross-layer KV reuse, or an EMA state.
  - The user rejected the convolution as foreign to transformers.
  - **Position adopted:** rely on the kept layers' attention for mixing across
    tokens; the bridges only map per-token distributions.
- **Possible later option:** LoRA on the kept layers, applied on the draft path
  only.

## 2026-09-26: comparison with EAGLE

- **EAGLE-1/2:** a fusion layer (FC over the concatenation of the next token's
  embedding and the target's top-layer feature), then **one new full decoder
  layer** with its own KV cache, then the target's LM head.
  - It conditions on the target's features for the whole committed prefix.
  - Published accept lengths are about 3.6–4.5 on 7–8B.
- **EAGLE-3** fuses low, middle and high-level features, and trains with
  "training-time test" (multi-step rollouts). It also uses a reduced draft
  vocabulary.
- **Key contrast:** EAGLE's draft layer is new, so it can't share the target's
  KV cache. SSD's draft layers *are* target layers, so SSD can (see the next
  entry).

## 2026-09-27 00:00: bug in the design, then a fix: share the target's KV cache

- **What was wrong:** the first engine gave the draft its own KV cache, with a
  "draft prefill" over the prompt through its bridged path. Every committed
  token was then re-processed by the draft. So layers 7 and 15 attended to
  *approximate*, bridge-derived keys and values for the entire context, even
  though the target had already computed the true ones.
- **How it was found:** the user asked why layers 0, 7 and 15 wouldn't simply
  use the full KV cache. They were right; the design was flawed.
- **The fix:** one KV cache shared by target and draft, keyed by base-layer
  index. Each round:
  1. The draft processes the root and the tree, attending to the target's real
     KV for the committed prefix, and appends temporary entries only at its own
     layers.
  2. Those entries are cropped before verification.
  3. The target verifies, writing real KV, and the accepted path is compacted
     into place.
  There is no draft prefill and no catch-up.
- **Training was changed to match.** Stage 2 (and chained Stage 1) takes
  `true_layer_inputs`: position t attends to the target's KV for positions < t
  and to its own state only at t. A test shows training-mode logits equal
  shared-cache inference logits.
- **Remaining mismatch, shared with EAGLE-1/2:** training matches inference
  exactly only at depth 1. Deeper tree nodes also attend to the root's and
  their ancestors' draft-computed KV, which training never shows.
- The training run in progress was killed and restarted from scratch.

## 2026-09-27: engine details

- **Tree drafting:** configurable `depth` (K), per-depth `branch` and `width`,
  and `max_nodes`. Each level is one batched draft forward with a tree mask.
- **Greedy verification** gives output identical to the target's own greedy
  decoding. Tests cover chains, trees, pruned trees and good and bad drafts,
  in fp32.
- **Sampling verification:** multi-candidate speculative sampling without
  replacement (Yang et al. 2024). It's exact for unpruned trees; a statistical
  test checks this, and it fails with TV ≈ 0.8 when verification is
  deliberately broken. Pruned trees are near-exact, like EAGLE-2.
- **Acceptance by depth:** `ssd-run` reports it as `d1=… d2=…`, each
  conditional on all shallower depths being accepted.

## 2026-09-27 (night 1): first real results, Llama-3.2-1B on the Mac GPU, greedy

5 prompts (KV-cache explanation, Python code, math word problem, short story,
list). "Accepted" is tokens per round including the bonus token; a plain
target step takes ~24 ms.

| Tree | `[0,7,15]` untrained: acc. | `[0,7,15]` untrained: speedup | `[0,7,15]` trained: acc. | `[0,7,15]` trained: speedup | `[0,1,7,14,15]` trained: acc. | `[0,1,7,14,15]` trained: speedup |
|---|---|---|---|---|---|---|
| chain, depth 4 | 1.01 | 0.34x | 1.70 | 0.56x | 2.03 | 0.58x |
| chain, depth 6 | 1.00 | 0.27x | 1.72 | 0.44x | 2.08 | 0.46x |
| default (depth 4, branch [3,2,2,1], width 8, 24 nodes) | 1.02 | 0.24x | 2.16 | 0.51x | 2.58 | 0.55x |
| wide (depth 6, 48 nodes) | 1.03 | 0.13x | 2.37 | 0.30x | 2.88 | 0.33x |

- **Final Stage 2 top-1 agreement:** 49% for `[0,7,15]`, 57% for
  `[0,1,7,14,15]`.
- **Training clearly works,** but everything is slower than plain decoding.
  - For the default tree, a round costs ~97 ms (draft ~48 ms + verify ~49 ms)
    for 2.16 tokens.
  - On the 1B, the 128k-row `lm_head` is ~21% of the whole model, so even a
    3-layer draft costs about half a target step per level.

## 2026-09-27: expectation for 8B (estimate, not measured)

- **Acceptance:** about the same as on the 1B (~2.0–2.5 tokens per round for 3
  layers). Bigger models are more redundant across layers, but 3 of 32 layers
  is a harder cut than 3 of 16.
- **Speed looks much better:** a draft level costs ~16% of a target step on 8B,
  against ~45% on the 1B.
- **Rough prediction:** ~1.4–1.5x on an A100, against EAGLE-2's published ~3x.
- **Biggest cheap lever:** more data, and self-distillation.

## 2026-09-27: side note on process

While the night queue was running, I (Claude) started an unrequested
performance branch (reduced draft vocabulary and copy-free GQA) in a separate
worktree. The user objected, since it hadn't been asked for and its test runs
briefly slowed training. The branch was kept, reviewed later, and merged at
19:50 (see below).

## 2026-09-27 (day): layer choice and self-distillation

- **Self-distillation data:** the target answered 9,216 UltraChat prompts,
  3.08M tokens, sampled at temperature 0.7 with top-p 0.9 applied within the
  top 64 tokens.
  - HF `generate` managed only ~120–180 tok/s on the Mac GPU, so we wrote our
    own batched sampler, which reached ~440 tok/s.
  - A full-vocab top-p sort cost more than the forward pass, so sampling was
    changed to top-k 64 first.
- **Copy-free grouped-query attention:** SDPA `enable_gqa` instead of repeating
  K/V gave 1.7x faster batched decoding. It's numerically identical, and the
  single-sequence timings didn't change.

| Draft | Data | Chain depth 4: acc. | Chain depth 4: speedup | Default tree: acc. | Default tree: speedup | Default tree: d1 |
|---|---|---|---|---|---|---|
| `[0,7,15]` | UltraChat, 1k + 1k steps | 1.70 | 0.55x | 2.16 | 0.51x | 62% |
| `[0,1,15]` | UltraChat | 1.61 | 0.53x | 2.08 | 0.50x | – |
| `[0,1,7,14,15]` | UltraChat | 2.03 | 0.58x | 2.58 | 0.55x | – |
| `[0,1,2,4,15]` | UltraChat | 1.87 | 0.53x | 2.46 | 0.52x | – |
| `[0,7,15]` | **self-distillation, 1k + 3k steps** | **1.93** | **0.62x** | **2.51** | **0.59x** | **72%** |

- **Lower-layer-heavy subsets lose slightly at both 3 and 5 layers.** Spreading
  kept layers so no bridge spans too big a gap works better. Stage 1 cosines:
  `[0,1,15]` 0.72; `[0,1,2,4,15]` 0.88 for the short bridge into layer 4 and
  0.76 for the long one into layer 15.
- **Self-distillation is the clearest win:** +14–16% accepted tokens at
  identical cost. It reaches the 5-layer UltraChat acceptance with only 3
  layers.

## 2026-09-27: numerical finding, bf16 greedy drift

- **Observation:** in about half the bf16 runs on the Mac GPU, greedy
  speculative output diverges from plain decoding partway through; in the case
  tested, at token 161 of 256. The same case in fp32 is identical.
- **Cause:** a batched tree verification sums in a different order than
  single-token decoding, which flips near-tied argmaxes. This is not a logic
  bug.
- **Consequence:** "identical to autoregressive" is guaranteed only in fp32.

## 2026-09-27: Mac GPU matmul behavior (noted, deliberately not optimized)

- **What we saw:** matmuls with up to 8 rows cost about the same as 1 row
  (~3.8 ms for the 1B `lm_head`), and get markedly slower beyond that. So
  verifying 25 tree nodes costs about 2 target steps, and wide trees lose.
- **What we did about it:** nothing. The user said to optimize for A100/H100
  only, so no Mac-specific tree tuning was done.

## 2026-09-27 (evening): speed work (hardware-agnostic) and reduced draft vocabulary

- **Merged the perf branch:**
  - `drafting.draft_vocab`: the draft scores only the N most frequent tokens.
    Verification uses the full vocabulary, so outputs are unchanged. The top
    32k tokens cover 99.35% of the target's self-distilled output (8k: 90.4%,
    16k: 96.0%).
  - Bridges cast to bf16 at inference.
  - The accepted-path cache compaction is skipped when it's already in place.
- **Training mode with a context-only prefix:** the draft runs only on the
  loss positions, with the target's KV over a longer window.
- **Re-evaluating the best checkpoint** (stored self-distillation) with this
  code, on the same 5 prompts:

| Tree | Accepted | d1 | Speedup |
|---|---|---|---|
| chain, depth 4 | 1.94 | 54% | 0.66x |
| chain, depth 6 | 2.01 | 55% | 0.54x |
| default | 2.50 | 72% | 0.60x |
| wide | 2.79 | 75% | 0.35x |

## 2026-09-27 (evening): online self-distillation, the user's idea

- **The idea:** don't store data at all. The target keeps generating, and the
  bridges train on its output as it's produced.
- **Refined by the user:** no new prompts. The target runs endlessly from
  **one fixed prompt**.
- **Implementation** (`ssd-train --stage online`):
  - 32 parallel streams from the same prompt, sampled at temperature 0.8 with
    top-p 0.95 within the top 64 tokens.
  - Streams continue past end-of-turn, so the model writes further turns
    itself, and slide when they reach 2048 tokens (keep 512, re-prefill at
    position 0).
  - While generating, the target records its inputs to the draft's layers and
    its final hidden states, so there's no second target pass.
  - Every 128 tokens per stream, the bridges train on those positions, with the
    384 preceding positions as context.
  - Constant LR 5e-5 after warmup; warm start from the stored
    self-distillation checkpoint.
- **Observation:** with one prompt ("Tell me about something you find
  fascinating, in detail."), every stream chose the **same topic,
  bioluminescence**. The text stays coherent, with no loops, well past 1,500
  tokens.
- **The user's framing:** keep it that way deliberately. If acceptance improves
  more in-domain (bioluminescence prompts) than on code, math and stories, the
  online training is demonstrably what's helping.
- **The run:** 12 hours, started 20:07. Afterwards, evaluation on the standard
  5 prompts (full vocabulary and 32k draft vocabulary) and on 3 in-domain
  prompts, for both the checkpoint before online training and the one after.
- **Early numbers:** step 60 reached 70% top-1 agreement on its own stream
  (measured on training text); generation runs at ~555 tok/s.

## Open questions (as of 2026-09-27 20:30)

- **Online training:** does it help beyond its own topic? (Pending the run
  above.)
- **Deeper tree levels:** acceptance there is limited by the mismatch between
  training and multi-step drafting. EAGLE-3's "training-time test" is the
  known fix.
- **8B / A100 behavior:** the real speedup, and whether 3 layers is viable or
  5–7 are needed. Stage 1 bridge cosines on 8B are the cheapest first signal.
- **Self-distillation on a 5-layer draft** (`[0,1,7,14,15]`), not yet run.
