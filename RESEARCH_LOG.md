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

## 2026-09-27 21:20: online run failed on memory, fixed and restarted

- **What happened:** the first 12-hour online run (started 20:07) slowed about
  15x after ~1 hour. From step 60 on, 20 steps took ~15 minutes instead of ~1.
- **Cause:** the training process's footprint reached **19 GB on the 18 GB
  Mac**, with ~17 GB of swap in use. Each slide re-prefilled 32 streams × 512
  tokens at once. PyTorch's caching allocator on the Mac GPU keeps freed
  blocks and may grow to ~1.7x the recommended limit before failing, so it
  crept into swap instead of erroring.
- **Also found:** an orphaned Python process from the 09:23 generation
  benchmark had been alive all day. It was killed. It was idle and probably not
  the cause.
- **Fix** (commit `4f843cb`):
  - `max_context` 2048 → 1024 and `keep_on_slide` 512 → 256.
  - Drop the old cache and free device memory around each slide and after each
    training pass.
  - Log device memory (`mem_gb`) with every metric line.
  - Training rows still get 384 tokens of context, so what the bridges see is
    unchanged.
- **Verification run** (120 steps): memory goes up and down between 10.2 and
  12.6 GB instead of climbing steadily. Top-1 agreement on the stream reached
  ~76–79% by steps 100–120.
- **Restarted from scratch at 21:20:** 12 hours from the same warm start. The
  queue then runs the standard evaluation (full vocabulary and 32k draft
  vocabulary) and the in-domain bioluminescence evaluation, before and after
  online training.

## 2026-09-27 22:45: online run #2 degenerated into a loop, fixed and restarted

- **What happened:** the 21:20 restart ran with stable memory (~12 GB), but by
  22:19 its top-1 agreement read **99%**. That's implausible for a 3-layer
  draft.
- **Diagnosis:** the streams had collapsed into an endless loop of role-header
  tokens (`<|start_header_id|>assistant<|end_header_id|>…`). Stream 0 was
  already looping at 21:30. The distinct-token ratio of a chunk, over all
  streams pooled, fell 0.24 → 0.04 over the first ~1,000 steps. The draft was
  learning to predict the loop, so the run was stopped and its checkpoint set
  aside (`online_degenerate_2130.pt`, unused).
- **Hypothesis 1, refuted:** that trimming a stream (the slide) broke it by
  dropping the `<|begin_of_text|>` attention sink. A direct test (a
  1,000-token stream slid with and without the prompt kept at the front)
  stayed coherent both ways.
- **Actual cause:** free-running past end-of-turn. The 1B occasionally falls
  into the header loop, which it never leaves. With 32 streams over hours, all
  of them eventually fall in. The earlier 1,500-token × 4-stream check was too
  short and too small to catch it.
- **Fix** (kept within the user's design: one fixed prompt, endless):
  - At end of turn, the same prompt is re-inserted as a new user turn, and the
    target answers again in the same conversation.
  - A loop detector re-prompts any stream whose last 64 tokens are less than
    25% distinct.
  - Inserted tokens aren't target samples, so there's no loss on predicting
    them.
  - The distinct metric is now per stream (not pooled), so its values aren't
    comparable to the 0.24 above.
- **Validation** (400 steps, ~20 minutes):
  - distinct stays 0.64–0.68 throughout;
  - ~516 end-of-turn re-prompts; the loop detector essentially never fired;
  - 2.5% of positions masked;
  - top-1 agreement on the stream 53% → 65%, a realistic curve;
  - memory 10.4–11.9 GB;
  - the text is coherent, and topics now drift (bioluminescent bays, tides).
- **Thermal:** the user worried about the Mac's temperature. macOS recorded no
  thermal or performance warning at any check. Apple Silicon throttles itself
  before damage; the main long-run cost is battery wear from heat at 100%
  charge.
- **Restarted:** 12 hours from the same warm start, followed by the same
  evaluations.

## 2026-09-28 11:10: online self-distillation results

- **Run:** `[0,7,15]`, warm-started from the stored self-distillation
  checkpoint, 12 hours (22:45 → 10:45).
- **Size:** 13,057 optimizer steps; **13.4M tokens** generated by the target
  from one fixed prompt.
- **Health:** no crashes. Memory stayed at ~12 GB and macOS recorded no
  thermal warnings. Stream diversity stayed at 0.60–0.67, and the loop
  detector fired only 4 times.
- **On its own stream:** top-1 agreement rose 53% → 78%, and KD loss fell
  4.1 → 1.1.

**In-domain vs. out-of-domain** (greedy, accepted tokens per round, "before"
= warm start, "after" = online):

| Tree | In-domain (3 prompts): before | In-domain: after | Out-of-domain (5 prompts): before | Out-of-domain: after |
|---|---|---|---|---|
| chain, depth 4 | 1.86 | **2.72** (+46%) | 1.94 | 1.73 (−11%) |
| chain, depth 6 | 1.89 | **2.90** (+53%) | 2.01 | 1.75 (−13%) |
| default (24 nodes) | 2.39 | **3.39** (+42%) | 2.50 | 2.23 (−11%) |
| wide (48 nodes) | 2.70 | **4.09** (+51%) | 2.79 | 2.44 (−13%) |

- **Depth-1 acceptance** on the default tree: in-domain 70% → 91%;
  out-of-domain 72% → 64%.
- **Best speed so far:** in-domain chain depth 4 reached **0.92x**, close to
  break-even even on the Mac.

**Per prompt, default tree:**

| Prompt | Before | After |
|---|---|---|
| training prompt ("Tell me about something you find fascinating…") | 2.28 | 3.74 |
| **"Explain how bioluminescence works in deep-sea animals."** (never trained on) | 2.42 | **3.71** |
| "Describe what visitors experience in a bioluminescent bay at night." | 2.46 | 2.72 |
| code (Fibonacci) | 3.21 | 2.32 (−28%, the largest drop) |
| KV cache, math, story, list | 2.16–2.51 | 1.94–2.42 (all slightly lower) |

**Conclusions:**
1. **Training on the target's own stream works, and strongly.** The user's
   test confirms it: acceptance rose +42–53% in-domain, including on a prompt
   the model never trained on.
2. **It specializes.** One-topic training cost 11–13% out-of-domain, and
   code lost the most. The bridges have limited capacity and adapt to whatever
   distribution they see. So the data distribution matters as much as its
   amount, and a single prompt narrows it.
3. **Implication:** online self-distillation over a diverse set of fixed
   prompts (for example, one per stream from EAGLE-3's ShareGPT/UltraChat
   prompts) should give broad gains. That's an untested hypothesis. It would
   also connect to domain-specific drafts: a bridge set per workload.

**Reduced draft vocabulary** (32k, on the online checkpoint, out-of-domain):
acceptance is essentially unchanged (default 2.23 → 2.20), while draft time
per round falls sharply:

| Tree | Draft time: full → 32k | Speedup: full → 32k |
|---|---|---|
| chain, depth 4 | 41 → 28 ms | 0.60 → 0.72x |
| default | 47 → 33 ms | 0.54 → 0.62x |
| wide | 118 → 49 ms | 0.31 → 0.48x |

These are Mac timings, but the direction is hardware-general: a 128k-row
`lm_head` on every draft step is pure overhead for tokens that are almost
never drafted.

## Open questions (as of 2026-09-28 11:10)

- **Diverse prompts:** does online self-distillation with diverse fixed
  prompts give the in-domain-sized gain everywhere, and does it keep improving
  with more tokens?
- **8B and fair comparison:** A100/H100 comparison with EAGLE-3, on
  Llama-3.1-8B-Instruct, with EAGLE-3's training prompts
  (ShareGPT + UltraChat, regenerated by the target), its 5 benchmarks
  (MT-bench, HumanEval, GSM8K, Alpaca, CNN/DM) and its public checkpoint run
  on the same machine. Compare acceptance length τ first.
- **Deeper tree levels:** the mismatch between training and multi-step
  drafting is still open; EAGLE-3's "training-time test" is the known fix.
- **Reduced vocabulary as default:** make `draft_vocab: 32768` the default.

## 2026-09-28: plan for SSD vs EAGLE-3 on H100

- **Principles set by the user:** use EAGLE-3's exact training and test data;
  follow their procedure wherever possible; SSD uses first-middle-last
  `[0,16,31]`.
- **EAGLE-3 recipe, verified from the paper and repo:**
  - Target: Llama-3.1-8B-Instruct, with the public checkpoint
    `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`.
  - Data: ShareGPT (68K) + UltraChat-200K (464K), responses regenerated by the
    target (with vLLM; their script isn't published).
  - Loss on assistant tokens only; max length 2048.
  - AdamW, lr 5e-5, betas (0.9, 0.95), gradient clipping 0.5, fp16, **40
    epochs**, draft vocabulary 32,000.
  - Evaluation: MT-bench / HumanEval / GSM8K / Alpaca / CNN-DM at T=0 and T=1,
    `max_new_token` 1024, 60-node tree.
  - Published T=0 mean: τ 6.23, speedup 4.44x.
- **Main obstacle:** the 40-epoch budget can't be reproduced on one H100.
  The proposal is to also retrain EAGLE-3 at SSD's token budget, for an
  equal-compute comparison next to the official checkpoint.
- **Details:** `PLAN_H100_EAGLE3.md`. It has 5 open decisions: target 3.1,
  training budget, training-time test for SSD, fp16 evaluation, and an optional
  online-training extra.

## 2026-09-28: plan debate (the user challenged my "EAGLE-3 will win" prior)

1. **"SSD runs the real layers, and more of them, not just top-layer
   features."** Conceded in large part. SSD's draft attends to the target's
   true KV at layers 0, 16 and 31 for the whole prefix, and predicts through
   the target's own last layer and head. EAGLE-3 fuses the target's low, middle
   and high features, which is the same intuition. The real difference is
   trained-for-purpose (~0.4B trained parameters) vs frozen-but-native. My
   prior rested on weak 1B evidence, so **τ is now considered genuinely open.**
2. **"Train on several steps at once" (EAGLE-3's "training-time test").** It
   unrolls the draft k steps in training, so later steps see the draft's own
   KV as they will at inference. SSD's training mode extends to it naturally.
   **Decision: SSD gets it.**
3. **"Draft cost doesn't matter if it's 6–7x cheaper."** Partly right. Draft
   cost is paid once per depth: at K = 7, rounds cost ≈1.4 (EAGLE-3) vs ≈1.8
   (SSD) target steps, so SSD needs ≈1.3x EAGLE-3's τ to match its speed. It's
   a handicap of roughly 25%, not a decisive one. **Decisions:** τ and speedup
   are both optimized and both headline (the user rejected making τ primary);
   CUDA graphs are added for the draft path.
