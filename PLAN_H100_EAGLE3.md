# Plan: SSD vs EAGLE-3 on Llama-3-8B-class, single H100

Status: **draft, awaiting decisions 1, 2, 4, 5 (section 3)**. Written 2026-09-28;
decision 3 and the metrics policy were settled the same day.

## 1. Principles (set by the user)

1. Use **the exact same training and test data** as EAGLE-3.
2. **Wherever possible, do what the EAGLE-3 authors did** to reach their published results.
3. SSD uses only the **first-middle-last** layer methodology: `[0, 16, 31]` on the 32-layer 8B.

## 2. What EAGLE-3 did (verified 2026-09-28)

Sources: the EAGLE-3 paper (arXiv 2503.01840, HTML version); the SafeAILab/EAGLE repo
(`README`, `eagle/traineagle3/{main.py, config.json, ds_config.json}`,
`eagle/evaluation/gen_ea_answer_llama3chat.py`).

| Item | EAGLE-3 | Source |
|---|---|---|
| Target model | **Llama-3.1-8B-Instruct** (their 8B results; no EAGLE-3 checkpoint for Llama-3.0) | paper, README |
| Public draft checkpoint | `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` | README |
| Training prompts | ShareGPT (~68K) + UltraChat-200K (~464K entries) | paper |
| Training responses | "We call the target model to generate responses rather than using a fixed dataset." Their generated file is local (`…/vllm/gedata/l318b.jsonl`), so generation was done with vLLM. **Their generation script and sampling settings are not published.** | paper, main.py |
| Loss mask | assistant tokens only; user and instruction tokens masked | main.py |
| Chat template | `tokenizer.apply_chat_template(messages)` | main.py |
| Max length | 2048 | main.py |
| Optimizer | AdamW, betas (0.9, 0.95), lr 5e-5, WarmupDecayLR (12k warmup, 800k total steps), weight decay 0 | paper, ds_config |
| Gradient clipping | 0.5 | paper, ds_config |
| Precision | fp16 (DeepSpeed ZeRO-2) | ds_config |
| Batch | micro-batch 1 per GPU, gradient accumulation 2, multi-GPU (GPU count not stated) | ds_config |
| Epochs | **40** | main.py |
| Draft vocabulary | **32,000** (`draft_vocab_size`) | config.json |
| Benchmarks | MT-bench, HumanEval, GSM8K, Alpaca, CNN/DailyMail; question files `eagle/data/<bench>/question.jsonl` | paper, eval script |
| Eval protocol | batch 1; T=0 and T=1; `max_new_token` 1024; their system prompt; MT-bench turn 2 uses the model's own turn-1 answer; 3 warmup runs; records `idxs`, `new_tokens`, `wall_time`; `evaluation/speed.py` computes the ratio vs. the baseline script | eval script, README |
| Eval dtype | fp16 | eval script |
| Tree | script defaults `total_token=60`, `depth=5`, `top_k=10`; the paper says EAGLE-3 uses "depth 6 to 8 while keeping the number of nodes the same as EAGLE-2". **Unresolved; settle it by reproducing their numbers.** | eval script, paper |
| Published result, 8B, T=0 | τ = 6.13 / 6.74 / 6.23 / 6.70 / 5.34 (MT-bench, HumanEval, GSM8K, Alpaca, CNN/DM), **mean τ 6.23, mean speedup 4.44x** | paper table |
| Published result, 8B, T=1 | mean τ 4.92, mean speedup 3.45x | paper table |

## 3. Decisions needed before renting

1. **Target model:** use **Llama-3.1-8B-Instruct**, not Llama-3-8B-Instruct. It's the only 8B that EAGLE-3 published and released a checkpoint for. It has the same architecture and tokenizer, so `[0,16,31]` is unaffected. *(Recommended.)*
2. **Training budget:** EAGLE-3 trained **40 epochs** over ~532k regenerated
   conversations on several GPUs, which is billions of tokens. On one H100
   (~20–25k tok/s, dominated by the 8B target's forward), one epoch takes
   roughly 5–8 h, so 40 epochs isn't feasible. Options:
   - **(a)** Train SSD for about 1–2 epochs and compare with the official
     checkpoint, stating the ~20–40x training-budget gap explicitly.
   - **(b)** Also retrain EAGLE-3 ourselves with *their* script, on the same
     data and the **same token budget** as SSD. This gives an equal-compute
     comparison alongside the official upper bound, at +8–12 H100-hours.
     *(Recommended.)*
3. **Training-time test: DECIDED (2026-09-28), yes.** SSD gets multi-step
   training as a real part of the method (see Phase 0). It's a training
   technique, not a layer choice, so principle 3 is unaffected.
4. **Precision:** evaluate everything in **fp16**, as their evaluation does.
   SSD's bridges stay fp32 for training. *(Recommended.)*
5. **Online self-distillation:** principle 1 means the main SSD run uses the
   same static regenerated data as EAGLE-3. Should a run using online
   self-distillation over the same prompts be added as a clearly labeled
   extra, or left out?

## 4. Phases

### Phase 0: on the Mac, before renting (no GPU cost)
- `configs/llama31_8b_subnetwork.yaml`: `draft_layers: [0, 16, 31]`,
  `draft_vocab: 32000`, tree matching EAGLE-3 (`max_nodes 60`, `branch/width 10`,
  depth TBD by Phase 1).
- **Data pipeline:**
  - Download ShareGPT (`Aeala/ShareGPT_Vicuna_unfiltered`, the standard EAGLE
    source; verify the exact file on the day) and `HuggingFaceH4/ultrachat_200k`.
  - Normalize both to `messages`.
  - Add a **vLLM regeneration script**: for every conversation, regenerate each
    assistant turn with Llama-3.1-8B-Instruct, conditioned on the target's own
    earlier turns. Their sampling settings are unpublished, so follow SpecForge's
    regeneration defaults if available, otherwise greedy, and record the choice.
- **Training on conversations:** SSD currently packs raw token streams and
  trains on every token. For parity: one conversation per row, up to 2048
  tokens, with the **assistant-only loss mask**.
- **Draft vocabulary:** top 32,000 tokens by frequency in the regenerated data,
  the same idea as EAGLE-3.
- **Benchmark harness:** an SSD counterpart of `gen_ea_answer_llama3chat.py`,
  using the same `question.jsonl` files, system prompt, multi-turn handling,
  `max_new_token=1024`, 3 warmups and output format. Their `speed.py` then
  scores SSD and EAGLE-3 identically.
- **Multi-step training ("training-time test") for SSD:** unroll the draft
  for k steps in training (k ≈ EAGLE-3's tree depth). At step j, a position
  attends to the target's true KV for the prefix, plus the draft's own KV from
  its previous j−1 steps; there's a loss at every step. It extends the existing
  `true_layer_inputs` mode (true prefix KV plus own KV on the diagonal) to a
  widening band of draft KV. The target's forward pass is computed once, and
  the draft's compute grows about k-fold.
- **Checks:**
  - Tests on tiny models.
  - An end-to-end dry run on the Mac with the 1B on a few hundred
    conversations and 2 questions per benchmark.
  - **This code has never run on CUDA,** so Phase 1 starts with the test suite
    on the H100.

### Phase 1: H100 setup and **reproducing EAGLE-3** (~1–2 h)
- Set up the environment. Run our tests on CUDA.
- Run the official EAGLE-3 checkpoint and baseline scripts on MT-bench at T=0,
  and find the tree depth that reproduces their τ ≈ 6.13.
- **Gate:** τ within ~5% of the paper, with speedup in the published range.
  Otherwise, stop and investigate before spending on training.

### Phase 2: regenerate training data with vLLM (~4–8 h)
- ~532k conversations; roughly 300M+ generated tokens (to be measured on a
  sample first).
- Save to disk. This one dataset feeds SSD and the equal-budget EAGLE-3 retrain.

### Phase 3: train SSD `[0,16,31]` (budget per decision 2)
- Stage 1 (short; it plateaus early), then Stage 2 on the regenerated data
  with the assistant-only mask.
- Checkpoint regularly. Watch the top-1 agreement at every unrolled depth on a
  held-out split.
- **Speed work (both methods optimized, not just τ):**
  - CUDA graphs for SSD's draft path. At batch 1, 3 layers plus 2 bridges are
    likely limited by kernel-launch overhead rather than memory bandwidth.
  - Profile both the draft and verify phases on the H100.
  - Tune SSD's tree for speed, not only for τ.

### Phase 3b (if decision 2b): retrain EAGLE-3 with the same data and token budget
- Their `eagle/traineagle3` script, or SpecForge, on 1 GPU.

### Phase 4: evaluation (~2–3 h)
- **For each** of the 5 benchmarks × T ∈ {0, 1}:
  - baseline (autoregressive);
  - EAGLE-3 official;
  - EAGLE-3 equal-budget (if 3b);
  - SSD with EAGLE-3's tree settings;
  - SSD with its own best tree, reported separately.
- **Metrics:** τ and speedup, both headline. **DECIDED (2026-09-28):** both are
  optimized, and neither is secondary. τ is hardware-independent; speedup is
  against each framework's own baseline, reported with absolute tok/s.

### Phase 5: write-up
- Append the results table and conclusions to `RESEARCH_LOG.md`.

## 5. Estimates

| | H100 hours |
|---|---|
| Phase 1 | 1–2 |
| Phase 2 | 4–8 |
| Phase 3 (1–2 epochs) | 6–14 |
| Phase 3b (optional) | 8–12 |
| Phase 4 | 2–3 |
| **Total** | **~13–27 without 3b, ~21–39 with it** |

## 6. Expectations and risks (honest)

- **SSD will likely have a lower τ than EAGLE-3.**
  - On the 1B, SSD reached τ ≈ 2.2–2.5 at depth 4 with a 24-node tree.
  - EAGLE-3 conditions on the target's top-layer features and has
    training-time test.
  - Its draft is also cheaper per step: 1 layer, against SSD's 3 layers plus
    bridges. With a 32k vocabulary, that's roughly 3% vs 10% of a target step
    on 8B.
- **What the experiment answers:** how far a pure subnetwork draft (no new
  layers) is from the state of the art, under identical data and protocol.
- **Speed arithmetic** (discussed 2026-09-28): speedup ≈ τ / (1 + K·c + v).
  Per-step draft cost c is ≈0.045 of a target step for EAGLE-3 and ≈0.10 for
  SSD (8B, 32k vocab). At K = 7, rounds cost ≈1.4 vs ≈1.8 target steps. SSD
  therefore needs ≈1.3x EAGLE-3's τ to match its speed, or must close the gap
  on the cost side (CUDA graphs, fewer steps).
- **The information argument is closer than first stated:** SSD's draft reads
  the target's *true* KV at layers 0, 16 and 31 for the whole prefix, and
  predicts through the target's real last layer and head. EAGLE-3 fuses
  low/mid/high target features. The real difference is trained-for-purpose
  (~0.4B parameters) vs frozen-but-native (~0.65B frozen + 67M bridges).
- **Risks:**
  - Unpublished regeneration settings (mitigated by the Phase 1 gate and a
    documented choice).
  - The unresolved tree depth (Phase 1 settles it).
  - First CUDA run of our code.
  - The multi-turn conversion of our training data.
