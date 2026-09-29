# H100 runbook: SSD vs EAGLE-3 on Llama-3.1-8B-Instruct

Step-by-step commands for the rented machine. The rationale is in
[PLAN_H100_EAGLE3.md](PLAN_H100_EAGLE3.md). Every phase ends with a
**gate**: if it fails, stop and investigate before spending more GPU hours.

Assumed machine: 1× H100 80GB, CUDA 12.x drivers, ≥300 GB free disk, Python
3.11. On Hugging Face, accept the Llama-3.1 license for your account first.

## Phase 0: setup (~20 min)

```bash
git clone git@github.com:yotamabramson/subnetwork-speculative-decoding.git ssd && cd ssd
echo "HF_TOKEN=hf_..." > .env                       # git-ignored

# Three environments: our code, EAGLE's pinned stack, and vLLM for regeneration.
python3.11 -m venv .venv        && .venv/bin/pip install -e ".[dev]"
python3.11 -m venv .venv-eagle  && python3.11 -m venv .venv-vllm
.venv-vllm/bin/pip install vllm

mkdir -p third_party
git clone https://github.com/SafeAILab/EAGLE.git third_party/EAGLE
git -C third_party/EAGLE checkout cb7e084           # the commit the plan was verified against
.venv-eagle/bin/pip install -r third_party/EAGLE/requirements.txt shortuuid deepspeed
export EAGLE="env PYTHONPATH=$PWD/third_party/EAGLE $PWD/.venv-eagle/bin/python"

.venv/bin/python -m pytest -q                        # our suite, now on CUDA
nvidia-smi
```
**Gate:** all tests pass.

## Phase 1: reproduce EAGLE-3 and settle the tree depth (~1–2 h)

```bash
cd third_party/EAGLE
for d in 5 6 7 8; do
  $EAGLE -m eagle.evaluation.gen_ea_answer_llama3chat --use_eagle3 \
    --ea-model-path yuhuili/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path meta-llama/Llama-3.1-8B-Instruct \
    --bench-name mt_bench --depth $d --temperature 0 --model-id eagle3-d$d \
    --answer-file ../../results/mt_bench/eagle3-d$d-t0.jsonl
done
$EAGLE -m eagle.evaluation.gen_baseline_answer_llama3chat \
  --ea-model-path yuhuili/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path meta-llama/Llama-3.1-8B-Instruct \
  --bench-name mt_bench --temperature 0 --model-id eagle-baseline --answer-file ../../results/mt_bench/eagle-baseline-t0.jsonl
cd ../..
for d in 5 6 7 8; do
  .venv/bin/python -m ssd.benchmark.eagle_report --tokenizer meta-llama/Llama-3.1-8B-Instruct --benches mt_bench \
    --pair "eagle3-d$d=results/{bench}/eagle3-d$d-t0.jsonl:results/{bench}/eagle-baseline-t0.jsonl"
done
```
**Gate:** some depth gives MT-bench τ within ~5% of the paper's **6.13**, with
a speedup in the paper's range (4.40x). Record that depth as `D`, and set
`drafting.depth: D` in `configs/llama31_8b_eagle3.yaml`.

## Phase 2: regenerate the training data (measure, then run; ~8–13 h)

```bash
.venv/bin/python -m ssd.data.eagle3_data prepare --out data/eagle3_prompts.jsonl   # ShareGPT + UltraChat sft+gen
wc -l data/eagle3_prompts.jsonl                                                     # expect ≈ 68K + 464K

# 1% sample first: throughput, drop rate, and a look at the text
.venv-vllm/bin/python -m ssd.data.eagle3_data regenerate --config configs/llama31_8b_eagle3.yaml \
  --inp data/eagle3_prompts.jsonl --out data/eagle3_regen.jsonl --backend vllm --limit 5000
# then the rest (resumable: it skips everything already written or dropped)
nohup .venv-vllm/bin/python -m ssd.data.eagle3_data regenerate --config configs/llama31_8b_eagle3.yaml \
  --inp data/eagle3_prompts.jsonl --out data/eagle3_regen.jsonl --backend vllm > logs/regen.log 2>&1 &

# held-out test split (1%) shared by SSD and the EAGLE-3 retrain
.venv/bin/python - <<'EOF'
import json, hashlib
rows = [l for l in open("data/eagle3_regen.jsonl") if l.strip()]
is_test = [int(hashlib.md5(json.loads(r)["id"].encode()).hexdigest(), 16) % 100 == 0 for r in rows]
test = [r for r, t in zip(rows, is_test) if t]
train = [r for r, t in zip(rows, is_test) if not t]
open("data/eagle3_train.jsonl", "w").writelines(train); open("data/eagle3_test.jsonl", "w").writelines(test)
print(len(train), len(test))
EOF
```
**Gate:** the sample text looks right (no loops or empty answers), and
throughput projects to within budget. The `--max-tokens 2048` default plus the
early drop avoid wasting time on conversations over 2048 tokens.

## Phase 3: train (budget: 1 epoch each, same data order and size)

Set `data.local_path: data/eagle3_train.jsonl` in the config, and set
`training.stage2.steps` to ⌈(train conversations) / 16⌉.

```bash
# memory smoke test: one full-length 7-step training step
.venv/bin/ssd-train --config configs/llama31_8b_eagle3.yaml --stage 2 --steps 3 --output-dir outputs/smoke
nvidia-smi --query-gpu=memory.used --format=csv                                     # while it runs

# SSD: stage 1 then stage 2 (writes the EAGLE-style 32k draft vocabulary first)
nohup .venv/bin/ssd-train --config configs/llama31_8b_eagle3.yaml > logs/ssd_train.log 2>&1 &

# EAGLE-3 retrain at the same budget (after SSD, or on a second GPU):
# their script hardcodes 40 epochs and an 800k-step schedule, so patch those.
sed -i 's/"num_epochs": 40/"num_epochs": 1/' third_party/EAGLE/eagle/traineagle3/main.py
.venv/bin/python - <<'EOF'
import json; p = "third_party/EAGLE/eagle/traineagle3/ds_config.json"; c = json.load(open(p))
n = sum(1 for _ in open("data/eagle3_train.jsonl"))
c["gradient_accumulation_steps"] = 16                        # = SSD's 16 conversations/step on 1 GPU
c["scheduler"]["params"]["total_num_steps"] = n // 16
c["scheduler"]["params"]["warmup_num_steps"] = 500
json.dump(c, open(p, "w"), indent=2)
EOF
cd third_party/EAGLE/eagle/traineagle3
nohup env PYTHONPATH=$PWD/../../.. $PWD/../../../../.venv-eagle/bin/deepspeed --num_gpus 1 main.py --deepspeed_config ds_config.json \
  --basepath meta-llama/Llama-3.1-8B-Instruct --trainpath ../../../../data/eagle3_train.jsonl \
  --testpath ../../../../data/eagle3_test.jsonl --savedir ../../../../outputs/eagle3_retrain > ../../../../logs/eagle3_train.log 2>&1 &
cd ../../../..
cp third_party/EAGLE/eagle/traineagle3/config.json outputs/eagle3_retrain/state_0/   # EaModel needs it
# verify the saved state loads with EaModel (weights + d2t/t2d draft-vocab buffers) before Phase 4
```
**Gate:** neither run diverges. SSD's per-depth top-1 (`top1_d1…d7`) rises
on the held-out split.

## Phase 4: evaluate (~2–3 h)

```bash
D=<depth from Phase 1>
for T in 0.0 1.0; do for b in mt_bench humaneval gsm8k alpaca sum; do
  # SSD and its own baseline (same code path)
  .venv/bin/python -m ssd.benchmark.eagle_bench --config configs/llama31_8b_eagle3.yaml --mode ssd \
    --bench-name $b --temperature $T --depth $D --model-id ssd --answer-file results/$b/ssd-t$T.jsonl
  .venv/bin/python -m ssd.benchmark.eagle_bench --config configs/llama31_8b_eagle3.yaml --mode baseline \
    --bench-name $b --temperature $T --model-id ssd-baseline --answer-file results/$b/ssd-baseline-t$T.jsonl
  # EAGLE-3 official, EAGLE-3 retrained, and EAGLE's baseline
  (cd third_party/EAGLE && for m in yuhuili/EAGLE3-LLaMA3.1-Instruct-8B:eagle3 ../../outputs/eagle3_retrain/state_0:eagle3-retrain; do
     $EAGLE -m eagle.evaluation.gen_ea_answer_llama3chat --use_eagle3 --ea-model-path ${m%%:*} \
       --base-model-path meta-llama/Llama-3.1-8B-Instruct --bench-name $b --depth $D --temperature $T \
       --model-id ${m##*:} --answer-file ../../results/$b/${m##*:}-t$T.jsonl; done
   $EAGLE -m eagle.evaluation.gen_baseline_answer_llama3chat --ea-model-path yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
     --base-model-path meta-llama/Llama-3.1-8B-Instruct --bench-name $b --temperature $T \
     --model-id eagle-baseline --answer-file ../../results/$b/eagle-baseline-t$T.jsonl)
done; done

for T in 0.0 1.0; do .venv/bin/python -m ssd.benchmark.eagle_report --tokenizer meta-llama/Llama-3.1-8B-Instruct \
  --pair "ssd=results/{bench}/ssd-t$T.jsonl:results/{bench}/ssd-baseline-t$T.jsonl" \
  --pair "eagle3=results/{bench}/eagle3-t$T.jsonl:results/{bench}/eagle-baseline-t$T.jsonl" \
  --pair "eagle3-retrain=results/{bench}/eagle3-retrain-t$T.jsonl:results/{bench}/eagle-baseline-t$T.jsonl"; done
```
Then append the tables and conclusions to `RESEARCH_LOG.md`.

## Tested on the Mac (dry run, 2026-09-29)

The whole pipeline ran with Llama-3.2-1B in place of the 8B, using
`configs/experiments/llama32_1b_eagle3_dry.yaml`:
- **prepare:** 36 prompts (ShareGPT plus both UltraChat splits);
- **regenerate:** turn by turn, with the local backend;
- **training:** EAGLE-style draft vocabulary, then Stage 1, then Stage 2
  with 3-step multi-step training;
- **eagle_bench:** 2 questions per benchmark, SSD and baseline modes;
- **eagle_report.**

Not tested on the Mac, so verify on the H100:
- the vLLM backend;
- EAGLE's own scripts (they need CUDA);
- CUDA memory of the 7-step training at 2048 tokens (Phase 3's smoke test).
