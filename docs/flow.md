# Pipeline quick-ref: data → y* → SFT → eval

Run everything from **repo root**.

---

## 1. Overview

```
filtered JSONL  →  y* gen  →  SFT (FSDP 2-GPU)  →  merge to HF  →  eval (3 benchmarks)  →  manifest CSV
```

---

## 2. Data

| Dataset | Path prefix | Files |
|---------|-------------|-------|
| WildFeedback | `datasets/wildfeedback/` | `tuples.jsonl`, `filtered_BEST.jsonl`, `filtered_DECENT.jsonl`, `filtered_NOISE.jsonl` |
| WildChat | `datasets/wildchat/` | `filtered_tuples.jsonl`, `filtered_best.jsonl`, `filtered_decent.jsonl`, `filtered_noise.jsonl` |

---

## 3. y* generation

**One template:** `jobs/train/sbatch_gen_ystar.sh`

| Env var | Required | Default | Notes |
|---------|----------|---------|-------|
| `INPUT` | yes | — | Path to filtered JSONL |
| `OUTPUT_DIR` | yes | — | Where to write `ystar_*.jsonl` |
| `MODE` | yes | — | `A` (nonthinking), `B` (thinking), `both` |
| `SUFFIX` | no | `""` | e.g. `_best` → `ystar_thinking_best.jsonl` |
| `GPUS` | no | `1` | Set 4 for large datasets |
| `TIME` | no | `12:00:00` | SBATCH time limit |
| `MODEL` | no | `Qwen/Qwen3-4B` | Base model for vLLM |

**Examples:**

```bash
# WildFeedback thinking, BEST tier
export INPUT=datasets/wildfeedback/filtered_BEST.jsonl OUTPUT_DIR=datasets/wildfeedback MODE=B SUFFIX=_best
sbatch --job-name=ystar_wf_think_best jobs/train/sbatch_gen_ystar.sh

# WildChat nonthinking, full, 4 GPUs
export INPUT=datasets/wildchat/filtered_tuples.jsonl OUTPUT_DIR=datasets/wildchat MODE=A SUFFIX=_full GPUS=4
sbatch --gres=gpu:4 --cpus-per-task=32 --mem=128G --job-name=ystar_wc_nothink_full jobs/train/sbatch_gen_ystar.sh
```

---

## 4. SFT

**One template:** `jobs/train/sbatch_sft_one.sh`

| Env var | Required | Default | Notes |
|---------|----------|---------|-------|
| `INPUT` | yes | — | Path to y* JSONL |
| `RUN_NAME` | yes | — | Dir name under `/data/.../offpolicy/fkl/` |
| `LR` | no | `8e-7` | Learning rate |
| `EPOCHS` | no | `2` | |
| `MODEL` | no | `Qwen/Qwen3-4B` | Set to `path/to/final` to resume/extend |
| `MASTER_PORT` | no | `29500` | Change when running multiple jobs |
| `SAVE_STEPS` | no | `10` | Set `999999` to skip step checkpoints |
| `SAVE_EPOCH_EVERY` | no | `1` | |
| `SHARDED_EPOCH_PREFIX` | no | `""` | e.g. `ext-` for extension runs |

**Examples:**

```bash
# Basic SFT
export INPUT=datasets/wildfeedback/ystar_thinking_best.jsonl RUN_NAME=sft_wf_think_best LR=5e-6
sbatch --job-name=sft_wf_think_best jobs/train/sbatch_sft_one.sh

# Extend from a previous run's final checkpoint for 6 more epochs
export INPUT=datasets/wildfeedback/ystar_thinking_best.jsonl \
       RUN_NAME=sft_wf_think_best_ext6 \
       MODEL=/data/.../sft_wf_think_best/final \
       EPOCHS=6 LR=5e-6 SHARDED_EPOCH_PREFIX=ext-
sbatch --job-name=sft_wf_ext6 jobs/train/sbatch_sft_one.sh
```

**Batch submit helpers** (use `sbatch_sft_one.sh` internally):

| Script | What it does |
|--------|-------------|
| `submit_sft_4wc_4wf.sh` | 4 WC + 4 WF SFT jobs (LR 8e-7) |
| `submit_sft_16_lr.sh` | 8 configs × 2 LRs (5e-6, 2e-6) |
| `submit_sft_priority_5x6ep.sh` | 5 priority runs, 6 epochs each |
| `submit_sft_extend_and_rerun.sh` | 3 extend + 5 re-run jobs |
| `submit_sft_wf_best_extend12.sh` | Extend WF-best by 12 epochs |

Output: `/data/group_data/cx_group/ssmurali/offpolicy/fkl/<RUN_NAME>/final/`

---

## 5. Merge FSDP → HuggingFace

Needed when eval requires HF format (vLLM can't load FSDP shards directly).

```bash
# List dirs to merge in eval_results/merge_fsdp_tasks.txt, then:
sbatch jobs/train/sbatch_merge_fsdp_2gpu.sh
```

Env: `MERGE_TASKS_FILE` (default `eval_results/merge_fsdp_tasks.txt`), `MERGE_BASE_MODEL` (default `Qwen/Qwen3-4B`).

---

## 6. Evaluation

### 6.1 Single model, single benchmark

```bash
sbatch jobs/eval/eval_arena_hard.sh  <model_id> <model_path>
sbatch jobs/eval/eval_alpaca_eval.sh <model_id> <model_path>
sbatch jobs/eval/eval_mmlu_pro.sh    <model_id> <model_path>
```

All need 1 GPU. Arena + Alpaca need `OPENAI_API_KEY` for judging.

MMLU-Pro optional env: `MMLU_PRO_SUBJECTS`, `MMLU_PRO_SMOKE=1`, `MMLU_PRO_TP`.

### 6.2 All models via manifest

```bash
# Add rows to eval_results/benchmark_manifest.csv (model_id, model_path, ...)
bash jobs/eval/submit_benchmark_manifest.sh           # submit all 3 benchmarks per row
bash jobs/eval/submit_benchmark_manifest.sh --dry-run # preview

# After jobs finish, fill in scores:
python scripts/eval/fill_benchmark_manifest.py --write
```

### 6.3 Checkpoint eval (pairwise judging)

```bash
# HelpSteer2 (Claude Haiku judge, default):
RUN_DIR=/path/to/run DATA_PATH=/path/to/validation.jsonl bash jobs/eval/eval_checkpoints.sh

# TL;DR (local Qwen judge):
TASK_TAG=tldr JUDGE_MODEL=Qwen/Qwen3-8B \
SYSTEM_PROMPT="Write summary..." \
RUN_DIR=... DATA_PATH=... bash jobs/eval/eval_checkpoints.sh
```

Env: `CKPTS`, `STYLE`, `BASELINE`, `WORLD_SIZE`, `EVAL_N`, etc.

---

## 7. Outputs & logs

| What | Where |
|------|-------|
| SFT checkpoints | `/data/.../offpolicy/fkl/<RUN_NAME>/final/` |
| Alpaca results | `alpaca_eval_data/results/<model_id>/` |
| Arena results | `arena-hard-auto/data/arena-hard-v2.0/model_answer/<model_id>.jsonl` |
| MMLU-Pro results | `eval_results/mmlu_pro/<model_id>/summary.json` |
| Manifest | `eval_results/benchmark_manifest.csv` |
| Slurm logs | `logs/` |

---

## 8. Other scripts (not the main pipeline)

| Script | Purpose |
|--------|---------|
| `jobs/train/sbatch_fkl_v2_baseline.sh` | FKL baseline (4-GPU, FP32) |
| `jobs/train/sbatch_fkl_fsdp.sh` | Older FKL FSDP training |
| `jobs/eval/eval_probe_signal.sh` | FKL signal probes |
| `jobs/eval/run_probe_signal_all_ckpts.sh` | Batch probe across checkpoints |
| `jobs/eval/run_winrate_wildfeedback.sh` | GPT-4o-mini winrate on y* files |
| `docs/wc_wf_priority.md` | WC/WF priority experiment notes |
