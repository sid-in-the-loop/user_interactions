#!/bin/bash
# Train 5 methods (SFT, FKL, JSD, DPO, RKL) sequentially on one
# matched-N (dataset, winrate) mixture. One GPU per sbatch.
#
# Usage:
#   sbatch jobs/deltaai/tac_train.sh DATASET WINRATE MATCH_N
# DATASET in {wildchat, webinstruct, polaris}
# WINRATE in {100, 70, 50, 20}
# MATCH_N  = target sample count (subsamples _for_methods & _for_sdpo to N)
#
# LR: 2e-6 for SFT/FKL/DPO/RKL, 1e-4 for JSD (ref-model-based loss needs
# bigger steps under LoRA — at 2e-6 it stalls near ln(2)).

#SBATCH --account=bgtw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=tac_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=96g
#SBATCH --time=24:00:00
#SBATCH --output=logs/tac_train_%x_%j.out
#SBATCH --error=logs/tac_train_%x_%j.err

set -euo pipefail

DATASET="${1:?Usage: sbatch tac_train.sh DATASET WINRATE MATCH_N}"
WINRATE="${2:?Usage: sbatch tac_train.sh DATASET WINRATE MATCH_N}"
MATCH_N="${3:?Usage: sbatch tac_train.sh DATASET WINRATE MATCH_N}"

case "$DATASET" in wildchat|webinstruct|polaris) ;; *) echo "bad dataset: $DATASET" >&2; exit 1 ;; esac
case "$WINRATE" in 100|70|50|20) ;; *) echo "bad winrate: $WINRATE" >&2; exit 1 ;; esac

module load python/miniforge3_pytorch/2.10.0
conda activate skywork

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"
export HF_HOME=/work/hdd/bgtw/ssredharan/models
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

# ──────────────────────────────────────────────────────────────────────────────
TRAIN_DIR="experiments/tac_winrates/data/training_inputs"
SRC_METHODS="$TRAIN_DIR/mix_${DATASET}_teacher_xo_w${WINRATE}_for_methods.jsonl"
SRC_SDPO="$TRAIN_DIR/mix_${DATASET}_teacher_xo_w${WINRATE}_for_sdpo.jsonl"

if [ ! -f "$SRC_METHODS" ] || [ ! -f "$SRC_SDPO" ]; then
  echo "Training input files missing; running build_training_inputs.py first." >&2
  python experiments/tac_winrates/build_training_inputs.py
fi
[ -f "$SRC_METHODS" ] || { echo "ERROR: missing $SRC_METHODS" >&2; exit 1; }
[ -f "$SRC_SDPO"    ] || { echo "ERROR: missing $SRC_SDPO"    >&2; exit 1; }

# matched-N: subsample methods + sdpo files to the SAME row indices so every
# method trains on the same set of examples.
IN_METHODS="$TRAIN_DIR/mix_${DATASET}_teacher_xo_w${WINRATE}_matched${MATCH_N}_for_methods.jsonl"
IN_SDPO="$TRAIN_DIR/mix_${DATASET}_teacher_xo_w${WINRATE}_matched${MATCH_N}_for_sdpo.jsonl"

python - <<PY
import json, random, shutil, sys
sm, ss = "$SRC_METHODS", "$SRC_SDPO"
dm, ds = "$IN_METHODS", "$IN_SDPO"
n = int("$MATCH_N")
with open(sm) as f: rows_m = [ln for ln in f if ln.strip()]
with open(ss) as f: rows_s = [ln for ln in f if ln.strip()]
if len(rows_m) != len(rows_s):
    sys.exit(f"ERROR: methods ({len(rows_m)}) vs sdpo ({len(rows_s)}) row count mismatch")
if len(rows_m) < n:
    sys.exit(f"ERROR: src has {len(rows_m)} rows < requested {n}")
if len(rows_m) == n:
    shutil.copyfile(sm, dm); shutil.copyfile(ss, ds)
    print(f"matched-N: src already at {n}, copied to {dm}, {ds}")
else:
    rng = random.Random(hash(("$DATASET","$WINRATE",n)) & 0xffffffff)
    idx = list(range(len(rows_m))); rng.shuffle(idx); idx = idx[:n]
    with open(dm, "w") as f: f.writelines(rows_m[i] for i in idx)
    with open(ds, "w") as f: f.writelines(rows_s[i] for i in idx)
    print(f"matched-N: sampled {n} of {len(rows_m)} rows -> {dm}, {ds}")
PY

CKPT_ROOT="/work/hdd/bgtw/ssredharan/checkpoints/tac_winrates/${DATASET}_w${WINRATE}_n${MATCH_N}"
mkdir -p "$CKPT_ROOT"

BASE_MODEL="Qwen/Qwen3-4B"

LORA_ARGS=(
  --lora_r 16
  --lora_alpha 32
  --lora_dropout 0.05
  --lora_target all-linear
)

TAG="${DATASET}_w${WINRATE}_n${MATCH_N}"
COMMON_METHODS_ARGS=(
  --input         "$IN_METHODS"
  --y_star_field  y_star_prefix30
  --model         "$BASE_MODEL"
  --epochs        2
  --batch_size    2
  --grad_accum    32
  --max_length    2048
  --num_ckpts     4
  --save_every    0
  "${LORA_ARGS[@]}"
  --wandb_project tac-winrates
)

echo "=============== tac_train: $TAG ==============="
echo "methods input  : $IN_METHODS"
echo "sdpo   input   : $IN_SDPO"
echo "ckpt root      : $CKPT_ROOT"
echo "==============================================="

# ── 1/5 SFT ─────────────────────────────────────────────────────────────────
echo "=== [$(date +%H:%M)] SFT (lr=2e-6) ==="
python scripts/fkl/train_methods.py \
    --objective  sft \
    --lr         2e-6 \
    --output_dir "$CKPT_ROOT/sft" \
    --run_name   "${TAG}_sft" \
    "${COMMON_METHODS_ARGS[@]}"

# ── 2/5 FKL ─────────────────────────────────────────────────────────────────
echo "=== [$(date +%H:%M)] FKL (lr=2e-6) ==="
python scripts/fkl/train_methods.py \
    --objective  fkl \
    --pure_kl \
    --lr         2e-6 \
    --output_dir "$CKPT_ROOT/fkl" \
    --run_name   "${TAG}_fkl" \
    "${COMMON_METHODS_ARGS[@]}"

# ── 3/5 JSD ─────────────────────────────────────────────────────────────────
echo "=== [$(date +%H:%M)] JSD (lr=1e-4) ==="
python scripts/fkl/train_methods.py \
    --objective  jsd \
    --beta       0.5 \
    --pure_kl \
    --lr         1e-4 \
    --output_dir "$CKPT_ROOT/jsd" \
    --run_name   "${TAG}_jsd_lr1e4" \
    "${COMMON_METHODS_ARGS[@]}"

# ── 4/5 DPO ─────────────────────────────────────────────────────────────────
echo "=== [$(date +%H:%M)] DPO (lr=2e-6) ==="
python scripts/fkl/train_methods.py \
    --objective  dpo \
    --dpo_beta   0.1 \
    --lr         2e-6 \
    --output_dir "$CKPT_ROOT/dpo" \
    --run_name   "${TAG}_dpo" \
    "${COMMON_METHODS_ARGS[@]}"

# ── 5/5 RKL (offline SDPO) with LoRA ────────────────────────────────────────
echo "=== [$(date +%H:%M)] RKL (offline SDPO, LoRA, lr=2e-6) ==="
python experiments/tac_winrates/train_sdpo_lora.py \
    --base_model    "$BASE_MODEL" \
    --train_jsonl   "$IN_SDPO" \
    --output_dir    "$CKPT_ROOT/rkl" \
    --num_epochs    2 \
    --batch_size    4 \
    --grad_accum    8 \
    --learning_rate 2e-6 \
    --num_ckpts     4 \
    "${LORA_ARGS[@]}" \
    --wandb_project tac-winrates \
    --run_name      "${TAG}_rkl"

echo "=== [$(date +%H:%M)] DONE: $TAG ==="
du -sh "$CKPT_ROOT"/*/ 2>/dev/null || true
