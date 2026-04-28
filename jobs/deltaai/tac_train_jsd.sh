#!/bin/bash
# Re-run JSD only with matched-N subsampling and LR=5e-6 (up from 2e-6).
# Overwrites $CKPT_ROOT/jsd/.
#
# Usage:
#   sbatch jobs/deltaai/tac_train_jsd.sh DATASET WINRATE MATCH_N

#SBATCH --account=bgtw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=tac_train_jsd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=96g
#SBATCH --time=06:00:00
#SBATCH --output=logs/tac_train_%x_%j.out
#SBATCH --error=logs/tac_train_%x_%j.err

set -euo pipefail

DATASET="${1:?Usage: sbatch tac_train_jsd.sh DATASET WINRATE MATCH_N}"
WINRATE="${2:?Usage: sbatch tac_train_jsd.sh DATASET WINRATE MATCH_N}"
MATCH_N="${3:?Usage: sbatch tac_train_jsd.sh DATASET WINRATE MATCH_N}"

case "$DATASET" in wildchat|webinstruct|polaris) ;; *) echo "bad dataset: $DATASET" >&2; exit 1 ;; esac
case "$WINRATE" in 100|70|50|20) ;; *) echo "bad winrate: $WINRATE" >&2; exit 1 ;; esac

module load python/miniforge3_pytorch/2.10.0
conda activate skywork

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"
export HF_HOME=/work/hdd/bgtw/ssredharan/models
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

TRAIN_DIR="experiments/tac_winrates/data/training_inputs"
SRC_METHODS="$TRAIN_DIR/mix_${DATASET}_teacher_xo_w${WINRATE}_for_methods.jsonl"
[ -f "$SRC_METHODS" ] || { echo "ERROR: missing $SRC_METHODS" >&2; exit 1; }

MATCHED_METHODS="$TRAIN_DIR/mix_${DATASET}_teacher_xo_w${WINRATE}_matched${MATCH_N}_for_methods.jsonl"
python - <<PY
import json, random, sys
src = "$SRC_METHODS"
dst = "$MATCHED_METHODS"
n   = int("$MATCH_N")
with open(src) as f:
    rows = [ln for ln in f if ln.strip()]
if len(rows) < n:
    sys.exit(f"ERROR: src has {len(rows)} rows < requested {n}")
if len(rows) == n:
    import shutil; shutil.copyfile(src, dst)
    print(f"matched-N: src already has {n}, copied to {dst}")
else:
    rng = random.Random(hash(("$DATASET","$WINRATE",n)) & 0xffffffff)
    rng.shuffle(rows)
    with open(dst, "w") as f:
        f.writelines(rows[:n])
    print(f"matched-N: sampled {n} of {len(rows)} -> {dst}")
PY

CKPT_ROOT="/work/hdd/bgtw/ssredharan/checkpoints/tac_winrates/${DATASET}_w${WINRATE}"
mkdir -p "$CKPT_ROOT"
BASE_MODEL="Qwen/Qwen3-4B"
TAG="${DATASET}_w${WINRATE}_n${MATCH_N}"

echo "=============== tac_train_jsd: $TAG ==============="
echo "input (matched): $MATCHED_METHODS"
echo "ckpt root      : $CKPT_ROOT/jsd"
echo "lr             : 5e-6  (was 2e-6)"
echo "==================================================="

echo "=== [$(date +%H:%M)] JSD (lr=5e-6, n=$MATCH_N) ==="
python scripts/fkl/train_methods.py \
    --objective    jsd \
    --beta         0.5 \
    --pure_kl \
    --input        "$MATCHED_METHODS" \
    --y_star_field y_star_prefix30 \
    --model        "$BASE_MODEL" \
    --epochs       2 \
    --batch_size   2 \
    --grad_accum   32 \
    --lr           5e-6 \
    --max_length   2048 \
    --num_ckpts    4 \
    --save_every   0 \
    --lora_r       16 \
    --lora_alpha   32 \
    --lora_dropout 0.05 \
    --lora_target  all-linear \
    --output_dir   "$CKPT_ROOT/jsd" \
    --run_name     "${TAG}_jsd_lr1e4" \
    --wandb_project tac-winrates

echo "=== [$(date +%H:%M)] DONE: $TAG jsd ==="
du -sh "$CKPT_ROOT/jsd" 2>/dev/null || true
