#!/bin/bash
# Submit all 6 LoRA ablation experiments for Qwen3-8B:
#   2 datasets (prefix30, full) × 3 objectives (sft, fkl, jsd)
# Model: Qwen/Qwen3-8B (single GPU each)
#
# Usage:
#   bash jobs/train/submit_lora_ablation_qwen3_8b.sh
#   bash jobs/train/submit_lora_ablation_qwen3_8b.sh --dry-run

set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

MODEL="Qwen/Qwen3-8B"
INPUT="datasets/wildfeedback/ystar_prefix_qwen3_8b.jsonl"
CKPT_BASE="checkpoints/lora_ablation_qwen3_8b"

submit() {
    local y_star_field="$1"
    local objective="$2"
    local tag="${y_star_field/y_star_/}_${objective}"   # e.g. prefix30_sft
    local out_dir="${CKPT_BASE}/${tag}"
    local run_name="qwen3_8b_lora_${tag}"

    local cmd=(sbatch
        --job-name="lora_8b_${tag}"
        --partition=general
        --gres=gpu:1
        --mem=64G
        --cpus-per-task=16
        --time=24:00:00
        --output="logs/%j_lora_8b_${tag}.out"
        --error="logs/%j_lora_8b_${tag}.err"
        --chdir="$REPO"
        --wrap="
            source ~/miniconda3/etc/profile.d/conda.sh
            conda activate opf
            export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}
            export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
            export PYTHONPATH=$REPO:\${PYTHONPATH:-}
            export PYTHONUNBUFFERED=1
            mkdir -p logs $out_dir

            PURE_KL=""
            [[ "$objective" != "sft" ]] && PURE_KL="--pure_kl"

            python scripts/fkl/train_lora.py \
                --input               $INPUT \
                --y_star_field        $y_star_field \
                --objective           $objective \
                --model               $MODEL \
                --output_dir          $out_dir \
                --run_name            $run_name \
                --lora_r              64 \
                --lora_alpha          128 \
                --batch_size          2 \
                --grad_accum          64 \
                --epochs              3 \
                --lr                  2e-6 \
                --max_length          2048 \
                --teacher_max_length  4096 \
                --num_ckpts           8 \
                --wandb_project       lora-ablation-qwen3-8b \
                $PURE_KL
        "
    )

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY RUN] lora_8b_${tag} → $out_dir"
    else
        job_id=$("${cmd[@]}")
        echo "Submitted [${tag}] → job $job_id → $out_dir"
    fi
}

echo "════════════════════════════════════════════════════"
echo "  Submitting 6 LoRA ablation jobs (Qwen3-8B)"
echo "  Model : $MODEL"
echo "  Input : $INPUT"
echo "════════════════════════════════════════════════════"

for y_star_field in y_star_prefix30 y_star_full; do
    for objective in sft fkl jsd; do
        submit "$y_star_field" "$objective"
    done
done

echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs   : logs/*_lora_8b_*.out"
echo "Ckpts  : $CKPT_BASE/"
