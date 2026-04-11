#!/bin/bash
# Submit all 6 LoRA ablation experiments:
#   2 datasets (prefix30, full) × 3 objectives (sft, fkl, jsd)
# Model: OLMo-3-7B-Instruct-SFT (single GPU each)
#
# Usage:
#   bash jobs/train/submit_lora_ablation.sh
#   bash jobs/train/submit_lora_ablation.sh --dry-run   # print sbatch commands only

set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

MODEL="allenai/OLMo-3-7B-Instruct-SFT"
INPUT="datasets/wildfeedback/ystar_prefix_olmo_7b_full_wf.jsonl"
CKPT_BASE="checkpoints/lora_ablation"

submit() {
    local y_star_field="$1"
    local objective="$2"
    local tag="${y_star_field/y_star_/}_${objective}"   # e.g. prefix30_sft
    local out_dir="${CKPT_BASE}/${tag}"
    local run_name="olmo_lora_${tag}"

    local cmd=(sbatch
        --job-name="lora_${tag}"
        --partition=general
        --gres=gpu:1
        --mem=64G
        --cpus-per-task=8
        --time=12:00:00
        --output="logs/%j_lora_${tag}.out"
        --error="logs/%j_lora_${tag}.err"
        --chdir="$REPO"
        --wrap="
            source ~/miniconda3/etc/profile.d/conda.sh
            conda activate opf
            export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}
            export PYTHONPATH=$REPO:\${PYTHONPATH:-}
            export PYTHONUNBUFFERED=1
            export OPENAI_API_KEY=${OPENAI_API_KEY:-}
            mkdir -p logs $out_dir

            python scripts/fkl/train_lora.py \
                --input               $INPUT \
                --y_star_field        $y_star_field \
                --objective           $objective \
                --model               $MODEL \
                --output_dir          $out_dir \
                --run_name            $run_name \
                --batch_size          4 \
                --grad_accum          8 \
                --epochs              2 \
                --lr                  2e-4 \
                --lora_r              16 \
                --lora_alpha          32 \
                --topk                50 \
                --max_length          2048 \
                --save_steps          200 \
                --eval_every_n_samples 1000 \
                --wandb_project       prefix-lora-ablation
        "
    )

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY RUN] ${cmd[*]}"
    else
        job_id=$("${cmd[@]}")
        echo "Submitted [$tag] → $job_id"
    fi
}

echo "════════════════════════════════════════════════════"
echo "  Submitting 6 LoRA ablation jobs"
echo "  Model : $MODEL"
echo "  Input : $INPUT"
echo "════════════════════════════════════════════════════"

# 2 datasets × 3 objectives
for y_star_field in y_star_prefix30 y_star_full; do
    for objective in sft fkl jsd; do
        submit "$y_star_field" "$objective"
    done
done

echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs   : logs/*_lora_*.out"
