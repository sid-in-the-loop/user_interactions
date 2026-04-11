#!/bin/bash
# Submit all 6 wildchat training jobs, each chained with 2 eval jobs:
#   - GPT-4o-mini: all checkpoints  → eval_results/alpaca/
#   - Gemini 2.5 Flash: 4 ckpts     → eval_results/alpaca_gemini/
#
# Usage:
#   OPENAI_API_KEY=... GOOGLE_API_KEY=... bash jobs/train/submit_wc_all.sh

set -e
cd /home/ssmurali/user_interactions

if [ -z "$OPENAI_API_KEY" ]; then echo "ERROR: OPENAI_API_KEY not set" && exit 1; fi
if [ -z "$GOOGLE_API_KEY" ]; then echo "ERROR: GOOGLE_API_KEY not set" && exit 1; fi

TRAIN_JOBS=(
    "jobs/train/sbatch_sft_p30_wc.sh    checkpoints/qwen3_8b_sft_p30_wc"
    "jobs/train/sbatch_sft_full_wc.sh   checkpoints/qwen3_8b_sft_full_wc"
    "jobs/train/sbatch_fkl_p30_wc.sh    checkpoints/qwen3_8b_fkl_p30_wc"
    "jobs/train/sbatch_fkl_full_wc.sh   checkpoints/qwen3_8b_fkl_full_wc"
    "jobs/train/sbatch_jsd_p30_wc.sh    checkpoints/qwen3_8b_jsd_p30_wc"
    "jobs/train/sbatch_jsd_full_wc.sh   checkpoints/qwen3_8b_jsd_full_wc"
)

for entry in "${TRAIN_JOBS[@]}"; do
    TRAIN_SH=$(echo $entry | awk '{print $1}')
    CKPT_DIR=$(echo $entry | awk '{print $2}')
    RUN=$(basename $CKPT_DIR)

    TRAIN_JID=$(sbatch --parsable --export=ALL "$TRAIN_SH")
    echo "Submitted train $RUN → job $TRAIN_JID"

    GPT_JID=$(sbatch --parsable --export=ALL \
        --job-name="eval_gpt4omini_${RUN}" \
        --dependency=afterok:$TRAIN_JID \
        jobs/eval/eval_ckpts_gpt4omini.sh "$CKPT_DIR")
    echo "  └─ eval gpt4omini → job $GPT_JID (after $TRAIN_JID)"

    GEM_JID=$(sbatch --parsable --export=ALL \
        --job-name="eval_gemini_${RUN}" \
        --dependency=afterok:$TRAIN_JID \
        jobs/eval/eval_ckpts_gemini.sh "$CKPT_DIR")
    echo "  └─ eval gemini    → job $GEM_JID (after $TRAIN_JID)"

    echo ""
done

echo "All jobs submitted."
