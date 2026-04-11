#!/bin/bash
# Unified win rate evaluation submission — wraps winrate_eval.py
#
# All arguments are forwarded verbatim to scripts/eval/winrate_eval.py.
# Set OPENAI_API_KEY or GOOGLE_API_KEY before submitting.
#
# Usage:
#   sbatch jobs/eval/submit_winrate.sh --input <file> [opts...]
#   sbatch jobs/eval/submit_winrate.sh --file-a <fa> --file-b <fb> [opts...]
#   sbatch jobs/eval/submit_winrate.sh --input <file> \
#       --comparisons prefix30:y_star_prefix30:y:y noprefix:y_star_noprefix:y:y
#
# Common options forwarded to winrate_eval.py:
#   --input <file>            Single-file mode (y vs y_star [vs y_base])
#   --file-a / --file-b       Pair mode: compare two separate JSONL files
#   --field-a / --field-b     Field names for pair mode (default: y_star)
#   --label-a / --label-b     Display labels for pair mode
#   --comparisons <specs...>  Multi-comparison mode: label_a:field_a:label_b:field_b
#   --output-dir <dir>        Output directory (default: data/winrate_results)
#   --judge gemini|gpt4omini  Judge model (default: gpt4omini)
#   --subsample <n>           Number of examples to sample (default: 2000)
#   --seed <n>                Random seed (default: 42)
#   --max-concurrent <n>      Max parallel API calls (default: 1000)
#
# Examples:
#   OPENAI_API_KEY=... sbatch jobs/eval/submit_winrate.sh \
#       --input datasets/wildfeedback/ystar_prefix_qwen3_8b.jsonl \
#       --output-dir data/winrate_results/qwen3_8b \
#       --subsample 500
#
#   GOOGLE_API_KEY=... sbatch jobs/eval/submit_winrate.sh \
#       --input datasets/wildfeedback/ystar_prefix_qwen3_8b.jsonl \
#       --output-dir data/winrate_results/qwen3_8b_gemini \
#       --judge gemini
#
#   OPENAI_API_KEY=... sbatch jobs/eval/submit_winrate.sh \
#       --input datasets/wildfeedback/ystar_prefix_qwen3_8b.jsonl \
#       --output-dir data/winrate_results/prefix_ablation \
#       --comparisons prefix30:y_star_prefix30:y:y \
#                     noprefix:y_star_noprefix:y:y \
#                     full:y_star_full:y:y \
#                     prefix30:y_star_prefix30:noprefix:y_star_noprefix \
#                     prefix30:y_star_prefix30:full:y_star_full \
#                     noprefix:y_star_noprefix:full:y_star_full
#
#SBATCH --partition=general
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00

set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "Usage: sbatch jobs/eval/submit_winrate.sh --input <file> [opts...]"
    echo "   or: sbatch jobs/eval/submit_winrate.sh --file-a <fa> --file-b <fb> [opts...]"
    echo "All opts are forwarded to scripts/eval/winrate_eval.py"
    exit 1
fi

# ── Environment ───────────────────────────────────────────────────────────────
source ~/miniconda3/etc/profile.d/conda.sh && conda activate opf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
mkdir -p logs
cd /home/ssmurali/user_interactions
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export PYTHONUNBUFFERED=1

# Preflight: require at least one API key
if [[ -z "${OPENAI_API_KEY:-}" && -z "${GOOGLE_API_KEY:-}" ]]; then
    echo "ERROR: set OPENAI_API_KEY or GOOGLE_API_KEY before submitting" >&2
    exit 1
fi

echo "════════════════════════════════════════"
echo " Win rate eval"
echo " Args: $*"
echo "════════════════════════════════════════"

python scripts/eval/winrate_eval.py "$@"
