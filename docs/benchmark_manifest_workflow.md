# Filling `benchmark_manifest.csv`

## What runs

| Column | Job | Output |
|--------|-----|--------|
| `arena_hard_prompt`, `arena_creative_writing` | `sbatch jobs/eval/eval_arena_hard.sh <model_id> <path>` | Judgments under `arena-hard-auto/data/arena-hard-v2.0/model_judgment/` |
| `alpaca_eval2_weighted_turbo` | `sbatch jobs/eval/eval_alpaca_eval.sh <model_id> <path>` | `alpaca_eval_data/results/<model_id>/.../leaderboard.csv` (currently **gpt-4o-mini** annotator unless you change the sbatch) |
| `mmlu_pro_5shot_cot` | `sbatch jobs/eval/eval_mmlu_pro.sh <model_id> <path>` | `eval_results/mmlu_pro/<model_id>/summary.json` |
| `ifeval_prompt_loose` | *Not wired* | Run lm-eval-harness / official IFEval separately |

**Important:** `<model_id>` must match the first column in the manifest (e.g. `wc_best_5e6_final`). That string is the filename for Arena answers and judgment, and the folder name for Alpaca/MMLU.

**Sharded FSDP checkpoints** (`epoch-6`, `ext-epoch-6`, …) usually have **no** `config.json` for a full HF model. vLLM will not load them. Merge to a single HF folder first, or point `model_path` at `final` / `extended_final`.

## Submit all jobs (3 benchmarks × N models)

```bash
cd /home/ssmurali/user_interactions
export OPENAI_API_KEY=...   # needed for Arena + Alpaca judging inside the jobs
bash jobs/eval/submit_benchmark_manifest.sh
```

Dry-run (print `sbatch` lines only):

```bash
bash jobs/eval/submit_benchmark_manifest.sh --dry-run
```

## Fill the CSV after results exist

```bash
python scripts/eval/fill_benchmark_manifest.py --write
```

Without `--write`, prints a table only.

Arena scores use the same judge folder that has judgment JSONLs (e.g. `gpt-4o-mini-2024-07-18`). Alpaca uses the first model row’s `win_rate` from `leaderboard.csv`.
