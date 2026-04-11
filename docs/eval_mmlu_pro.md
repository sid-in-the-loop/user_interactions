# MMLU-Pro evaluation

Benchmark: [MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro) (NeurIPS 24) — 10-way MCQ, CoT-style prompt (same template as upstream `evaluate_from_local.py`).

**Implementation:** `scripts/eval/mmlu_pro_eval.py` (vLLM, batched). No clone of the MMLU-Pro repo required; dataset is pulled from Hugging Face [`TIGER-Lab/MMLU-Pro`](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro).

## Run (Slurm)

```bash
sbatch jobs/eval/eval_mmlu_pro.sh <run_name> <model_path>
```

Examples:

```bash
sbatch jobs/eval/eval_mmlu_pro.sh base_qwen3 Qwen/Qwen3-4B
sbatch jobs/eval/eval_mmlu_pro.sh wf_best /data/group_data/cx_group/ssmurali/offpolicy/fkl/sft_wf_thinking_best_bs8_ga32_lr5e6_6ep/final
```

**Outputs:** `eval_results/mmlu_pro/<run_name>/`

- `summary.json` / `summary.csv` — per-subject + overall accuracy  
- `<subject>.json` — per-question predictions + raw generations  
- `run_log.txt`

## Optional environment (job script)

| Variable | Effect |
|----------|--------|
| `MMLU_PRO_SUBJECTS` | Default `all`. Comma-separated substrings to filter categories (e.g. `Math,Computer Science`). |
| `MMLU_PRO_SMOKE=1` | Cap at 20 questions per subject (sanity check). |
| `MMLU_PRO_TP` | Tensor parallel size (default `1`). |

## Local (no Slurm)

```bash
conda activate off-policy-feedback
cd /path/to/user_interactions
python scripts/eval/mmlu_pro_eval.py \
  --model_path Qwen/Qwen3-4B \
  --output_dir eval_results/mmlu_pro/debug \
  --max_questions_per_subject 5
```

## Notes

- **Wall time:** Full test split is large (~12k items across subjects); allow many hours on one GPU, or use `MMLU_PRO_SMOKE=1` first.
- **VRAM:** Uses `--max_model_len 8192` and `--max_new_tokens 2048`; reduce `--gen_batch_size` if OOM.
- **Citation:** Wang et al., *MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark*, arXiv:2406.01574.
