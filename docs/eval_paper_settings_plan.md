# Benchmark plan (paper-aligned settings)

Paper settings to match:

| Benchmark | Setting |
|-----------|---------|
| **AlpacaEval 2.0** | Default; completions judged with **Weighted AlpacaEval GPT-4 Turbo** |
| **Arena-Hard v2** | Default; judged with **GPT-4.1** |
| **IFEval** | **Prompt-level loose** |
| **MMLU-Pro** | **CoT, 5-shot** (validation split for demos) |

---

## 1. Gap vs current repo (must fix before “exact” runs)

| Piece | Current repo | Paper |
|-------|----------------|-------|
| AlpacaEval | `eval_alpaca_eval.sh` uses `--annotators_config gpt-4o-mini` | **Weighted GPT-4 Turbo** annotator (AlpacaEval package name, e.g. weighted turbo config — set after checking `alpaca_eval --help` / installed configs) |
| Arena-Hard | `eval_arena_hard.sh` writes `judge_model: gpt-4o-mini` | **gpt-4.1** (match `show_result.py` default; ensure `gen_judgment` endpoint list includes it) |
| MMLU-Pro | `mmlu_pro_eval.py` `--ntrain 5` | Already **5-shot CoT** ✓ |
| IFEval | Not integrated | Add **lm-evaluation-harness** or official IFEval script; metric = **prompt-level loose** |

**Action:** Update `jobs/eval/eval_alpaca_eval.sh` and `jobs/eval/eval_arena_hard.sh` (or paper-specific copies) once annotator string and Arena judge id are confirmed against your `alpaca_eval` install and OpenAI account access to GPT-4.1 / GPT-4 Turbo.

---

## 2. Model list (all under `FKL=/data/group_data/cx_group/ssmurali/offpolicy/fkl`)

**Base**

| `model_id` | Path |
|------------|------|
| `base_qwen3_4b` | `Qwen/Qwen3-4B` |

**Wildchat — thinking best**

| `model_id` | Path |
|------------|------|
| `wc_best_5e6_final` | `$FKL/sft_wc_thinking_best_bs8_ga32_lr5e6/final` |
| `wc_best_5e6_extended` | `$FKL/sft_wc_thinking_best_bs8_ga32_lr5e6/extended_final` |
| `wc_best_2e6_final` | `$FKL/sft_wc_thinking_best_bs8_ga32_lr2e6/final` |
| `wc_best_2e6_extended` | `$FKL/sft_wc_thinking_best_bs8_ga32_lr2e6/extended_final` |

**Wildchat — thinking full**

| `model_id` | Path |
|------------|------|
| `wc_full_2e6_final` | `$FKL/sft_wc_thinking_full_bs8_ga32_lr2e6/final` |
| `wc_full_2e6_extended` | `$FKL/sft_wc_thinking_full_bs8_ga32_lr2e6/extended_final` |
| `wc_full_5e6_ep1` … `ep4` | `$FKL/sft_wc_thinking_full_bs8_ga32_lr5e6_6ep/epoch-{1,2,3,4}` |

**Wildfeedback — thinking best (6ep + extension checkpoints)**  
*Logical epoch labels: `epoch-6` = end of first 6ep; your `ext-epoch-6` → report as **epoch 12**; `ext-epoch-12` → **epoch 18**.*

| `model_id` | On-disk path |
|------------|----------------|
| `wf_best_2e6_ep6` | `.../sft_wf_thinking_best_bs8_ga32_lr2e6_6ep/epoch-6` |
| `wf_best_2e6_ep12` | `.../sft_wf_thinking_best_bs8_ga32_lr2e6_6ep/ext-epoch-6` |
| `wf_best_2e6_ep18` | `.../sft_wf_thinking_best_bs8_ga32_lr2e6_6ep/ext-epoch-12` |
| `wf_best_5e6_ep6` | `.../sft_wf_thinking_best_bs8_ga32_lr5e6_6ep/epoch-6` |
| `wf_best_5e6_ep12` | `.../..._lr5e6_6ep/ext-epoch-6` |
| `wf_best_5e6_ep18` | `.../..._lr5e6_6ep/ext-epoch-12` |

**Wildfeedback — thinking full lr2e6** (same epoch naming)

| `model_id` | Path |
|------------|------|
| `wf_full_2e6_ep6` | `.../sft_wf_thinking_full_bs8_ga32_lr2e6_6ep/epoch-6` |
| `wf_full_2e6_ep12` | `.../.../ext-epoch-6` |
| `wf_full_2e6_ep18` | `.../.../ext-epoch-12` |

**Note:** Your list repeated **wf best lr2e6** for items 5 and 8. Item **7** is **wf full lr2e6**. If you intended **wf full lr5e6** as an 8th line, add the same ep6/ep12/ep18 pattern for `sft_wf_thinking_full_bs8_ga32_lr5e6_6ep` when that run exists.

**Sharded dirs** (`epoch-*`, `ext-epoch-*`): vLLM / HF load must use the **sharded FSDP layout** (rank_*.pt). If load fails, merge to a single `final`-style dir once per checkpoint before eval.

---

## 3. What to run per benchmark

### Arena-Hard v2 (two headline numbers)

1. **Generate** (GPU, vLLM): one job per `model_id` — `gen_answer_vllm.py` (full bench).
2. **Judge** (API): `gen_judgment.py` with **GPT-4.1**.
3. **Score**:
   - **Hard prompt:** `show_result.py --category hard_prompt --judge-names gpt-4.1`
   - **Creative writing:** `show_result.py --category creative_writing --judge-names gpt-4.1`

Store: win rate or category score as in Arena-Hard docs.

### AlpacaEval 2.0

1. Generate (GPU): `alpaca_eval_gen.py` → `model_outputs.json`.
2. Annotate (API): `alpaca_eval` with **weighted GPT-4 Turbo** config.
3. Read **leaderboard.csv** (LC win rate + length-controlled etc. per AlpacaEval 2 defaults).

### MMLU-Pro

- One GPU job per model:  
  `python scripts/eval/mmlu_pro_eval.py --model_path <path> --output_dir eval_results/mmlu_pro/<model_id> --ntrain 5`  
- Aggregate **overall accuracy** from `summary.json`.

### IFEval

- Run official or **lm-eval** task with **prompt-level loose**; one row per model. (Integrate script + sbatch in a follow-up.)

---

## 4. Effective execution plan (minimize cost / wall time)

**Phase A — Generation (GPU-bound)**  
- Queue **Arena** + **Alpaca** generations in parallel (same vLLM stack), cap concurrent jobs by cluster quota.  
- **MMLU-Pro** jobs are long; schedule after or in parallel with spare GPUs.

**Phase B — API judging (cost-bound)**  
- Arena + Alpaca judges: **serialize or limit concurrency** (rate limits).  
- Only run judges after **all** generations for a model exist.

**Phase C — Aggregate**  
- Script reads: Arena `show_result` outputs, Alpaca `leaderboard.csv`, MMLU-Pro `summary.json`, IFEval metrics → **one master CSV**.

**Dependency:** Each `model_id` row needs: sharded→merged loader OK, disk visible on compute node.

---

## 5. Master CSV schema (one row per `model_id`)

Suggested columns:

| Column | Content |
|--------|---------|
| `model_id` | Short id (see tables above) |
| `model_path` | Full path or HF id |
| `arena_hard_prompt` | Score (GPT-4.1) |
| `arena_creative_writing` | Score (GPT-4.1) |
| `alpaca_lc_win_rate` | (or columns AlpacaEval 2 exports for weighted turbo) |
| `mmlu_pro_accuracy` | Overall CoT 5-shot |
| `ifeval_prompt_loose` | Prompt-level loose |
| `notes` | e.g. sharded / merged |

Fill `NA` until that eval finishes.

---

## 6. Files in repo

- **Manifest (editable):** `eval_results/benchmark_manifest.csv` — lists every `model_id` + `model_path` for batch submit.
- **This plan:** `docs/eval_paper_settings_plan.md`

Next implementation steps (when you approve):  
(1) Paper-specific job scripts or env flags for Alpaca + Arena judges.  
(2) `scripts/eval/aggregate_benchmark_csv.py` to merge outputs into the master CSV.  
(3) IFEval wiring + one sbatch template.
