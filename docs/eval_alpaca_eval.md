# AlpacaEval 2.0 — How to Run

Benchmark: 805 general instruction-following prompts, pairwise vs GPT-4 Turbo reference answers.  
Judge: `gpt-4o-mini` (patched into `weighted_alpaca_eval_gpt4_turbo` config — see Setup below).  
Key metric: **LC Win Rate** (length-controlled, penalises longer outputs).

---

## Setup (one-time)

The default `weighted_alpaca_eval_gpt4_turbo` config hardcodes `gpt-4-1106-preview` which is retired. Patch it:

```bash
sed -i 's/gpt-4-1106-preview/gpt-4o-mini/g' \
    /data/user_data/ssmurali/miniconda3/envs/opf/lib/python3.11/site-packages/alpaca_eval/evaluators_configs/weighted_alpaca_eval_gpt4_turbo/configs.yaml
```

Verify:
```bash
grep model_name /data/user_data/ssmurali/miniconda3/envs/opf/lib/python3.11/site-packages/alpaca_eval/evaluators_configs/weighted_alpaca_eval_gpt4_turbo/configs.yaml
# → model_name: "gpt-4o-mini"
```

Needs `OPENAI_API_KEY` in environment.

---

## Step 1 — Generate model outputs

Use `scripts/eval/alpaca_eval_gen.py`. Run on a compute node (needs GPU).

```bash
cd /home/ssmurali/user_interactions

python scripts/eval/alpaca_eval_gen.py \
    --model_name <name_for_results> \
    --model_path <path_to_model> \
    --output_file alpaca_eval_data/results/<name>/model_outputs.json \
    --mode nonthinking   # or: thinking
```

Key args:

| Arg | Default | Notes |
|-----|---------|-------|
| `--mode` | `nonthinking` | `nonthinking`: `enable_thinking=False`. `thinking`: enables reasoning but strips `<think>` blocks before saving. |
| `--max_new_tokens` | 2048 | AlpacaEval standard |
| `--temperature` | 0.7 | AlpacaEval standard |
| `--max_model_len` | 16384 | Enough headroom for thinking mode |
| `--tensor_parallel_size` | 1 | Increase for larger models |

Model path must be a local directory (env sets `TRANSFORMERS_OFFLINE=1`). Use full path, e.g.:
```
~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/<hash>/
```

The script skips generation if `output_file` already exists.

---

## Step 2 — Run the judge

```bash
alpaca_eval \
    --model_outputs alpaca_eval_data/results/<name>/model_outputs.json \
    --annotators_config weighted_alpaca_eval_gpt4_turbo \
    --output_path alpaca_eval_data/results/<name>
```

Outputs written to `alpaca_eval_data/results/<name>/weighted_alpaca_eval_gpt4_turbo/`:
- `leaderboard.csv` — full leaderboard with your model inserted
- `annotations.json` — per-example judge decisions

---

## Running multiple models

Generate all outputs first, then loop the judge:

```bash
for model in model_a model_b model_c; do
    alpaca_eval \
        --model_outputs alpaca_eval_data/results/$model/model_outputs.json \
        --annotators_config weighted_alpaca_eval_gpt4_turbo \
        --output_path alpaca_eval_data/results/$model
done
```

The judge caches annotations in the package dir (`annotations_seed0_configs.json`), so re-running on already-judged examples is fast.

---

## Reading results

```bash
grep "<model_name>" alpaca_eval_data/results/<name>/weighted_alpaca_eval_gpt4_turbo/leaderboard.csv
```

Columns that matter:
- `length_controlled_winrate` — primary metric
- `win_rate` — raw win rate (biased toward longer outputs)
- `avg_length` — sanity check; if this is very low, the model collapsed

---

## Caveats

- **Judge is gpt-4o-mini, not gpt-4-turbo** — absolute LC win rates are inflated vs the paper. Qwen3-4B base should be ~37.9% per the paper; we get ~61%. Only use for **relative comparisons within this setup**.
- **Nonthinking mode is standard** for trained models — think and nothink give identical LC win rates post-SFT (for 4B), and nothink wins for 8B. Base 8B is an exception where thinking actually helps (74.4% vs 69.2%).
- **Length collapse is a red flag** — if avg_length drops below ~1500 after SFT, LC win rate may look okay but raw win rate will tank. Check both.

---

## Quick reference — known model paths

| Model | Path |
|-------|------|
| Qwen3-4B base | `~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/<hash>/` |
| Qwen3-8B base | `~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/<hash>/` |
| SFT 4B trained | `/data/group_data/cx_group/ssmurali/offpolicy/fkl/sft_wc_thinking_best_bs8_ga32_lr5e6/extended_final_hf` |
| SFT 8B trained | `/data/group_data/cx_group/ssmurali/offpolicy/fkl/qwen3_8b/sft_wf_thinking_best_lr5e6/final` |
