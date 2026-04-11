# Baseline comparison: runs 1–6

## 1. Current results (verified)

All six models have **AlpacaEval** (LC winrate) and **Arena-Hard** (hard_prompt) results.

### AlpacaEval 2.0 (length-controlled winrate %)

| Model | LC Winrate |
|-------|------------|
| baseline_sft_fp32 | 23.80 |
| baseline_v1_s500 | 18.03 |
| baseline_v1 | 19.18 |
| baseline_v2_s500 | 13.48 |
| baseline_v2_s1000 | 10.98 |
| baseline_v2 | 9.76 |

### Arena-Hard v2.0 — hard_prompt (%)

| Model | Scores (%) | CI (%) |
|-------|------------|--------|
| baseline_sft_fp32 | 7.2 | ±0.8 |
| baseline_v1_s500 | 3.1 | ±0.5 |
| baseline_v1 | 1.5 | ±0.2 |
| baseline_v2_s500 | 0.7 | ±0.2 |
| baseline_v2_s1000 | 0.6 | ±0.2 |
| baseline_v2 | 0.5 | ±0.2 |

---

## 2. Run grouping and setup

- **Group (1):** baseline_sft_fp32 — 2 GPUs, eff batch 256, ~248 steps, same data (y_star.jsonl).
- **Group (2,3):** baseline_v1_s500, baseline_v1 — 4 GPUs, eff batch 64, ~530 steps/epoch; same dataset. v1_s500 = checkpoint at 500 steps; baseline_v1 = full run (~998 steps).
- **Group (4,5,6):** baseline_v2_s500, baseline_v2_s1000, baseline_v2 — same training setup as v1 family but on **y_star_processed.jsonl** (noisier / filtered).

So overall run-wise, **(1) is comparable to (2,3)** in terms of global batch size, dataset (unfiltered y_star), and objective; the main distinction is **steps** (248 vs 500/998) and **batch size** (256 vs 64).

---

## 3. Performance by group

| Group | Models | AlpacaEval LC | Arena hard_prompt |
|-------|--------|---------------|-------------------|
| (1) | baseline_sft_fp32 | **23.80** | **7.2** |
| (2,3) | baseline_v1_s500, baseline_v1 | 18.03, 19.18 | 3.1, 1.5 |
| (4,5,6) | baseline_v2_s500, s1000, baseline_v2 | 13.48 → 9.76 | 0.7 → 0.5 |

---

## 4. Takeaways

1. **Hyperparameter change and continued training caused a performance dip.**  
   (1) has higher AlpacaEval and Arena-Hard than (2) and (3) despite same dataset and same 2 epochs. The difference is **fewer, larger-batch steps** (248, eff batch 256) vs **more, smaller-batch steps** (500/998, eff batch 64). So the change in batch size / step count (and possibly fp32 vs other) led to (1) outperforming (2,3).

2. **The noisier (processed) dataset was detrimental to FKL.**  
   (4,5,6) use y_star_processed.jsonl; (2,3) use y_star.jsonl. Across AlpacaEval and Arena-Hard (hard_prompt), (4,5,6) are consistently worse than (2,3), and performance drops further along training (s500 → s1000 → final). So the noisier / filtered target data hurt this FKL-style SFT more.

---

*Results from: `alpaca_eval_data/results/*/weighted_alpaca_eval_gpt-4o-mini-2024-07-18/leaderboard.csv` and `arena-hard-auto` `show_result.py --judge-names gpt-4o-mini-2024-07-18 --category hard_prompt`.*
