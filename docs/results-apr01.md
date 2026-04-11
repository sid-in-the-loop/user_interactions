# AlpacaEval Results — Apr 1 2026

AlpacaEval 2.0 on 805 prompts. Judge: `gpt-4o-mini` (patched into `weighted_alpaca_eval_gpt4_turbo` config — model name changed from retired `gpt-4-1106-preview`). Reference model: GPT-4 Turbo (baked into benchmark).

**Note on absolute numbers:** Because the judge is gpt-4o-mini instead of gpt-4-turbo, LC win rates are inflated vs the paper (~37.9% for Qwen3-4B base). Do NOT compare absolute numbers to the paper or to old results in `results-mar26.md` (which also used gpt-4o-mini but a different config). Use these results only for relative comparisons within this table.

---

## Results

| Model | LC Win Rate | Win Rate | Avg Length |
|-------|-------------|----------|------------|
| base_qwen3_4b_nothink | 60.9% | 70.5% | 2653 |
| base_qwen3_4b_think | 56.9% | 64.8% | 2538 |
| sft4b_trained_nothink | 53.6% | 43.0% | 1500 |
| sft4b_trained_think | 53.6% | 43.0% | 1522 |
| base_qwen3_8b_nothink | 69.2% | 78.3% | 2723 |
| base_qwen3_8b_think | **74.4%** | 73.8% | 2279 |
| sft8b_trained_nothink | **70.5%** | 70.9% | 2321 |
| sft8b_trained_think | 67.2% | 66.2% | 2213 |

SFT models: 4B = `sft_wc_thinking_best_bs8_ga32_lr5e6/extended_final_hf`, 8B = `qwen3_8b/sft_wf_thinking_best_lr5e6/final`.

---

## What's confusing about these results

### 1. 4B SFT tanks hard; 8B SFT barely moves

The 4B trained model drops from 60.9% → 53.6% LC win rate and collapses in raw win rate (70.5% → 43.0%). Average length goes from 2653 → 1500 tokens — a huge drop. The LC metric partially compensates for shorter answers, so the raw win rate drop is the real signal.

The 8B trained model barely changes: 69.2% → 70.5% nothink (essentially flat), with length drop from 2723 → 2321. Both metrics hold.

Why the difference? Unknown. Both are SFT on thinking y* data. Possible explanations: (a) 4B is more susceptible to distribution shift from WildChat-style training data and collapses its general response style, (b) the 4B training run overfit or used too high a learning rate, (c) the 8B has enough capacity to maintain general instruction-following while absorbing the new style.

### 2. Thinking vs nonthinking flips direction between 4B and 8B base

- **4B base:** nothink wins (60.9% > 56.9%) — thinking mode hurts
- **8B base:** think wins (74.4% > 69.2%) — thinking mode helps

This is counterintuitive. AlpacaEval is mostly simple instruction-following tasks, not hard reasoning, so you'd expect thinking to either not help or hurt (wasted tokens). Yet the 8B model benefits from it. The most likely explanation is that the 8B's thinking traces are higher quality and lead to better-calibrated final answers, while the 4B's thinking traces are noisier and add length without quality gain.

After SFT, both models prefer nothink (sft8b nothink 70.5% > sft8b think 67.2%), suggesting training on y* — which has think blocks stripped — might have degraded thinking mode quality.

### 3. SFT think ≈ SFT nothink for 4B (53.6% vs 53.6%)

The 4B trained model in both modes gives literally identical LC win rates and near-identical raw win rates. This could mean the model's thinking traces post-SFT are not adding any signal — they either got too short to matter, or the SFT removed the model's ability to leverage reasoning.

### 4. 4B y* targets are very short (~145 words stripped)

The training targets (y* from `ystar_thinking_best_clean.jsonl`) average ~145 words after stripping think blocks. The trained 4B model outputs ~1500 chars on AlpacaEval ≈ ~250 words. So the model partially learned the conciseness of y* but didn't fully collapse. The length drop from base (2653) → trained (1500) is clearly from imitating y* style — but the model is still 1.7x longer than its training targets. Whether the conciseness is good or bad depends on the task; for AlpacaEval it hurts raw win rate.

---

## Bottom line

Use **nothinking mode** for all evaluations as the standard (best or equal in all trained model conditions).

The 4B SFT result on AlpacaEval is bad and its cause is unclear — training did something that hurt general instruction-following badly. The 8B SFT result is good — training didn't hurt general ability. Domain-specific evals (winrate vs y, vs y*) needed to see if the 4B SFT actually learned what it was supposed to learn.

---

*Results from: `alpaca_eval_data/results/*/weighted_alpaca_eval_gpt4_turbo/leaderboard.csv`*
