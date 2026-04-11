# Evaluation Results — Mar 26 2026

WildFeedback SFT models (8B and 4B), Qwen3 base. All results on WildFeedback data.

---

## Alpaca Eval — LC Winrate vs GPT-4 Turbo

Judge: `weighted_alpaca_eval_gpt-4o-mini-2024-07-18`

| Model | LC Winrate | Annotator |
|---|---|---|
| base_qwen3_8b | 25.6% | gpt-4o-mini ✓ |
| 8b_think_best_final | **45.1%** | gpt-4o-mini ✓ |
| 8b_think_best_ext_final | 34.9% | gpt-4o-mini ✓ |
| 8b_think_full_final | 31.4% | gpt-4o-mini ✓ |
| 8b_nothink_best_final | 11.3% | gpt-4o-mini ✓ |
| 8b_nothink_full_final | 6.7% | gpt-4o-mini ✓ |
| base_qwen3_4b | 19.1% | gpt-4o-mini ✓ |

---

## Arena-Hard v2.0 — Win% vs GPT-4.1

Win = model wins BOTH games (A/B and B/A). Baseline: `gpt-4.1`.

| Model | Overall | Creative Writing | Hard Prompt |
|---|---|---|---|
| base_qwen3_8b | 7.6% | 17.6% | 2.6% |
| 8b_think_best_final | 8.7% | 13.6% | 6.2% |
| **8b_think_best_ext_final** | **11.2%** | 17.2% | 8.2% |
| 8b_think_full_final | 9.9% | 14.0% | 7.8% |
| 8b_nothink_best_final | 1.3% | 0.8% | 1.6% |
| 8b_nothink_full_final | 0.7% | 0.0% | 1.0% |

---

## Winrate vs GPT-4 (y* from SFT model vs y)

y* = trained model output, y = original GPT-4 response from WildFeedback.
n=500 fixed sample, position-bias removed. Net = (W−L)/N.

| Model | W | L | T | Net Winrate |
|---|---|---|---|---|
| 8b thinking_best | 109 | 28 | 44 | **+44.8%** |
| 8b thinking_full | 263 | 132 | 105 | +26.2% |
| 8b nonthinking_best | 29 | 100 | 49 | −39.9% |
| 8b nonthinking_full | 72 | 284 | 124 | −44.2% |
| 4b thinking_best | 92 | 34 | 18 | **+40.3%** |
| 4b thinking_full | 198 | 138 | 102 | +13.7% |
| 4b nonthinking_best | 35 | 85 | 59 | −27.9% |
| 4b nonthinking_full | 84 | 272 | 127 | −38.9% |

---

## Ablation Winrates (8B)

| Comparison | W | L | T | Net |
|---|---|---|---|---|
| thinking_best vs thinking_full | 0 | 0 | 181 | **0.0%** (tie) |
| thinking vs nonthinking full | 296 | 60 | 124 | **+49.2%** |
| 8b y* vs 4b y* (thinking full) | 109 | 83 | 246 | +5.9% |
| 8b ybase vs 4b ybase | 86 | 124 | 272 | −7.9% |

---

## Does Hindsight (o) Help at Inference? (y* vs y_base)

y_base = Qwen3-8B output on x alone (no feedback o). Apples-to-apples baseline.

| Comparison | W | L | T | Net |
|---|---|---|---|---|
| ystar 8b nothink vs ybase 8b (nothink) | 126 | 219 | 147 | −18.9% |
| ystar 8b think vs ybase 8b (think) | 43 | 287 | 116 | −54.7% |
| y GPT-4 vs ybase 8b | 40 | 321 | 131 | −57.1% |

---

## Key Takeaways

1. **Thinking mode is essential** — all nonthinking SFT models lose to GPT-4; thinking models win by +26–45%
2. **Best subset vs full disagree across benchmarks** — `think_best` wins alpaca (45.1% LC) but `think_best_ext` wins arena-hard (11.2%)
3. **4B is competitive with 8B** — only +5.9% net margin on thinking full; 4B ybase actually beats 8B ybase by 7.9%
4. **Hindsight is a training signal, not an inference signal** — raw y* with o at inference is *worse* than y_base; the value comes from SFT on y* data, not from seeing o at test time
5. **Alpaca LC winrate is suspiciously high** — 45.1% for an 8B model vs GPT-4 Turbo; worth verifying outputs aren't inflated in length
