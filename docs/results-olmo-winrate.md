# OLMo-3-7B-Instruct-SFT Win Rate — WildFeedback (Apr 1 2026)

Model: `allenai/OLMo-3-7B-Instruct-SFT`  
Dataset: WildFeedback full (`tuples.jsonl`), 477 samples (500 sampled, 477 with both y_base and y*)  
Judge: GPT-4o-mini, position-bias removed (two flipped calls, win only if wins both)  
Metric: wins / (wins + losses), ties excluded. 95% Wilson CI.

---

## Results

| Comparison | Win Rate | 95% CI | Wins | Losses | Ties |
|---|---|---|---|---|---|
| OLMo base vs GPT-4 | 32.6% | [27.9, 37.6] | 115 | 238 | 124 |
| OLMo+hindsight vs GPT-4 | 11.8% | [8.9, 15.6] | 43 | 320 | 114 |
| OLMo+hindsight vs OLMo base | 20.4% | [16.5, 24.9] | 73 | 285 | 119 |

Raw outputs: `data/winrate_results/olmo_wf_500/`

---

## Interpretation

**OLMo base loses to GPT-4 ~2:1 (32.6%).** Expected — `y` was written by GPT-4, a much stronger model.

**OLMo+hindsight is worse than OLMo base (20.4% win rate against base).** Giving OLMo the full hindsight context (x + GPT-4 answer + user feedback) *hurts* output quality — it loses to its own baseline ~3:1. Win rate against GPT-4 also collapses from 32.6% → 11.8%.

**OLMo doesn't benefit from the hindsight prompt.** Unlike Qwen3, OLMo-SFT appears to get confused by the long context containing another model's answer and user critique, producing shorter or lower-quality responses rather than an improved rewrite.

**Implication for y* generation:** OLMo is not a viable teacher for the hindsight distillation pipeline on WildFeedback. The y* responses it produces are worse than its own y_base, making them poor training targets. Stick to Qwen3 as teacher.

---

## Prompt Ablation (Apr 2 2026)

The original teacher prompt gave OLMo no instruction on what to do with `y` and `o` — it just dumped them in context. Two revised prompts were tested on all 477 samples:

**Prompt A** — minimal: *"Use the follow-up as a hint about what was missing. Re-answer the original request only. Do not respond to the follow-up directly."*

**Prompt B** — explicit: *"Your task is ONLY to re-answer the original user request — better than the previous response, informed by what the follow-up reveals was missing. Do not address the follow-up directly. Do not acknowledge the previous response."*

| | Original | Prompt A | Prompt B |
|---|---|---|---|
| y* vs y_base win rate | 20.4% [16.5, 24.9] | 23.4% [19.4, 27.9] | 23.5% [19.5, 28.0] |
| y* vs GPT-4 win rate | 11.8% [8.9, 15.6] | 11.3% [8.4, 14.9] | 13.7% [10.6, 17.6] |
| Prompt A vs Prompt B | — | — | 111–112, −0.2% (coin flip) |

Raw outputs: `data/winrate_results/olmo_prompt_ablation/`

Better prompts give a marginal lift (~+3pp on y* vs y_base) but CIs overlap and y* still loses to y_base ~3:1. A and B are indistinguishable. **Prompt framing is not the bottleneck — OLMo fundamentally cannot leverage hindsight context.**

---

## (x, o)-only Ablation — xo template (Apr 2 2026)

Hypothesis: OLMo gets confused by seeing another model's answer (`y`) in context. Tested a prompt family where `y` is never shown — the feedback `o` is injected inline into the last user turn of `x` using the paper Table 1 template.

**Template (xo):** `[hindsight context] The following is a future user message.\nUse this to guide your answer to the user prompt: {o}`

Tested on the fixed 500-sample subset (`winrate_500_ids.json`).

| Comparison | Win Rate | 95% CI | Wins | Losses | Ties |
|---|---|---|---|---|---|
| y*_xo vs y_base | 38.8% | — | — | — | — |
| y*_xo vs y*_old (x,y,o) | — | — | — | — | — |

Raw outputs: `data/winrate_results/olmo_xo_ablation/`

**Removing `y` from context substantially helps.** The xo template (~38.8% vs y_base) is a large improvement over the (x,y,o) prompt (~20%). OLMo can use a future feedback note when it doesn't have to reconcile a competing answer.

---

## Teacher Prompt Ablation (xo-family, 4 variants) — Apr 2 2026

Having established the xo approach works, tested 4 instruction phrasings around the `o` injection on 500 samples. All share the same (x,o)-only context structure; only the injection text differs.

| Label | Injection text |
|---|---|
| **A** (paper xo) | `[hindsight context] The following is a future user message.\nUse this to guide your answer to the user prompt: {o}` |
| **B** (explicit) | `A user will follow up with: {o}\nGiven this follow-up, provide a better answer to the original request. Focus on what the follow-up reveals was missing or wrong. Do not address the follow-up directly.` |
| **C** (minimal) | `Note: {o}` |
| **D** (Socratic) | `The user responded with: {o}\nBefore answering, consider what this reveals about what was missing from a good response. Then provide an improved answer to the original request only.` |

Judge: GPT-4o-mini, position-bias removed. ~480 overlapping samples per comparison.

### vs y_base

| Prompt | Win Rate | 95% CI | Wins | Losses | Ties |
|---|---|---|---|---|---|
| A (paper xo) | 37.7% | [32.6, 43.1] | 123 | 203 | 147 |
| B (explicit) | 30.3% | [25.7, 35.4] | 104 | 239 | 140 |
| **C (minimal)** | **41.3%** | **[36.0, 46.8]** | **131** | **186** | **162** |
| D (Socratic) | 35.3% | [30.5, 40.4] | 125 | 229 | 129 |

### vs y*_A (head-to-head)

| Prompt | Win Rate vs A | 95% CI | Wins | Losses | Ties |
|---|---|---|---|---|---|
| B | 41.3% | [36.0, 46.8] | 131 | 186 | 163 |
| **C** | **59.0%** | **[53.3, 64.4]** | **174** | **121** | **183** |
| D | 48.0% | [42.7, 53.4] | 157 | 170 | 155 |

### Avg response length

| Prompt | Words |
|---|---|
| A | 212.2 |
| B | 202.6 |
| C | 221.8 |
| D | 206.2 |

Raw outputs: `data/winrate_results/olmo_teacher_ablation/`  
Chart: `data/winrate_results/olmo_teacher_ablation/teacher_ablation.png`

**Winner: Prompt C** — `"Note: {o}"` — beats A by 59.0% head-to-head (CI [53.3, 64.4], clear margin) and has the highest vs-y_base rate (41.3%). Less instruction is better: verbose prompts (B, D) constrain OLMo's generation and hurt quality.

**However, all prompts still lose to y_base (all < 50%).** This is not a framing problem — it is a capability ceiling. The xo ablation recovered ~20pp over the original (x,y,o) approach (20% → 41%), but could not cross 50% under any prompt variant tested.

This matters because the winrate comparison measures *response quality*, not training signal quality. For hindsight distillation to work, the assumption is y* > y_base given (x, o): the model should be able to produce a *better* response when it sees the feedback. OLMo violates this assumption — its unconditional response π_student(·|x) is judged better than its hindsight-conditioned response most of the time. SFT on y* would therefore fit the model to *worse* responses than it already produces, providing no useful training signal.

**OLMo is not a viable teacher model.** Use Qwen3 (or another model that can actually leverage hindsight context) for y* generation. If OLMo must be used, Prompt C (`xo_C`) is the least-bad option.
