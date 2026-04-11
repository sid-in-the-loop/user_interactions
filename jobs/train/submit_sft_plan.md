# SFT submit plan: extend done runs + re-run quit runs

**QOSMaxGRESPerUser (PD):** Your pending jobs (6607209–6607216) are waiting because you're at the GPU limit for your user. They will start when some of the currently running jobs (6607205, 6607206, 6607207, 6607208) finish. No change needed—just wait.

---

## 1. Extend the 3 completed runs (load from `final`, train more epochs)

These already have a saved `final` checkpoint. We run **more epochs** from that checkpoint and write to a **new** output dir so the original `final` is not overwritten.

| Run | Epochs to run | Load from (MODEL) | New output (RUN_NAME) | INPUT | LR |
|-----|----------------|------------------|------------------------|-------|-----|
| think_full_lr2e6 | **2** more | .../sft_wc_thinking_full_bs8_ga32_lr2e6/final | sft_wc_think_full_lr2e6_ext2 | ystar_thinking_full.jsonl | 2e-6 |
| think_best_lr5e6 | **6** more | .../sft_wc_thinking_best_bs8_ga32_lr5e6/final | sft_wc_think_best_lr5e6_ext6 | ystar_thinking_best.jsonl | 5e-6 |
| think_best_lr2e6 | **6** more | .../sft_wc_thinking_best_bs8_ga32_lr2e6/final | sft_wc_think_best_lr2e6_ext6 | ystar_thinking_best.jsonl | 2e-6 |

**Total: 3 extend jobs.**

---

## 2. Re-run the 5 that quit (4 epochs from base model)

Same data/LR as original, but **4 epochs** so they get more training and are comparable to the extended thinking runs. Use distinct `MASTER_PORT` if you submit many at once.

| Run | INPUT | LR | RUN_NAME (job name) | EPOCHS |
|-----|-------|-----|---------------------|--------|
| think_full_lr5e6 | ystar_thinking_full.jsonl | 5e-6 | sft_wc_think_full_lr5e6 | 4 |
| nothink_full_lr5e6 | ystar_nonthinking_full.jsonl | 5e-6 | sft_wc_nothink_full_lr5e6 | 4 |
| nothink_full_lr2e6 | ystar_nonthinking_full.jsonl | 2e-6 | sft_wc_nothink_full_lr2e6 | 4 |
| nothink_best_lr5e6 | ystar_nonthinking_best.jsonl | 5e-6 | sft_wc_nothink_best_lr5e6 | 4 |
| nothink_best_lr2e6 | ystar_nonthinking_best.jsonl | 2e-6 | sft_wc_nothink_best_lr2e6 | 4 |

**Total: 5 re-run jobs.** Set `EPOCHS=4`; MODEL stays default (base).

---

## 3. Epoch summary

| Config | Already done | This plan | Total epochs from base |
|--------|--------------|-----------|-------------------------|
| think_full_lr2e6 | 2 | +2 extend | 4 |
| think_best_lr5e6 | 2 | +6 extend | 8 |
| think_best_lr2e6 | 2 | +6 extend | 8 |
| think_full_lr5e6 | 0 (quit) | re-run 4 | 4 |
| nothink_full (both LRs) | 0 (quit) | re-run 4 | 4 |
| nothink_best (both LRs) | 0 (quit) | re-run 4 | 4 |

Re-runs use 4 epochs so those configs get comparable training to the extended thinking runs; you can set EPOCHS=6 in the script if you want more.
