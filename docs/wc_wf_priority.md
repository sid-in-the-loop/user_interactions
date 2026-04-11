# WC / WF SFT priority

## Flagged important (wildchat — already trained / on disk)

| Model | LR | Checkpoint root (`.../offpolicy/fkl/`) | Role |
|-------|-----|------------------------------------------|------|
| WC thinking **best** | 2e-6 | `sft_wc_thinking_best_bs8_ga32_lr2e6/` (`final/` + extend in `sft_wc_think_best_lr2e6_ext6/`) | Primary WC best |
| WC thinking **best** | 5e-6 | `sft_wc_thinking_best_bs8_ga32_lr5e6/` + `sft_wc_think_best_lr5e6_ext6/` | Primary WC best (higher LR) |
| WC thinking **full** | 2e-6 | `sft_wc_thinking_full_bs8_ga32_lr2e6/` + `sft_wc_think_full_lr2e6_ext2/` | Primary WC full |

## Five 6-epoch runs (submit together)

Submit from repo root:

```bash
bash jobs/train/submit_sft_priority_5x6ep.sh
```

| # | Corpus | Config | LR | Output dir (under `fkl/`) |
|---|--------|--------|-----|---------------------------|
| 1 | Wildfeedback | thinking best | 5e-6 | `sft_wf_thinking_best_bs8_ga32_lr5e6_6ep/` |
| 2 | Wildfeedback | thinking best | 2e-6 | `sft_wf_thinking_best_bs8_ga32_lr2e6_6ep/` |
| 3 | Wildfeedback | thinking full | 5e-6 | `sft_wf_thinking_full_bs8_ga32_lr5e6_6ep/` |
| 4 | Wildfeedback | thinking full | 2e-6 | `sft_wf_thinking_full_bs8_ga32_lr2e6_6ep/` |
| 5 | Wildchat | thinking full | 5e-6 | `sft_wc_thinking_full_bs8_ga32_lr5e6_6ep/` |

Each run: **6 epochs**, sharded **`epoch-1` … `epoch-6`**, plus **`final/`** (full HF merge). Step checkpoints are disabled except rare saves (`SAVE_STEPS` large) to save disk.

**Prereqs:** `datasets/wildfeedback/ystar_thinking_best.jsonl` and `ystar_thinking_full.jsonl` must exist.

## WF thinking-best +12 epochs (after 6ep `final/` exists)

Same dirs as `..._lr5e6_6ep` / `..._lr2e6_6ep`: loads each `final/`, trains 12 more epochs, writes sharded **`ext-epoch-6`** and **`ext-epoch-12`**, then overwrites **`final/`**.

```bash
bash jobs/train/submit_sft_wf_best_extend12.sh
```
