# Experiment Updates

Rolling log of experiment status. Most recent on top.


## 2026-04-20 — tac_winrates: mixtures built for wc+wi, training moving to Babel

### Generation (Qwen3-4B teacher, greedy T=0, y_star_0 only)

| dataset      | rows    | status                          | jobid    |
|--------------|---------|---------------------------------|----------|
| wildchat     | 33,918  | COMPLETE (~5h)                  | 2161741  |
| webinstruct  | 49,999  | COMPLETE (same job)             | 2161741  |
| polaris      | TBD     | RUNNING (~15h in, ~33h left)    | 2161722  |

### Judge (wildchat + webinstruct, pos-bias-corrected)

- COMPLETE (jobid 2161742, ~1.5h)
- Outputs: results/eval_wildchat.csv, results/eval_webinstruct.csv
- Raw winrates: wildchat 25.2%, webinstruct 24.0%

### Mixtures (8,500 rows each, capped at wildchat w100 ceiling)

Built for wildchat + webinstruct only; polaris waits on its gen job.

| winrate | wildchat         | webinstruct      |
|--------:|------------------|------------------|
| w100    | 8,500 wins       | 8,500 wins       |
| w50     | 4,250w + 4,250n  | 4,250w + 4,250n  |
| w20     | 1,700w + 6,800n  | 1,700w + 6,800n  |

Total on disk: 462 MB (6 files). Deterministic via seed=0 —
rebuildable any time.

Pending job tac_mixtures (2161743) will rebuild all 3 datasets
once polaris gen finishes. Harmless overlap.

### Training plan (this study)

- Grid: 2 datasets x 3 winrates = 6 jobs
- Dropped w70 in favor of 100/50/20 (wider dynamic range)
- Equal N=8,500 across cells so #gradient-steps is identical
- Moving to Babel L40s (cheaper, less contention); 462 MB mixtures
  rsynced Delta -> Babel using SSH agent forwarding
- 4 methods per job: SFT -> JSD -> DPO -> RKL

### Outstanding

- Polaris gen job 2161722 still running
- Polaris training: 3 more jobs once mixtures land (100/50/20)
- Eval of trained checkpoints: to-do (per-method, across winrates)
- Plotting: winrate (x) vs. downstream eval score (y), one line
  per method; samples-seen is constant so steps/samples axes are
  interchangeable


## Earlier work (pre-this-log)

Prior commits in git history cover:
- Base SFT / FKL / JSD / RKL / DPO / RLAD pipelines (commit 57f8e1e)
- Repo hygiene / large-file cleanup (commit d36aae9)
- Project scaffold (commit c9f9a6c)
