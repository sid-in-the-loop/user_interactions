"""Three mechanism diagnostics for the KL-collapse hypothesis on webinstruct.

Runs on one GPU (~20 min total) in a single pass over all LoRA checkpoints:
  #1 KL(π_θ ‖ π_base) drift — avg KL on a shared 40-prompt pool, per ckpt
  #2 Teacher entropy comparison — H(p_T) for webinstruct x+o vs wildchat x+o
  #3 AIME GT-answer logprob — seq logprob of boxed answer per ckpt

Outputs:
  diagnostics/mech/kl_drift.json
  diagnostics/mech/teacher_entropy.json
  diagnostics/mech/aime_gt_logprob.json
  plots/mech_diagnostics/*.pdf
"""
import json, re, pathlib, time, gc
import torch, numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE        = "Qwen/Qwen3-4B"
CKPT_ROOT   = pathlib.Path("/work/hdd/bgtw/ssredharan/checkpoints/tac_winrates")
OUT         = pathlib.Path("/u/ssredharan/user_interactions/diagnostics/mech"); OUT.mkdir(parents=True, exist_ok=True)
PLOTS       = pathlib.Path("/u/ssredharan/user_interactions/plots/mech_diagnostics"); PLOTS.mkdir(parents=True, exist_ok=True)
N_POOL      = 40   # prompts in shared KL pool (20 wildchat + 20 webinstruct)
N_TOK       = 48   # last N positions used for KL
N_ENT       = 100  # prompts per dataset for teacher entropy
MAX_CTX     = 1024 # cap input context

device = "cuda"
print("Loading base Qwen3-4B (bf16)...")
base_model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                                  trust_remote_code=True).to(device).eval()
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

def tokenize_chat_user(text):
    msgs = [{"role": "user", "content": text}]
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                  return_tensors="pt", chat_template_kwargs={"enable_thinking": False})
    return ids[:, -MAX_CTX:].to(device)

# ───────────────────────────────────────────────────────────────── shared pool ──
print("Loading shared prompt pool...")
shared = []
for ds in ["wildchat", "webinstruct"]:
    path = f"/u/ssredharan/user_interactions/experiments/tac_winrates/data/training_inputs/mix_{ds}_teacher_xo_w100_for_methods.jsonl"
    n_taken = 0
    with open(path) as f:
        for line in f:
            if n_taken >= N_POOL // 2: break
            r = json.loads(line)
            shared.append({"ds": ds, "x": r["x"][0]["content"], "o": r.get("verdict",""), "y_star": r["y_star_prefix30"][:1000]})
            n_taken += 1
print(f"  shared pool: {len(shared)}")

# Cache base logits on shared pool (KL denominator)
print("Caching base logits on shared pool...")
base_cache = []
with torch.no_grad():
    for p in shared:
        ids = tokenize_chat_user(p["x"])
        out = base_model(ids, use_cache=False)
        base_cache.append(out.logits[0, -N_TOK:].float().cpu())
        del out
print(f"  cached {len(base_cache)} base-logit tensors")

# ──────────────────────────────────────────────────────── #2 teacher entropy ──
print("Computing teacher H(p_T | x+o) for webinstruct vs wildchat...")
entropy_hist = {"wildchat": [], "webinstruct": []}
with torch.no_grad():
    for ds in ["wildchat", "webinstruct"]:
        path = f"/u/ssredharan/user_interactions/experiments/tac_winrates/data/training_inputs/mix_{ds}_teacher_xo_w100_for_methods.jsonl"
        n_taken = 0
        with open(path) as f:
            for line in f:
                if n_taken >= N_ENT: break
                r = json.loads(line)
                x = r["x"][0]["content"]
                # Teacher-like context: x + "Feedback on the attempt:" + verdict/y_base as proxy "o"
                # (we don't have raw o easily here, but x alone gets us the full prompt entropy)
                ids = tokenize_chat_user(x)
                logits = base_model(ids, use_cache=False).logits[0, -64:]  # last 64 positions
                probs = torch.softmax(logits.float(), dim=-1)
                ent = -(probs * torch.log(probs.clamp(min=1e-12))).sum(-1)  # nats per position
                entropy_hist[ds].extend(ent.cpu().tolist())
                n_taken += 1
        print(f"  {ds}: {len(entropy_hist[ds])} position entropies sampled")
json.dump({k: v[:5000] for k,v in entropy_hist.items()}, open(OUT/"teacher_entropy.json","w"))

# ───────────────────────────────────────────────────────────── #1 KL drift  ──
def iter_ckpts():
    rows = []
    for run_dir in sorted(CKPT_ROOT.glob("*_n*/")):
        run = run_dir.name
        for method in ["sft","fkl","dpo","rkl"]:
            md = run_dir / method
            if not md.exists(): continue
            for ck in sorted(md.iterdir()):
                if (ck/"adapter_config.json").exists():
                    rows.append((run, method, ck.name, ck))
    return rows

ckpts = iter_ckpts()
print(f"\nFound {len(ckpts)} LoRA checkpoints")

def step_key(name):
    if name in ("final","final_model"): return 10**9
    m = re.search(r"(\d+)", name); return int(m.group(1)) if m else 0

# ──────────────────────────────────────────────────── load GT AIME answers ──
print("Loading AIME GT answers for #3...")
aime = []
for pth in ["data/benchmark_data/math/aime25.jsonl","data/benchmark_data/math/aime26.jsonl"]:
    with open(f"/u/ssredharan/user_interactions/{pth}") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                aime.append({"problem": r["problem"], "answer": str(r["answer"])})
print(f"  {len(aime)} AIME problems")

# For each AIME problem, we want log pi_theta(boxed_answer | x). For "answer" being
# just an integer 0-999, we tokenize " \\boxed{N}" appended to the prompt end,
# and sum log-probs of those tokens conditional on the prompt.
def build_aime_inputs():
    items = []
    for p in aime:
        msgs = [{"role": "user", "content": p["problem"]}]
        prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                              chat_template_kwargs={"enable_thinking": False})
        ans_text = f"\\boxed{{{p['answer']}}}"
        full_text = prompt_text + ans_text
        enc_prompt = tok(prompt_text, return_tensors="pt", add_special_tokens=False)
        enc_full   = tok(full_text,   return_tensors="pt", add_special_tokens=False)
        prompt_len = enc_prompt["input_ids"].shape[1]
        full_ids   = enc_full["input_ids"][:, -MAX_CTX:].to(device)
        # answer token positions: from prompt_len to end
        items.append({"ids": full_ids, "ans_start": max(0, prompt_len - (enc_full["input_ids"].shape[1] - full_ids.shape[1]))})
    return items

aime_inputs = build_aime_inputs()

# ────────────────────────────────────────────────── main per-ckpt loop ──
def kl_and_aime_for_ckpt(adapter_path):
    pm = PeftModel.from_pretrained(base_model, str(adapter_path))
    pm.eval()
    kl_total, kl_tokens = 0.0, 0
    with torch.no_grad():
        # KL on shared pool
        for p, base_logits in zip(shared, base_cache):
            ids = tokenize_chat_user(p["x"])
            out = pm(ids, use_cache=False)
            pt_logits = out.logits[0, -N_TOK:].float()
            pt_lp = torch.log_softmax(pt_logits, dim=-1)
            pt_p  = torch.softmax(pt_logits, dim=-1)
            pb_lp = torch.log_softmax(base_logits.to(device).float(), dim=-1)
            kl = (pt_p * (pt_lp - pb_lp)).sum(-1).mean().item()
            kl_total += kl * pt_logits.shape[0]
            kl_tokens += pt_logits.shape[0]
            del out
        # AIME GT logprob (sum over answer tokens, averaged across problems)
        aime_logprobs = []
        for item in aime_inputs:
            ids = item["ids"]
            out = pm(ids, use_cache=False)
            # Predict token t from logits at position t-1
            logits = out.logits[0, item["ans_start"]-1 : -1].float()
            targets = ids[0, item["ans_start"]:]
            lp = torch.log_softmax(logits, dim=-1)
            tok_lp = lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            aime_logprobs.append(tok_lp.sum().item())
            del out
    # cleanup adapter
    try: pm.unload()
    except Exception: pass
    del pm; torch.cuda.empty_cache(); gc.collect()
    return kl_total/max(1,kl_tokens), float(np.mean(aime_logprobs))

print("\nIterating LoRA ckpts for KL drift + AIME GT-logprob...")
results = []
t0 = time.time()
for i, (run, method, name, path) in enumerate(ckpts):
    try:
        kl, aime_lp = kl_and_aime_for_ckpt(path)
        results.append({"run": run, "method": method, "ckpt": name,
                        "kl_vs_base": kl, "aime_gt_logprob": aime_lp,
                        "step": step_key(name)})
    except Exception as e:
        print(f"  [{i+1}] FAIL {run}/{method}/{name}: {e}")
        continue
    if (i+1) % 5 == 0:
        dt = time.time()-t0
        eta = dt/(i+1)*(len(ckpts)-i-1)
        print(f"  [{i+1:3d}/{len(ckpts)}]  {run}/{method}/{name}  KL={kl:.3f}  aime_lp={aime_lp:.2f}  "
              f"elapsed={dt/60:.1f}m eta={eta/60:.1f}m")

json.dump(results, open(OUT/"kl_drift.json","w"), indent=2)
json.dump(results, open(OUT/"aime_gt_logprob.json","w"), indent=2)
print(f"\nSaved {len(results)} ckpt rows to {OUT}/")

# ─────────────────────────────────────────────────────── quick plots ──
print("Plotting...")
METHODS = ["sft","dpo","fkl","rkl"]
COLORS  = {"sft":"#888","dpo":"#2ca02c","fkl":"#1f77b4","rkl":"#d62728"}

# KL drift by method (final only)
fig, ax = plt.subplots(figsize=(10,5))
finals = [r for r in results if r["ckpt"] in ("final","final_model")]
for m in METHODS:
    vals = [r["kl_vs_base"] for r in finals if r["method"]==m]
    labels = [r["run"] for r in finals if r["method"]==m]
    xs = np.arange(len(vals))
    ax.bar(xs + METHODS.index(m)*0.2, vals, 0.2, label=m.upper(), color=COLORS[m])
ax.set_xticks(np.arange(6)+0.3)
ax.set_xticklabels(sorted(set(r["run"] for r in finals)), rotation=20, fontsize=8)
ax.set_ylabel("KL(π_θ ‖ π_base) [nats/token, final ckpt]")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(PLOTS/"kl_final_bars.pdf"); plt.close()

# Entropy histograms
fig, ax = plt.subplots(figsize=(8,5))
for ds, col in [("wildchat","#08306b"), ("webinstruct","#7f2704")]:
    vals = entropy_hist[ds]
    ax.hist(vals, bins=50, alpha=0.55, color=col, label=f"{ds} (n={len(vals)})", density=True)
ax.set_xlabel("per-token entropy (nats)")
ax.set_ylabel("density")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(PLOTS/"teacher_entropy.pdf"); plt.close()

# AIME GT-logprob trajectory per method, avg across datasets
fig, axes = plt.subplots(1, 4, figsize=(18,4), sharey=True)
for j, m in enumerate(METHODS):
    ax = axes[j]
    per_run = {}
    for r in results:
        if r["method"] != m: continue
        per_run.setdefault(r["run"], []).append(r)
    for run, rows in per_run.items():
        rows = sorted(rows, key=lambda r: r["step"])
        ys = [r["aime_gt_logprob"] for r in rows]
        xs = np.linspace(0.2, 1.0, len(ys)) if ys else []
        color = "#08306b" if "wildchat" in run else "#7f2704"
        ax.plot(xs, ys, "o-", color=color, alpha=0.6, linewidth=1.2, markersize=4)
    ax.set_title(m.upper()); ax.set_xlabel("training progress")
    if j==0: ax.set_ylabel("Σ log π_θ(boxed_answer | x)")
    ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(PLOTS/"aime_gt_logprob.pdf"); plt.close()

print(f"\nAll done. Results under {OUT} and plots under {PLOTS}")
