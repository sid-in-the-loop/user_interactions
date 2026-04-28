"""Teacher-vs-base distribution analysis (no LoRA, fast).

For 500 random (x, o, y*) tuples per dataset from the w100 mixture files:
  a. Response length distribution of y*
  b. Per-position entropy profile of y* under teacher context H(π_teacher(·|x+o+y*[:i]))
  c. KL(π_teacher ‖ π_base) averaged per response (teacher-forced on y*)
  d. Fraction of positions with entropy > threshold (0.5, 1.0, 2.0 nats)
  e. Tail-concentration of KL mass — at positions where KL is high, how much of it
     falls on vocabulary tokens that have low probability under π_base? (Hypothesis:
     teacher drift is concentrated in tokens that base considered improbable.)

Outputs:
  diagnostics/teacher_dist/summary.json
  plots/teacher_dist/{lengths,entropy_profile,kl_per_resp,high_entropy_frac,tail_concentration}.pdf
"""
import json, pathlib, time, random
import numpy as np, torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE      = "Qwen/Qwen3-4B"
OUT       = pathlib.Path("/u/ssredharan/user_interactions/diagnostics/teacher_dist"); OUT.mkdir(parents=True, exist_ok=True)
PLOTS     = pathlib.Path("/u/ssredharan/user_interactions/plots/teacher_dist"); PLOTS.mkdir(parents=True, exist_ok=True)
MIXDIR    = "/u/ssredharan/user_interactions/experiments/tac_winrates/data/mixtures"
N_PER_DS  = 500
MAX_CTX   = 2048
SEED      = 42
YSTAR_TMPL = (
    "{x}\n\nHere is a partial attempt at the solution:\n\n\n"
    "Feedback on the attempt:\n{o}\n\nNow provide a complete response."
)

device = "cuda"
print(f"Loading {BASE} (bf16)...")
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                              trust_remote_code=True).to(device).eval()
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

def load_samples(ds, n):
    path = f"{MIXDIR}/mix_{ds}_teacher_xo_w100.jsonl"
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("y_star") and r.get("source_o") is not None:
                    rows.append({"x": r["x"], "o": r["source_o"], "y_star": r["y_star"]})
    rng = random.Random(SEED); rng.shuffle(rows)
    return rows[:n]

samples = {}
for ds in ["wildchat", "webinstruct"]:
    samples[ds] = load_samples(ds, N_PER_DS)
    print(f"  {ds}: {len(samples[ds])} samples")

def build_ids(msgs_text, y_star):
    """Apply chat template with one user turn + y_star as assistant turn."""
    msgs = [
        {"role": "user", "content": msgs_text},
        {"role": "assistant", "content": y_star},
    ]
    full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False,
                                   chat_template_kwargs={"enable_thinking": False})
    user_only = tok.apply_chat_template([msgs[0]], tokenize=False, add_generation_prompt=True,
                                         chat_template_kwargs={"enable_thinking": False})
    ids = tok(full, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    prompt_len = tok(user_only, add_special_tokens=False, return_tensors="pt")["input_ids"].shape[1]
    return ids, prompt_len

def process_sample(x, o, y_star):
    """Return per-position entropy (teacher), KL(teacher‖base), and base-prob-of-teacher-argmax stats."""
    # Teacher context uses the exact y* generation template x+o
    teacher_msg = YSTAR_TMPL.format(x=x, o=o)
    base_msg    = x

    ids_t, plen_t = build_ids(teacher_msg, y_star)
    ids_b, plen_b = build_ids(base_msg,    y_star)

    # If either gets truncated too long, cap from the left and drop y* portion accordingly
    if ids_t.shape[0] > MAX_CTX or ids_b.shape[0] > MAX_CTX:
        return None  # skip — too long

    with torch.no_grad():
        ids_t_dev = ids_t.unsqueeze(0).to(device)
        ids_b_dev = ids_b.unsqueeze(0).to(device)
        # Logits at position i predict token i+1
        t_logits = model(ids_t_dev, use_cache=False).logits[0]  # [T, V]
        b_logits = model(ids_b_dev, use_cache=False).logits[0]  # [T, V]

    # y* spans [plen, len] — positions [plen-1, len-1] in the logits produce those tokens
    ystar_len = ids_t.shape[0] - plen_t
    if ystar_len < 5: return None  # too short to be meaningful

    # Align the y* region (same tokens in both, just different prefixes)
    t_ystar_logits = t_logits[plen_t-1 : plen_t-1+ystar_len].float()  # [L, V]
    b_ystar_logits = b_logits[plen_b-1 : plen_b-1+ystar_len].float()

    p_t = torch.softmax(t_ystar_logits, dim=-1)
    p_b = torch.softmax(b_ystar_logits, dim=-1)
    log_p_t = torch.log(p_t.clamp(min=1e-12))
    log_p_b = torch.log(p_b.clamp(min=1e-12))

    # (b) per-position entropy under teacher
    ent = -(p_t * log_p_t).sum(-1)  # [L]
    # (c) KL(teacher || base) per position
    kl  = (p_t * (log_p_t - log_p_b)).sum(-1)  # [L]
    # (e) tail concentration: at top-1 of teacher, what is base prob?
    top1 = p_t.argmax(-1)
    base_prob_of_teacher_top1 = p_b.gather(-1, top1.unsqueeze(-1)).squeeze(-1)  # [L]

    return {
        "len": ystar_len,
        "entropy_mean": ent.mean().item(),
        "entropy_positions": ent.cpu().tolist()[:256],  # cap for storage
        "kl_per_response": kl.mean().item(),
        "kl_sum": kl.sum().item(),
        "base_prob_of_teacher_top1": base_prob_of_teacher_top1.cpu().tolist()[:256],
    }

# ───────────────────────────────────────────────────────── main loop ──
print("\nProcessing samples (each: 2 forward passes)...")
results = {"wildchat": [], "webinstruct": []}
t0 = time.time()
n_done = 0
for ds in ["wildchat", "webinstruct"]:
    for i, s in enumerate(samples[ds]):
        r = process_sample(s["x"], s["o"], s["y_star"])
        if r is not None: results[ds].append(r)
        n_done += 1
        if n_done % 100 == 0:
            dt = time.time()-t0
            print(f"  [{n_done}/{2*N_PER_DS}]  elapsed={dt/60:.1f}m  "
                  f"wc_kept={len(results['wildchat'])} web_kept={len(results['webinstruct'])}")
print(f"  done in {(time.time()-t0)/60:.1f}m")

# ───────────────────────────────────────────────────────── aggregate ──
summary = {}
for ds in ["wildchat", "webinstruct"]:
    R = results[ds]
    if not R: continue
    lens = [r["len"] for r in R]
    ents = [r["entropy_mean"] for r in R]
    kls  = [r["kl_per_response"] for r in R]
    # (d) high-entropy fractions per response
    high_entropy_fracs = {thr: [] for thr in [0.5, 1.0, 2.0]}
    for r in R:
        ep = np.array(r["entropy_positions"])
        for thr in high_entropy_fracs:
            high_entropy_fracs[thr].append(float((ep > thr).mean()))
    # (e) base probability at teacher-top1, avg over high-KL positions
    tail = []
    for r in R:
        bp = r["base_prob_of_teacher_top1"]
        if bp: tail.append(float(np.mean(bp)))

    summary[ds] = {
        "n": len(R),
        "length_mean": float(np.mean(lens)),
        "length_median": float(np.median(lens)),
        "entropy_mean": float(np.mean(ents)),
        "kl_mean": float(np.mean(kls)),
        "kl_sum_mean": float(np.mean([r["kl_sum"] for r in R])),
        "high_entropy_frac": {str(k): float(np.mean(v)) for k,v in high_entropy_fracs.items()},
        "base_prob_of_teacher_top1_mean": float(np.mean(tail)) if tail else None,
    }

json.dump(summary, open(OUT/"summary.json","w"), indent=2)
# Also save raw per-sample values so future plot tweaks don't require recompute.
raw = {ds: {
    "len":                          [r["len"] for r in results[ds]],
    "entropy_mean":                 [r["entropy_mean"] for r in results[ds]],
    "kl_per_response":              [r["kl_per_response"] for r in results[ds]],
    "base_prob_of_teacher_top1":    [x for r in results[ds] for x in r["base_prob_of_teacher_top1"]],
} for ds in results}
json.dump(raw, open(OUT/"raw.json","w"))
print(f"\nSummary:\n{json.dumps(summary, indent=2)}")

# ───────────────────────────────────────────────────────── plots ──
colors = {"wildchat": "#08306b", "webinstruct": "#7f2704"}

# (a) response length
fig, ax = plt.subplots(figsize=(7,4))
for ds in ["wildchat","webinstruct"]:
    ax.hist([r["len"] for r in results[ds]], bins=40, alpha=0.55,
            color=colors[ds], label=f"{ds} (n={len(results[ds])})", density=True)
ax.set_xlabel("y* length (tokens)"); ax.set_ylabel("density")
ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
plt.savefig(PLOTS/"lengths.pdf"); plt.close()

# (b) entropy distributions (mean per response)
fig, ax = plt.subplots(figsize=(7,4))
for ds in ["wildchat","webinstruct"]:
    ax.hist([r["entropy_mean"] for r in results[ds]], bins=40, alpha=0.55,
            color=colors[ds], label=ds, density=True)
ax.set_xlabel("mean per-token teacher entropy (nats)"); ax.set_ylabel("density")
ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
plt.savefig(PLOTS/"entropy_profile.pdf"); plt.close()

# (c) KL per response — x-axis clamped to [0, 2] nats for readability
fig, ax = plt.subplots(figsize=(7,4))
for ds in ["wildchat","webinstruct"]:
    ax.hist([r["kl_per_response"] for r in results[ds]], bins=40, alpha=0.55,
            color=colors[ds], label=ds, density=True, range=(0, 2))
ax.set_xlim(0, 2)
ax.set_xlabel("KL(teacher ‖ base) (nats/token)")
ax.set_ylabel("density"); ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
plt.savefig(PLOTS/"kl_per_resp.pdf"); plt.close()

# (d) high-entropy fraction bars
fig, ax = plt.subplots(figsize=(7,4))
thresholds = [0.5, 1.0, 2.0]
x = np.arange(len(thresholds)); w = 0.35
for i, ds in enumerate(["wildchat","webinstruct"]):
    vals = [summary[ds]["high_entropy_frac"][str(t)] for t in thresholds]
    ax.bar(x + (i-0.5)*w, vals, w, label=ds, color=colors[ds])
ax.set_xticks(x); ax.set_xticklabels([f">{t} nats" for t in thresholds])
ax.set_ylabel("fraction of y* positions"); ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(PLOTS/"high_entropy_frac.pdf"); plt.close()

# (e) tail concentration — base prob of teacher's top-1, lower = teacher drifts to tail
fig, ax = plt.subplots(figsize=(7,4))
for ds in ["wildchat","webinstruct"]:
    all_bp = []
    for r in results[ds]:
        all_bp.extend(r["base_prob_of_teacher_top1"])
    ax.hist(all_bp, bins=50, alpha=0.55, color=colors[ds],
            label=ds, density=True, range=(0,1))
ax.set_xlabel("P_base(arg max teacher) — lower = teacher chose a base-tail token")
ax.set_ylabel("density"); ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
plt.savefig(PLOTS/"tail_concentration.pdf"); plt.close()

print(f"\nAll artifacts under {OUT} and {PLOTS}")
