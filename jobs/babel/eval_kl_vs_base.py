"""Compute KL(π_ckpt ∥ π_base) on a fixed prompt sample for all ckpts of one run.

Approach:
  1. Load base model once (Qwen3-4B).
  2. Load 200 fixed prompts (alpaca + math mix).
  3. For each ckpt:
     - Generate from base on prompts (greedy, max_new=256). Tokens = x.
     - Compute log p_base(x | prompt) and log p_ckpt(x | prompt) on the SAME tokens.
     - KL ≈ mean( log p_ckpt - log p_base ) over generated tokens.
     (This is sample-based KL on policy π_ckpt's actions seen by base — ok for a
     diagnostic measure, not strict KL. Reverse-KL similar.)

Output: per-ckpt JSON with kl_ckpt_to_base (and reverse).

Usage:
  python eval_kl_vs_base.py \
    --run_ckpt_dir /data/.../WC-1_sft_teacher_wins_cond_xo \
    --base_model Qwen/Qwen3-4B \
    --out_path .../WC-1/kl_vs_base.json
"""
import argparse, json, os, sys
from pathlib import Path

# strip user_interactions from sys.path so we don't shadow wandb
for p in list(sys.path):
    if "user_interactions" in p: sys.path.remove(p)

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def load_prompts(n_alpaca=150, n_math=50):
    """Fixed prompt sample. Mix of alpaca + math-500 prompts."""
    from datasets import load_dataset
    out = []
    # alpaca
    ds = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", trust_remote_code=True)["eval"]
    for i in range(min(n_alpaca, len(ds))):
        out.append({"src": "alpaca", "idx": i, "instruction": ds[i]["instruction"]})
    # math-500
    math_path = "/home/ssmurali/user_interactions/CFT-Eric-Zhu/evaluation_script/evaluate_math/data/math-500/test.jsonl"
    if os.path.exists(math_path):
        rows = [json.loads(l) for l in open(math_path)][:n_math]
        for i, r in enumerate(rows):
            q = r.get("problem") or r.get("question") or r.get("instruction") or ""
            out.append({"src": "math", "idx": i, "instruction": q})
    return out


def build_chat(tokenizer, instruction):
    """Apply Qwen3 chat template. enable_thinking=False to avoid <think> wrapper."""
    msgs = [{"role": "user", "content": instruction}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)


@torch.no_grad()
def generate_continuation(model, tokenizer, prompt, max_new=256, device="cuda"):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500).to(device)
    out = model.generate(
        **inputs, max_new_tokens=max_new, do_sample=False,
        temperature=1.0, top_p=1.0, top_k=0, pad_token_id=tokenizer.eos_token_id,
    )
    full = out[0]
    cont = full[inputs.input_ids.shape[1]:]
    return inputs.input_ids[0], cont  # both 1d tensors on device


@torch.no_grad()
def logprob_of_continuation(model, prompt_ids, cont_ids, device="cuda"):
    """log p(cont | prompt). Returns sum of token logprobs."""
    full = torch.cat([prompt_ids, cont_ids], dim=0).unsqueeze(0).to(device)
    logits = model(full).logits[0]                        # [T, V]
    # next-token prediction: position t predicts token t+1
    target = full[0, 1:]                                  # [T-1]
    logp = F.log_softmax(logits[:-1], dim=-1)             # [T-1, V]
    tok_logp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [T-1]
    # only keep continuation positions
    cont_start = prompt_ids.shape[0] - 1
    return tok_logp[cont_start:cont_start + cont_ids.shape[0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_ckpt_dir", required=True, help="path to dir with step-*/ + final/")
    ap.add_argument("--base_model", default="Qwen/Qwen3-4B")
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--n_prompts", type=int, default=80)  # 80 prompts × 256 tokens × ckpts
    ap.add_argument("--max_new", type=int, default=256)
    args = ap.parse_args()

    device = "cuda"
    print(f"[init] loading base {args.base_model} …")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(args.base_model,
                                                torch_dtype=torch.bfloat16,
                                                trust_remote_code=True).to(device).eval()

    # Discover ckpts
    ckpts = sorted([d for d in os.listdir(args.run_ckpt_dir)
                    if d.startswith("step-")],
                   key=lambda d: int(d.split("-")[1]))
    if os.path.isdir(os.path.join(args.run_ckpt_dir, "final")):
        ckpts.append("final")
    print(f"[init] found ckpts: {ckpts}")

    prompts_data = load_prompts(n_alpaca=int(args.n_prompts*0.7),
                                n_math=int(args.n_prompts*0.3))
    print(f"[init] loaded {len(prompts_data)} prompts")

    # Step 1 — generate continuations from BASE on the prompt set (fixed reference)
    print("[step 1] generating continuations from base …")
    prompt_records = []
    for p in prompts_data:
        text = build_chat(tokenizer, p["instruction"])
        pid, cid = generate_continuation(base, tokenizer, text, max_new=args.max_new, device=device)
        if cid.shape[0] < 5: continue                         # skip too-short
        prompt_records.append({"src": p["src"], "idx": p["idx"], "prompt_ids": pid.cpu(),
                               "cont_base": cid.cpu()})
    print(f"  → {len(prompt_records)} continuations")

    # Step 2 — log p_base on those continuations (one pass, baseline reference)
    print("[step 2] log p_base on base continuations …")
    base_logp_per_prompt = []
    for r in prompt_records:
        lp = logprob_of_continuation(base,
                                     r["prompt_ids"].to(device),
                                     r["cont_base"].to(device),
                                     device=device)
        base_logp_per_prompt.append(lp.cpu().sum().item() / max(1, lp.shape[0]))

    # Step 3 — for each ckpt, swap in LoRA, compute log p_ckpt(cont_base|prompt) and the gen-from-ckpt KL
    results = {}
    base_with_lora = base  # we'll attach LoRA via PeftModel.from_pretrained
    print("[step 3] iterating over ckpts …")
    for ck in ckpts:
        ck_path = os.path.join(args.run_ckpt_dir, ck)
        print(f"  ckpt: {ck}")
        # Load LoRA on top of base. PeftModel attaches adapters in-place; we re-load each iteration.
        if hasattr(base_with_lora, "unload"):
            try: base_with_lora = base_with_lora.unload()
            except: pass
        try:
            ck_model = PeftModel.from_pretrained(base, ck_path).to(device).eval()
        except Exception as e:
            print(f"    ! failed to load {ck}: {e}")
            continue

        # KL(π_ckpt || π_base) ≈ E_{x ~ π_base} [log π_ckpt(x) - log π_base(x)]
        # (Here we're using base's continuations as the reference distribution.)
        diffs = []
        for i, r in enumerate(prompt_records):
            lp_ckpt = logprob_of_continuation(ck_model,
                                              r["prompt_ids"].to(device),
                                              r["cont_base"].to(device), device=device)
            mean_diff = (lp_ckpt.cpu().sum().item() / max(1, lp_ckpt.shape[0])) - base_logp_per_prompt[i]
            diffs.append(mean_diff)
        kl_ckpt_minus_base = -sum(diffs) / max(1, len(diffs))   # > 0 = ckpt is *worse* on base text → KL increased
        results[ck] = {"kl_vs_base": float(kl_ckpt_minus_base), "n": len(diffs)}
        print(f"    KL ≈ {kl_ckpt_minus_base:+.4f} (n={len(diffs)})")

        # cleanup
        del ck_model
        torch.cuda.empty_cache()

    Path(os.path.dirname(args.out_path)).mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump({"run_ckpt_dir": args.run_ckpt_dir,
                   "base_model": args.base_model,
                   "n_prompts": len(prompt_records),
                   "results": results}, f, indent=2)
    print(f"[done] wrote {args.out_path}")


if __name__ == "__main__":
    main()
