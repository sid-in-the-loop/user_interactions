"""WildChat judging pipeline (Prompt 3).

Two judges, three conditioning variants per row, both orderings each.

Judge 1 — student (Qwen3-4B, instruct-tuned):
  Offline vLLM. Single llm.generate() call with greedy decoding (T=0,
  max_tokens=4). Prompt is chat-templated with enable_thinking=False; user
  message contains the flattened conversation (x + o) and the two candidate
  responses labelled [Response A] / [Response B]. Free-form output, parsed by
  first letter.

Judge 2 — gpt4o_mini:
  OpenAI API, parallel ThreadPool. Same prompt/protocol.

For each (example, conditioning variant):
  AB ordering: A=y_star, B=y      → y_star wins iff verdict=='A'
  BA ordering: A=y,      B=y_star → y_star wins iff verdict=='B'
  winner = y_star iff y_star wins both orderings
  winner = y      iff y wins      both orderings
  else tie

Outputs:
  data/judgments.jsonl  — one row per example, all verdicts + agreement flags.
  data/judgments_summary.txt — winrates + Wilson 95% CIs + agreement rates.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from common.stats import wilson


CONDS = ("cond_xo", "cond_xyo", "cond_xyo_ystart")


# ─── Prompt construction ─────────────────────────────────────────────────────

JUDGE_USER_TEMPLATE = """\
You are an impartial judge evaluating which of two assistant responses better continues a conversation.

[Conversation]
{conv}
[/Conversation]

[Response A]
{a}
[/Response A]

[Response B]
{b}
[/Response B]

Which response is a better next assistant turn given the conversation? Consider helpfulness, correctness, and how well it addresses the user's most recent message.

Output exactly one letter: A or B."""


def flatten_conv(x_list, o_dict):
    """x_list (list of message dicts) + o_dict (single message dict) → flat string."""
    msgs = list(x_list) + [o_dict]
    parts = []
    for m in msgs:
        role = (m.get("role") or "user").upper()
        content = m.get("content") or ""
        parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


def build_judge_user_text(x_list, o_dict, a_text, b_text):
    return JUDGE_USER_TEMPLATE.format(
        conv=flatten_conv(x_list, o_dict),
        a=a_text,
        b=b_text,
    )


def get_y_text(y_field):
    """y is stored as a dict ({'role':'assistant','content':...}); return content."""
    if isinstance(y_field, dict):
        return y_field.get("content", "") or ""
    return str(y_field) if y_field else ""


def build_jobs(gens):
    """For every (eid, cond, order), yield (key, x_list, o_dict, a_text, b_text)."""
    for d in gens:
        eid = d["example_id"]
        x_list = d["x"]
        o_dict = d["o"]
        y_text = get_y_text(d["y"])
        if not y_text:
            continue
        for cond in CONDS:
            ystar = d.get(f"y_star_{cond}", "")
            if not ystar:
                continue
            yield ((eid, cond, "AB"), x_list, o_dict, ystar, y_text)
            yield ((eid, cond, "BA"), x_list, o_dict, y_text, ystar)


# ─── Resolve both-orderings rule ─────────────────────────────────────────────

def resolve(verdict_pair):
    """{'AB': 'A'|'B'|None, 'BA': ...} → 'y_star' | 'y' | 'tie'."""
    ab = verdict_pair.get("AB")
    ba = verdict_pair.get("BA")
    if ab == "A" and ba == "B":
        return "y_star"
    if ab == "B" and ba == "A":
        return "y"
    return "tie"


# ─── Student judge (Qwen3-4B offline vLLM) ───────────────────────────────────

def parse_letter(text):
    s = (text or "").strip().upper()
    if s.startswith("A"):
        return "A"
    if s.startswith("B"):
        return "B"
    return "tie"


def _try_build_judge(tok, x_list, o_dict, a, b):
    """Build chat-templated prompt and return (prompt_str, n_tokens)."""
    user_text = build_judge_user_text(x_list, o_dict, a, b)
    msgs = [{"role": "user", "content": user_text}]
    prompt = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    n = len(tok.encode(prompt, add_special_tokens=False))
    return prompt, n


def _fit_judge_prompt(tok, x_list, o_dict, a, b, max_input):
    """Robust multi-pass truncation. Returns (prompt_str, ok)."""
    # Pass 0: as-is
    prompt, n = _try_build_judge(tok, x_list, o_dict, a, b)
    if n <= max_input:
        return prompt, True

    # Pass 1: clip a and b proportionally
    over = n - max_input
    a_ids = tok.encode(a or "", add_special_tokens=False)
    b_ids = tok.encode(b or "", add_special_tokens=False)
    half = over // 2 + 64
    a2 = tok.decode(a_ids[:max(50, len(a_ids) - half)], skip_special_tokens=True) if a_ids else a
    b2 = tok.decode(b_ids[:max(50, len(b_ids) - half)], skip_special_tokens=True) if b_ids else b
    prompt, n = _try_build_judge(tok, x_list, o_dict, a2, b2)
    if n <= max_input:
        return prompt, True

    # Pass 2: aggressively clip a and b to ~200 tokens each
    a3 = tok.decode(a_ids[:200], skip_special_tokens=True) if a_ids else a
    b3 = tok.decode(b_ids[:200], skip_special_tokens=True) if b_ids else b
    prompt, n = _try_build_judge(tok, x_list, o_dict, a3, b3)
    if n <= max_input:
        return prompt, True

    # Pass 3: drop oldest x messages until it fits (keep at least 1)
    msgs_x = list(x_list)
    while len(msgs_x) > 1 and n > max_input:
        msgs_x = msgs_x[1:]
        prompt, n = _try_build_judge(tok, msgs_x, o_dict, a3, b3)
    if n <= max_input:
        return prompt, True

    # Pass 4: clip the last remaining x message's content
    if msgs_x:
        last = dict(msgs_x[-1])
        c_ids = tok.encode(last.get("content", "") or "", add_special_tokens=False)
        last["content"] = tok.decode(c_ids[:200], skip_special_tokens=True) if c_ids else ""
        msgs_x = list(msgs_x[:-1]) + [last]
        prompt, n = _try_build_judge(tok, msgs_x, o_dict, a3, b3)

    return prompt, (n <= max_input)


def student_judge_offline(jobs_list, args):
    """Returns dict: {(eid, cond, order): 'A'|'B'|'tie'}."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    print(f"[student] loading {args.model} (bf16, max_model_len={args.max_model_len}, "
          f"max_num_seqs={args.max_num_seqs}) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    prompts = []
    keys = []
    n_truncated = 0
    n_skipped = 0
    max_user_input = args.max_model_len - 32  # judge max_tokens=4, +28 safety margin
    for (key, x_list, o_dict, a, b) in tqdm(jobs_list, desc="student: building prompts",
                                             mininterval=1.0, dynamic_ncols=True):
        # Quick path: try without truncation first via _try_build_judge,
        # only call _fit_judge_prompt if needed (saves chat-template work).
        prompt, n = _try_build_judge(tok, x_list, o_dict, a, b)
        if n <= max_user_input:
            keys.append(key); prompts.append(prompt)
            continue
        # Slow robust path
        prompt, ok = _fit_judge_prompt(tok, x_list, o_dict, a, b, max_user_input)
        if not ok:
            n_skipped += 1
            continue
        n_truncated += 1
        keys.append(key); prompts.append(prompt)

    print(f"[student] {len(prompts)} judge prompts "
          f"(truncated={n_truncated}, skipped_oversize={n_skipped})", flush=True)

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        trust_remote_code=True,
    )
    sampling = SamplingParams(n=1, temperature=0.0, max_tokens=4)

    t0 = time.time()
    outs = llm.generate(prompts, sampling, use_tqdm=True)
    print(f"[student] judging done in {(time.time()-t0)/60:.1f} min", flush=True)

    verdicts = {}
    for key, out in zip(keys, outs):
        verdicts[key] = parse_letter(out.outputs[0].text)
    return verdicts


# ─── GPT-4o-mini judge ───────────────────────────────────────────────────────

GPT_SYSTEM = (
    "You are an impartial judge of assistant responses. You will receive a "
    "conversation and two candidate next assistant responses labeled A and B. "
    "Decide which response is better. Output exactly one token: A or B."
)


def _gpt4o_call(client, model, x_list, o_dict, a, b, retries=4):
    user_text = build_judge_user_text(x_list, o_dict, a, b)
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": GPT_SYSTEM},
                    {"role": "user", "content": user_text},
                ],
                temperature=0.0,
                max_tokens=4,
            )
            return parse_letter(resp.choices[0].message.content)
        except Exception as e:
            if attempt == retries - 1:
                return "tie"
            time.sleep(2 ** attempt)
    return "tie"


def gpt4o_judge(jobs_list, args):
    """Returns dict: {(eid, cond, order): 'A'|'B'|'tie'}."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("[gpt4o] OPENAI_API_KEY not set; skipping", flush=True)
        return {}

    from openai import OpenAI
    client = OpenAI()

    verdicts = {}

    def worker(item):
        key, x_list, o_dict, a, b = item
        v = _gpt4o_call(client, args.openai_model, x_list, o_dict, a, b)
        return key, v

    print(f"[gpt4o] running {len(jobs_list)} calls with {args.openai_workers} workers ...",
          flush=True)
    pbar = tqdm(total=len(jobs_list), desc="gpt4o-mini",
                mininterval=2.0, dynamic_ncols=True)
    with ThreadPoolExecutor(max_workers=args.openai_workers) as ex:
        futs = {ex.submit(worker, item): item[0] for item in jobs_list}
        for fut in as_completed(futs):
            try:
                key, v = fut.result()
            except Exception as e:
                pbar.write(f"worker error: {e}")
                pbar.update(1)
                continue
            verdicts[key] = v
            pbar.update(1)
    pbar.close()
    return verdicts


# ─── Output assembly ─────────────────────────────────────────────────────────

def assemble_judgments(gens, student_verdicts, gpt4o_verdicts, output_path):
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Counters for summary
    counts = {(j, c): {"y_star": 0, "y": 0, "tie": 0}
              for j in ("student", "gpt4o_mini") for c in CONDS}
    agreement = {c: {"both": 0, "agree": 0} for c in CONDS}

    with open(out_path, "w") as f:
        for d in tqdm(gens, desc="assembling judgments", mininterval=1.0, dynamic_ncols=True):
            eid = d["example_id"]
            verdicts_block = {}
            for cond in CONDS:
                s_pair = {
                    "AB": student_verdicts.get((eid, cond, "AB")),
                    "BA": student_verdicts.get((eid, cond, "BA")),
                }
                o_pair = {
                    "AB": gpt4o_verdicts.get((eid, cond, "AB")),
                    "BA": gpt4o_verdicts.get((eid, cond, "BA")),
                }
                s_winner = resolve(s_pair) if s_pair["AB"] and s_pair["BA"] else None
                o_winner = resolve(o_pair) if o_pair["AB"] and o_pair["BA"] else None
                agree = (s_winner is not None and o_winner is not None
                         and s_winner == o_winner)

                verdicts_block[cond] = {
                    "student":    {"order_AB": s_pair["AB"], "order_BA": s_pair["BA"],
                                   "winner": s_winner},
                    "gpt4o_mini": {"order_AB": o_pair["AB"], "order_BA": o_pair["BA"],
                                   "winner": o_winner},
                    "agreement":  agree,
                }

                if s_winner is not None:
                    counts[("student", cond)][s_winner] += 1
                if o_winner is not None:
                    counts[("gpt4o_mini", cond)][o_winner] += 1
                if s_winner is not None and o_winner is not None:
                    agreement[cond]["both"] += 1
                    if agree:
                        agreement[cond]["agree"] += 1

            row = {
                "example_id": eid,
                "x": d["x"], "y": d["y"], "o": d["o"],
                "y_base":                 d.get("y_base", ""),
                "y_star_cond_xo":         d.get("y_star_cond_xo", ""),
                "y_star_cond_xyo":        d.get("y_star_cond_xyo", ""),
                "y_star_cond_xyo_ystart": d.get("y_star_cond_xyo_ystart", ""),
                "verdicts": verdicts_block,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return counts, agreement


def print_summary(counts, agreement, n_rows, summary_path):
    lines = []
    lines.append(f"N rows: {n_rows}")
    lines.append("")
    lines.append("=== Per-judge winrates (y_star vs y, both-orderings rule) ===")
    lines.append(f"{'judge':<12} {'cond':<18} {'wins':>6} {'losses':>7} {'ties':>5} "
                 f"{'N_decided':>10} {'winrate':>9} {'95% CI':>20}")
    for j in ("student", "gpt4o_mini"):
        for c in CONDS:
            cnt = counts[(j, c)]
            wins, losses, ties = cnt["y_star"], cnt["y"], cnt["tie"]
            n_dec = wins + losses
            p, lo, hi = wilson(wins, n_dec)
            lines.append(f"{j:<12} {c:<18} {wins:>6} {losses:>7} {ties:>5} "
                         f"{n_dec:>10} {p:>9.3f} [{lo:.3f}, {hi:.3f}]")
    lines.append("")
    lines.append("=== Agreement (student vs gpt4o_mini) per conditioning ===")
    for c in CONDS:
        a = agreement[c]
        rate = a["agree"] / a["both"] if a["both"] else 0.0
        lines.append(f"  {c:<18}  {a['agree']}/{a['both']}  = {rate:.4f}")
    text = "\n".join(lines)
    print(text)
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_path).write_text(text + "\n")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens",
        default="experiments-for-apr26-may1/wildchat_prefix_decision/data/"
                "wildchat_filtered_qwen3_4b_4variants_generations.jsonl")
    ap.add_argument("--output",
        default="experiments-for-apr26-may1/wildchat_prefix_decision/data/judgments.jsonl")
    ap.add_argument("--summary",
        default="experiments-for-apr26-may1/wildchat_prefix_decision/data/judgments_summary.txt")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--max_num_seqs", type=int, default=2048)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    ap.add_argument("--openai_model", default="gpt-4o-mini")
    ap.add_argument("--openai_workers", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", choices=["student", "gpt4o", "both"], default="both")
    args = ap.parse_args()

    gens = []
    with open(args.gens) as f:
        for line in f:
            if line.strip():
                gens.append(json.loads(line))
            if args.limit and len(gens) >= args.limit:
                break
    print(f"loaded {len(gens)} generations from {args.gens}", flush=True)

    jobs_list = list(build_jobs(gens))
    print(f"built {len(jobs_list)} judge jobs ({len(gens)} rows × 3 conds × 2 orderings, "
          f"minus rows with empty fields)", flush=True)

    student_verdicts, gpt4o_verdicts = {}, {}
    if args.only in ("student", "both"):
        student_verdicts = student_judge_offline(jobs_list, args)
    if args.only in ("gpt4o", "both"):
        gpt4o_verdicts = gpt4o_judge(jobs_list, args)

    counts, agreement = assemble_judgments(gens, student_verdicts, gpt4o_verdicts,
                                            args.output)
    print(f"wrote → {args.output}", flush=True)
    print_summary(counts, agreement, len(gens), args.summary)
    print(f"summary → {args.summary}")


if __name__ == "__main__":
    main()
