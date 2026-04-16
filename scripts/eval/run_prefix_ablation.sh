#!/bin/bash
# Run prefix ablation: generate y_base + y* at each prefix % (0-100)
# using the EXACT same prompts as generate_ystar_prefix.py.
# Only generates the single prefix variant at each %, not all 3 conditions.
#
# Usage:
#   bash scripts/eval/run_prefix_ablation.sh <SERVER_URL> <INPUT> <OUTPUT_DIR> [SUBSAMPLE]

set -euo pipefail

SERVER_URL="${1:?Usage: bash run_prefix_ablation.sh SERVER_URL INPUT OUTPUT_DIR [SUBSAMPLE]}"
INPUT="${2:?}"
OUTPUT_DIR="${3:?}"
SUBSAMPLE="${4:-1000}"

mkdir -p "$OUTPUT_DIR"

python3 << 'PYEOF'
import json, random, re, sys, os
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from tqdm import tqdm
from transformers import AutoTokenizer

SERVER_URL = os.environ["SERVER_URL"]
INPUT = os.environ["INPUT"]
OUTPUT_DIR = os.environ["OUTPUT_DIR"]
SUBSAMPLE = int(os.environ["SUBSAMPLE"])

# ── Auto-detect model ──
resp = requests.get(f"{SERVER_URL}/v1/models")
MODEL = resp.json()["data"][0]["id"]
print(f"Model: {MODEL}")

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

# ── Prompts (exact copy from generate_ystar_prefix.py) ──

SYSTEM_PREFIX = (
    "You are a helpful assistant. You will be shown a conversation, the user's "
    "follow-up feedback, and the beginning of a reference response. Study the "
    "partial response and the feedback carefully, then generate a complete, "
    "high-quality improved response from scratch. Do not simply continue from "
    "where the partial response ends — write a full response addressing the "
    "user's original request. Output only your response."
)

USER_PREFIX_TEMPLATE = """\
<conversation history>
{x}

<user follow-up>
{o}

<partial reference response (first {pct}% of tokens)>
{prefix}

Given the conversation, feedback, and the partial reference response above, \
generate a complete improved response from scratch."""

SYSTEM_NOPREFIX = (
    "You are a helpful assistant. Given a conversation and a follow-up message "
    "from the user, respond directly and concisely to the original request, "
    "taking the follow-up into account. Do not explain your reasoning. Output "
    "only your revised response."
)

USER_NOPREFIX_TEMPLATE = """\
<conversation history>
{x}

<user follow-up>
{o}

Given the above follow-up, provide an improved response to the original request."""

SYSTEM_FULL = (
    "You are a helpful assistant. You will be shown a conversation, the user's "
    "follow-up feedback, and a complete reference response. Study the reference "
    "response and the feedback carefully, then generate a new, improved response "
    "from scratch. Do not copy the reference response — write your own improved "
    "version that addresses the user's original request. Output only your response."
)

USER_FULL_TEMPLATE = """\
<conversation history>
{x}

<user follow-up>
{o}

<complete reference response>
{full_y}

Given the conversation, feedback, and the complete reference response above, \
generate an improved response from scratch."""

SYSTEM_YBASE = (
    "You are a helpful assistant. Respond directly and helpfully to the user's request."
)


# ── Helpers (exact copy from generate_ystar_prefix.py) ──

def format_conversation(x):
    parts = []
    for msg in x:
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n\n".join(parts)

def get_y_content(y):
    if isinstance(y, dict):
        return y.get("content", "") or ""
    return str(y) if y else ""

def get_o_content(o):
    if isinstance(o, dict):
        return o.get("content", "") or ""
    return str(o) if o else ""

def extract_prefix(y_content, prefix_frac):
    token_ids = tokenizer.encode(y_content, add_special_tokens=False)
    if not token_ids:
        return ""
    n_keep = max(1, int(len(token_ids) * prefix_frac))
    return tokenizer.decode(token_ids[:n_keep], skip_special_tokens=True)

def strip_think_blocks(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in text:
        text = text.split("<think>")[0]
    return text.strip()

def call_chat(messages):
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 1.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    for attempt in range(3):
        try:
            r = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload, timeout=300)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            return strip_think_blocks(content)
        except Exception as e:
            if attempt == 2:
                print(f"Error: {e}", file=sys.stderr)
                return ""
    return ""


# ── Load and subsample ──

with open(INPUT) as f:
    data = [json.loads(l) for l in f if l.strip()]
print(f"Loaded {len(data)} rows")

random.seed(42)
if len(data) > SUBSAMPLE:
    data = random.sample(data, SUBSAMPLE)
print(f"Subsampled to {len(data)}")


# ── Generate y_base at temp=1.0 ──

print("\nGenerating y_base (temp=1.0)...")

def gen_ybase(idx):
    row = data[idx]
    messages = [{"role": "system", "content": SYSTEM_YBASE}]
    for turn in row.get("x", []):
        messages.append({"role": turn["role"], "content": turn["content"]})
    return idx, call_chat(messages)

ybase = [None] * len(data)
with ThreadPoolExecutor(max_workers=32) as ex:
    futs = {ex.submit(gen_ybase, i): i for i in range(len(data))}
    for f in tqdm(as_completed(futs), total=len(data), desc="y_base"):
        idx, result = f.result()
        ybase[idx] = result


# ── Generate y* at each prefix fraction ──

FRACS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
prefix_results = {pct: [None] * len(data) for pct in FRACS}

for pct in FRACS:
    print(f"\nGenerating y* at prefix={pct}%...")

    def gen_prefix(idx, _pct=pct):
        row = data[idx]
        x_text = format_conversation(row.get("x", []))
        o_text = get_o_content(row.get("o", ""))
        y_text = get_y_content(row.get("y", ""))

        if _pct == 0:
            messages = [
                {"role": "system", "content": SYSTEM_NOPREFIX},
                {"role": "user", "content": USER_NOPREFIX_TEMPLATE.format(x=x_text, o=o_text)},
            ]
        elif _pct == 100:
            messages = [
                {"role": "system", "content": SYSTEM_FULL},
                {"role": "user", "content": USER_FULL_TEMPLATE.format(x=x_text, o=o_text, full_y=y_text)},
            ]
        else:
            frac = _pct / 100.0
            prefix = extract_prefix(y_text, frac)
            messages = [
                {"role": "system", "content": SYSTEM_PREFIX},
                {"role": "user", "content": USER_PREFIX_TEMPLATE.format(
                    x=x_text, o=o_text, prefix=prefix, pct=_pct
                )},
            ]
        return idx, call_chat(messages)

    with ThreadPoolExecutor(max_workers=32) as ex:
        futs = {ex.submit(gen_prefix, i): i for i in range(len(data))}
        for f in tqdm(as_completed(futs), total=len(data), desc=f"prefix={pct}%"):
            idx, result = f.result()
            prefix_results[pct][idx] = result


# ── Write merged output ──

outpath = os.path.join(OUTPUT_DIR, "prefix_ablation_merged.jsonl")
with open(outpath, "w") as f:
    for i, row in enumerate(data):
        out = {
            "conversation_id": row.get("conversation_id", ""),
            "turn_index": row.get("turn_index"),
            "x": row["x"],
            "y": row.get("y"),
            "o": row.get("o"),
            "y_base": ybase[i] or "",
        }
        for pct in FRACS:
            out[f"y_star_{pct}"] = prefix_results[pct][i] or ""
        f.write(json.dumps(out, ensure_ascii=False) + "\n")

print(f"\nWrote {len(data)} rows to {outpath}")
print(f"Fields: y_base, " + ", ".join(f"y_star_{p}" for p in FRACS))
PYEOF
