#!/usr/bin/env python3
"""
Step 2 — Measure FKL signal on the probe set for a single checkpoint.

Inference only (no optimizer/gradients): runs on 1 GPU with batched forward passes.
  - Pass 1: context = x only → logits over next token at each position in y.
  - Pass 2: context = x + hindsight block (Future User Message: o) → same.
Computes per-token KL(π(·|x,o,y_{<i}) || π(·|x,y_{<i})) (full softmax KL).
Writes one line per sample to output jsonl: id, category, per_token_kl.

Usage (1 GPU, batch_size=32):
  python scripts/fkl/measure_fkl_signal.py \
    --probe_set results/probe_set.json \
    --checkpoint /path/to/checkpoint \
    --output results/baseline_v1_signal.jsonl \
    --batch_size 32 \
    [--max_length 2048]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Hindsight block format (match offline_sdpo_trainer / SDPO)
HINDSIGHT_BLOCK_TEMPLATE = (
    "\n\n[HINDSIGHT CONTEXT]\n"
    "The following is a user response to your previous, insufficient attempt. Improve your response to the user prompt.\n"
    "Future User Message: {o}"
)


def _decode_single_token(tokenizer, token_id: int) -> str:
    """Decode a single token id for readable one-line logging."""
    try:
        return tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode([token_id], skip_special_tokens=False)


def _escape_token_str(s: str) -> str:
    """Escape whitespace so token strings stay one-line in logs."""
    return (
        (s or "")
        .replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def _content(msg: dict) -> str:
    return (msg.get("content") or msg.get("value") or "").strip()


def build_x_with_hindsight(x_messages: list, o: str) -> list:
    """Copy x and append hindsight block to the last message."""
    block = HINDSIGHT_BLOCK_TEMPLATE.format(o=o)
    out = copy.deepcopy(x_messages)
    if not out:
        return out
    last = out[-1]
    content = _content(last)
    # Preserve key format
    if "content" in last:
        last["content"] = content + block
    else:
        last["value"] = content + block
    return out


def apply_chat_template_opt(tokenizer, messages: list, add_generation_prompt: bool) -> str:
    """Apply chat template; pass enable_thinking=False if supported (Qwen)."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def _prepare_one_sample(
    tokenizer,
    sample: dict,
    device: torch.device,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int] | None:
    """
    Build input_ids_no_o, input_ids_with_o, and lengths for one sample.
    Returns (input_ids_no_o, input_ids_with_o, y_tokens, len_prompt_no_o, len_prompt_with_o, n_y)
    or None if sample has no y.
    """
    x = sample["x"]
    y_text = (sample.get("y") or "").strip()
    o_text = (sample.get("o") or "").strip()
    if not y_text:
        return None

    prompt_no_o = apply_chat_template_opt(tokenizer, x, add_generation_prompt=True)
    x_hindsight = build_x_with_hindsight(x, o_text)
    prompt_with_o = apply_chat_template_opt(tokenizer, x_hindsight, add_generation_prompt=True)

    toks_no_o = tokenizer(prompt_no_o, add_special_tokens=True, return_tensors="pt")
    y_tokens = tokenizer(y_text, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)
    if y_tokens.dim() == 0:
        y_tokens = y_tokens.unsqueeze(0)
    toks_with_o = tokenizer(prompt_with_o, add_special_tokens=True, return_tensors="pt")

    len_prompt_no_o = toks_no_o["input_ids"].shape[1]
    len_prompt_with_o = toks_with_o["input_ids"].shape[1]
    n_y = y_tokens.shape[0]

    if len_prompt_no_o + n_y > max_length:
        keep = max(0, max_length - n_y)
        toks_no_o["input_ids"] = toks_no_o["input_ids"][:, -keep:]
        len_prompt_no_o = toks_no_o["input_ids"].shape[1]
    if len_prompt_with_o + n_y > max_length:
        keep = max(0, max_length - n_y)
        toks_with_o["input_ids"] = toks_with_o["input_ids"][:, -keep:]
        len_prompt_with_o = toks_with_o["input_ids"].shape[1]

    input_ids_no_o = torch.cat([toks_no_o["input_ids"], y_tokens.unsqueeze(0)], dim=1)
    input_ids_with_o = torch.cat([toks_with_o["input_ids"], y_tokens.unsqueeze(0)], dim=1)
    return (
        input_ids_no_o.squeeze(0),
        input_ids_with_o.squeeze(0),
        y_tokens,
        len_prompt_no_o,
        len_prompt_with_o,
        n_y,
    )


def _run_batch_kl(
    model,
    tokenizer,
    batch: list[dict],
    batch_tensors: list[tuple],
    device: torch.device,
    pad_id: int,
) -> tuple[list[list[float]], list[list[float]]]:
    """Run two forward passes and return per-sample (per_token_kl, per_token_logprob_delta)."""
    if not batch_tensors:
        empty = [[] for _ in batch]
        return empty, empty

    ids_no = [t[0] for t in batch_tensors]
    ids_with = [t[1] for t in batch_tensors]
    lens_no = [t[3] for t in batch_tensors]
    lens_with = [t[4] for t in batch_tensors]
    n_ys = [t[5] for t in batch_tensors]

    max_len_no = max(seq.shape[0] for seq in ids_no)
    max_len_with = max(seq.shape[0] for seq in ids_with)

    def pad_cat(seqs: list[torch.Tensor], max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        padded = []
        for s in seqs:
            pad_len = max_len - s.shape[0]
            padded.append(
                F.pad(s, (0, pad_len), value=pad_id) if pad_len > 0 else s
            )
        stacked = torch.stack(padded).to(device)
        mask = (stacked != pad_id).long()
        return stacked, mask

    input_no, mask_no = pad_cat(ids_no, max_len_no)
    input_with, mask_with = pad_cat(ids_with, max_len_with)

    with torch.no_grad():
        out_no = model(input_ids=input_no, attention_mask=mask_no)
        logits_no = out_no.logits  # (B, L_no, V)
        out_with = model(input_ids=input_with, attention_mask=mask_with)
        logits_with = out_with.logits  # (B, L_with, V)

    results_kl = []
    results_delta = []
    for b in range(len(batch_tensors)):
        len_p_no = lens_no[b]
        len_p_with = lens_with[b]
        n_y = n_ys[b]
        kls = []
        deltas = []
        y_tokens = batch_tensors[b][2]
        y_ids = y_tokens.tolist() if hasattr(y_tokens, "tolist") else list(y_tokens)
        for i in range(n_y):
            idx_no = len_p_no - 1 + i
            idx_with = len_p_with - 1 + i
            if idx_no < 0 or idx_with < 0:
                continue
            if idx_no >= logits_no.shape[1] or idx_with >= logits_with.shape[1]:
                break
            logits_p = logits_no[b, idx_no, :].float()
            logits_q = logits_with[b, idx_with, :].float()
            log_p = F.log_softmax(logits_p, dim=-1)
            log_q = F.log_softmax(logits_q, dim=-1)
            q = log_q.exp()
            kl = F.kl_div(log_p, q, reduction="sum").item()
            kls.append(kl)
            if i < len(y_ids):
                token_id = int(y_ids[i])
                # Signed hindsight effect on realized token y_i.
                deltas.append((log_q[token_id] - log_p[token_id]).item())
        results_kl.append(kls)
        results_delta.append(deltas)
    return results_kl, results_delta


def per_token_kl(
    model,
    tokenizer,
    sample: dict,
    device: torch.device,
    dtype: torch.dtype,
    max_length: int,
) -> list[float]:
    """
    For one probe sample (x, y, o), compute at each token position i in y:
    KL(π(·|x,o,y_{<i}) || π(·|x,y_{<i})).
    Returns list of KL values (one per token in y). Used when batch_size=1.
    """
    prep = _prepare_one_sample(tokenizer, sample, device, max_length)
    if prep is None:
        return []
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    kls, _ = _run_batch_kl(model, tokenizer, [sample], [prep], device, pad_id)
    return kls[0]


def main():
    parser = argparse.ArgumentParser(description="Measure FKL signal on probe set for one checkpoint.")
    parser.add_argument("--probe_set", type=Path, required=True, help="Path to probe_set.json")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint dir (model + tokenizer)")
    parser.add_argument("--output", type=Path, required=True, help="Output jsonl path (one line per sample)")
    parser.add_argument(
        "--token_kl_txt_out_dir",
        type=Path,
        default=None,
        help="Optional: write per-sample token/value logs as txt files.",
    )
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length (truncate to avoid OOM)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for forward passes (1 GPU inference); reduce if OOM")
    parser.add_argument("--device", type=str, default=None, help="Device (default: cuda if available)")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16
    pad_id = None  # set after tokenizer load

    # Load probe set
    with open(args.probe_set) as f:
        probe = json.load(f)
    if isinstance(probe, dict):
        probe = probe.get("samples", probe.get("data", [probe]))
    if not isinstance(probe, list):
        probe = list(probe)
    print(f"Loaded {len(probe)} probe samples from {args.probe_set}")

    # Load model and tokenizer from checkpoint (inference only — 1 GPU, no optimizer)
    # IMPORTANT: do not call .resolve() here; HuggingFace repo ids (e.g. "Qwen/Qwen3-4B")
    # would otherwise be treated as local filesystem paths.
    ckpt = str(args.checkpoint)
    # Paths starting with / or . are filesystem; HF treats them as repo_id otherwise
    is_local = ckpt.startswith("/") or ckpt.startswith(".")
    load_kw = {"local_files_only": True} if is_local else {}
    print(f"Loading model and tokenizer from {ckpt} ...")
    tokenizer = AutoTokenizer.from_pretrained(ckpt, **load_kw)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
        attn_implementation="sdpa",
        **load_kw,
    )
    if device.type != "cuda" or getattr(model, "device_map", None) is None and getattr(model, "hf_device_map", None) is None:
        model = model.to(device)
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    batch_size = max(1, args.batch_size)

    ckpt_tag = Path(ckpt).name
    token_out_dir = None
    if args.token_kl_txt_out_dir is not None:
        token_out_dir = args.token_kl_txt_out_dir / ckpt_tag
        token_out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w") as out_f:
        for start in tqdm(range(0, len(probe), batch_size), desc="Probe"):
            batch = probe[start : start + batch_size]
            batch_tensors = []
            for s in batch:
                prep = _prepare_one_sample(tokenizer, s, device, args.max_length)
                batch_tensors.append(prep)

            valid = [(i, t) for i, t in enumerate(batch_tensors) if t is not None]
            if not valid:
                for s in batch:
                    out_f.write(json.dumps({"id": s.get("id", ""), "category": s.get("category", ""), "per_token_kl": []}) + "\n")
                out_f.flush()
                continue

            indices, preps = zip(*valid)
            try:
                kls_list, deltas_list = _run_batch_kl(
                    model, tokenizer, [batch[i] for i in indices], list(preps), device, pad_id
                )
            except torch.cuda.OutOfMemoryError:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                # Fallback: process this batch one sample at a time
                kls_list = []
                deltas_list = []
                for i in indices:
                    try:
                        kls_one, deltas_one = _run_batch_kl(
                            model, tokenizer, [batch[i]], [batch_tensors[i]], device, pad_id
                        )
                        kls = kls_one[0]
                        deltas = deltas_one[0]
                    except Exception:
                        kls = []
                        deltas = []
                    kls_list.append(kls)
                    deltas_list.append(deltas)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            except Exception as e:
                print(f"[WARN] batch at {start}: {e}", file=sys.stderr)
                kls_list = [[] for _ in indices]
                deltas_list = [[] for _ in indices]

            # Reorder: batch order, with [] for skip_idx
            result_by_batch_idx = {}
            for ii, kls, deltas in zip(indices, kls_list, deltas_list):
                result_by_batch_idx[ii] = (kls, deltas)
            for i, s in enumerate(batch):
                kls, deltas = result_by_batch_idx.get(i, ([], []))
                out_f.write(json.dumps({
                    "id": s.get("id", ""),
                    "category": s.get("category", ""),
                    "per_token_kl": kls,
                    "per_token_logprob_delta": deltas,
                }) + "\n")

                if token_out_dir is not None:
                    prep = batch_tensors[i]
                    sample_id = s.get("id", "")
                    category = s.get("category", "")
                    tok_path = token_out_dir / f"{sample_id}.txt"

                    if prep is None:
                        tok_path.write_text(
                            f"sample_id={sample_id}\ncategory={category}\nstatus=prep_missing\n",
                            encoding="utf-8",
                        )
                        continue

                    y_tokens = prep[2]
                    token_ids = y_tokens.tolist() if hasattr(y_tokens, "tolist") else list(y_tokens)
                    token_strs = [
                        _escape_token_str(_decode_single_token(tokenizer, int(tid))) for tid in token_ids
                    ]

                    m = min(len(token_ids), len(kls), len(deltas)) if kls is not None else 0
                    mismatch_note = ""
                    if len(token_ids) != len(kls) or len(token_ids) != len(deltas):
                        mismatch_note = (
                            f"\n#mismatch: len(y_tokens)={len(token_ids)} "
                            f"len(per_token_kl)={len(kls)} len(per_token_logprob_delta)={len(deltas)}"
                        )

                    with open(tok_path, "w", encoding="utf-8") as tf:
                        tf.write(f"sample_id={sample_id}\ncategory={category}\n")
                        tf.write("token_index\ttoken_id\ttoken_str\tkl\tlogprob_delta\n")
                        for idx in range(m):
                            tf.write(
                                f"{idx}\t{int(token_ids[idx])}\t{token_strs[idx]}\t"
                                f"{float(kls[idx]):.10f}\t{float(deltas[idx]):.10f}\n"
                            )
                        tf.write(mismatch_note + "\n" if mismatch_note else "")
            out_f.flush()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
