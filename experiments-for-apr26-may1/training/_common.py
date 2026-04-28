"""Shared helpers for the SFT/FKL/PC-SDPO training scripts.

- model + tokenizer + LoRA loading
- dataset row parsing (WC chat vs WI raw text)
- prompt builders for {x}, {x,o}, {x,y,o} contexts
- collator skeleton
- step → checkpoint scheduler
- short labels for wandb run names
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


# ─── Model / LoRA setup ──────────────────────────────────────────────────────

def load_model_and_tokenizer(model_name: str, lora_r: int = 16, lora_alpha: int = 32,
                              lora_dropout: float = 0.05,
                              attn_impl: str = "auto") -> Tuple[Any, Any]:
    """Load model with LoRA adapters; return (model, tokenizer)."""
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    extra = {}
    if attn_impl != "auto":
        extra["attn_implementation"] = attn_impl
    else:
        try:
            from transformers.utils import is_flash_attn_2_available
            if is_flash_attn_2_available():
                extra["attn_implementation"] = "flash_attention_2"
            else:
                extra["attn_implementation"] = "sdpa"
        except Exception:
            extra["attn_implementation"] = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        **extra,
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tok


# ─── Dataset row parsing ─────────────────────────────────────────────────────

def is_wildchat_row(d: Dict[str, Any]) -> bool:
    """If x is a list of message dicts, the row is a WildChat row (chat template).
    If x is a string, it's a WebInstruct row (raw text)."""
    return isinstance(d.get("x"), list)


def get_text(field):
    """y/o may be dicts (chat) or strings (raw). Pull the displayable text."""
    if field is None:
        return ""
    if isinstance(field, dict):
        return field.get("content", "") or ""
    return str(field)


# ─── Prompt builders ─────────────────────────────────────────────────────────

WI_PROMPT_X     = "Problem:\n{x}\n\nSolution:\n"
WI_PROMPT_XYO   = ("Problem:\n{x}\n\nA student attempted the following solution:\n"
                   "{y}\n\nCritique of the attempt:\n{o}\n\n"
                   "Using the critique, write a corrected, complete solution to the problem.\n\n"
                   "Solution:\n")
WI_PROMPT_XO    = ("Problem:\n{x}\n\nFeedback on a previous attempt at this problem:\n"
                   "{o}\n\nWrite a complete, correct solution to the problem.\n\n"
                   "Solution:\n")


def build_prompt_text(d: Dict[str, Any], context: str, tokenizer) -> str:
    """Build the prompt string (no completion) for one of:
       'x'   — vanilla student context
       'xo'  — (x, o)
       'xyo' — (x, y, o)
    Routes to chat template (WildChat) or WI raw text format.
    """
    assert context in ("x", "xo", "xyo")

    if is_wildchat_row(d):
        # WildChat: chat template with multi-turn message list
        x_list = d["x"]
        y_dict = d["y"]
        o_dict = d["o"]

        if context == "x":
            messages = list(x_list)
        elif context == "xo":
            messages = list(x_list) + [o_dict]
        elif context == "xyo":
            messages = list(x_list) + [y_dict, o_dict]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    else:
        # WebInstruct: raw text (Qwen2.5-Math-7B base, no chat template)
        x = d["x"]
        y = get_text(d.get("y"))
        o = get_text(d.get("o"))
        if context == "x":
            return WI_PROMPT_X.format(x=x)
        if context == "xo":
            return WI_PROMPT_XO.format(x=x, o=o)
        if context == "xyo":
            return WI_PROMPT_XYO.format(x=x, y=y, o=o)


def get_completion_text(d: Dict[str, Any]) -> str:
    """y_star — the target completion (string for both WC and WI datasets after Prompt 4)."""
    s = d.get("y_star")
    if isinstance(s, dict):
        return s.get("content", "") or ""
    return s or ""


# ─── Tokenization with completion mask ───────────────────────────────────────

def tokenize_prompt_completion(prompt_text: str, completion_text: str,
                                tokenizer, max_length: int,
                                append_eos: bool = True
                                ) -> Tuple[List[int], List[int], int]:
    """Return (input_ids, completion_mask, prompt_len). completion_mask is 1
    for completion tokens, 0 for prompt tokens. Truncates from prompt side
    if too long."""
    if append_eos and tokenizer.eos_token is not None and \
            not completion_text.endswith(tokenizer.eos_token):
        completion_text = completion_text + tokenizer.eos_token

    prompt_ids     = tokenizer.encode(prompt_text,     add_special_tokens=False)
    completion_ids = tokenizer.encode(completion_text, add_special_tokens=False)

    # Truncate prompt from the LEFT to fit
    budget_prompt = max(64, max_length - len(completion_ids))
    if len(prompt_ids) > budget_prompt:
        prompt_ids = prompt_ids[-budget_prompt:]

    # Truncate completion from the right if total still too long
    if len(prompt_ids) + len(completion_ids) > max_length:
        completion_ids = completion_ids[: max_length - len(prompt_ids)]

    input_ids = prompt_ids + completion_ids
    completion_mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
    return input_ids, completion_mask, len(prompt_ids)


# ─── Checkpoint scheduling ───────────────────────────────────────────────────

def evenly_spaced_steps(total_steps: int, n_intermediate: int = 4) -> List[int]:
    """Return n_intermediate step numbers, evenly spaced, plus the final step.
    Returns a sorted list. e.g. total=100, n=4 → [20, 40, 60, 80, 100]."""
    if total_steps <= 0:
        return []
    if n_intermediate <= 0:
        return [total_steps]
    chunk = total_steps / (n_intermediate + 1)
    intermediate = [max(1, int(round((i + 1) * chunk))) for i in range(n_intermediate)]
    intermediate = sorted(set(intermediate + [total_steps]))
    intermediate = [s for s in intermediate if s <= total_steps]
    return intermediate


# ─── Wandb naming helpers ────────────────────────────────────────────────────

def short_model_name(model_name: str) -> str:
    """Qwen/Qwen3-4B → qwen3-4b ; Qwen/Qwen2.5-Math-7B → qwen25-math-7b ."""
    s = model_name.lower().split("/")[-1]
    s = s.replace("qwen2.5", "qwen25")
    s = re.sub(r"[^a-z0-9._-]", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def short_dataset_name(path: str) -> str:
    """File stem, sanitized."""
    stem = Path(path).stem
    return re.sub(r"[^a-z0-9._-]", "-", stem.lower()).strip("-")


def parse_dataset_facets(dataset_path: str) -> dict:
    """Pull (family, direction, conditioning) from a dataset path. Returns
    {family, direction, conditioning, dataset_short}. Best-effort: missing
    facets become 'unknown'."""
    p = Path(dataset_path)
    stem = p.stem.lower()
    parent = p.parent.name.lower()

    family = "unknown"
    if "wildchat" in str(p).lower(): family = "wildchat"
    elif "webinstruct" in str(p).lower(): family = "webinstruct"

    direction = "unknown"
    if stem.startswith("teacher_wins"):       direction = "wins"
    elif stem.startswith("teacher_loses"):    direction = "loses"

    conditioning = "unknown"
    for c in ("cond_xyo_ystart", "cond_xyo", "cond_xo"):
        if c in stem:
            conditioning = c; break

    return {"family":       family,
            "direction":    direction,
            "conditioning": conditioning,
            "dataset_short": short_dataset_name(dataset_path)}


def build_run_name(run_id: str, objective: str, model_name: str, dataset_path: str) -> str:
    """Long, descriptive wandb run name. Order: family→model→direction→
    conditioning→objective→run_id, so wandb sorts naturally by experiment."""
    f = parse_dataset_facets(dataset_path)
    return (f"{f['family']}__{short_model_name(model_name)}__{f['direction']}"
            f"__{f['conditioning']}__{objective}__{run_id}")


def build_run_tags(run_id: str, objective: str, model_name: str, dataset_path: str) -> list:
    """Wandb tags for filtering. Each is a single token."""
    f = parse_dataset_facets(dataset_path)
    return [
        f["family"],                      # webinstruct | wildchat
        short_model_name(model_name),     # qwen3-4b | qwen25-math-7b
        f["direction"],                   # wins | loses
        f["conditioning"],                # cond_xo | cond_xyo | cond_xyo_ystart
        objective,                        # sft | fkl | sdpo | pc_sdpo
        run_id,                           # WI-1, WC-7, ...
    ]


# ─── jsonl loader ────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows
