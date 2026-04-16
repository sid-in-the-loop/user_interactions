"""
Extended LoRA training with 8 additional methods beyond SFT/FKL/JSD.

Objectives:
  Offline (no rollouts):
    dpo       — DPO: y*=chosen, y=rejected, ref=frozen init
    jsd_is1   — JSD + sequence-level IS weights (clipped)
    jsd_is2   — JSD + token-level clipped NLL weighting
    jsd_is3   — JSD + token masking (p_θ0 > τ → zero grad)
    jsd_is4   — JSD + length-normalized surprisal weighting

  Rollout (stubs — need generation infra):
    rkl       — Reverse KL with REINFORCE
    rlad      — RKL + clipped trust region (ε=0.2)
    distillm2 — DistiLLM-2: SKL on y* + SRKL on rollouts

  Existing (imported from train_lora.py):
    sft, fkl, jsd

Usage:
  # Precompute init logprobs first (for IS methods):
  python scripts/fkl/precompute_logprobs.py --input DATA --output logprobs.pt

  # Train:
  python scripts/fkl/train_methods.py \
      --input DATA --y_star_field y_star_prefix30 \
      --objective jsd_is1 --pure_kl \
      --logprobs_file logprobs.pt \
      --model Qwen/Qwen3-8B --output_dir checkpoints/jsd_is1
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

try:
    from peft import LoraConfig, get_peft_model, TaskType
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def get_content(obj) -> str:
    if obj is None:           return ""
    if isinstance(obj, str):  return obj
    if isinstance(obj, dict): return obj.get("content") or ""
    return str(obj)


def infer_condition(y_star_field: str) -> str:
    if "prefix30" in y_star_field: return "prefix30"
    if "noprefix" in y_star_field: return "noprefix"
    if "full"     in y_star_field: return "full"
    return "prefix30"


# ─────────────────────────────────────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────────────────────────────────────

class PrefixDataset(Dataset):
    """Student [x + y*] and Teacher [x + prefix(y) + o + y*] tokenization."""

    def __init__(self, data, tokenizer, max_length, y_star_field, teacher_max_length=None):
        self.condition = infer_condition(y_star_field)
        self.teacher_max_length = teacher_max_length or min(max_length * 2, 8192)
        self.items = self._tokenize(data, tokenizer, max_length, y_star_field)

    def _tokenize(self, data, tokenizer, max_length, y_star_field):
        items, skipped = [], 0
        for orig_idx, item in enumerate(tqdm(data, desc="Tokenizing")):
            y_star_raw = item.get(y_star_field)
            if not y_star_raw:
                skipped += 1; continue

            y_star_msg = {"role": "assistant", "content": get_content(y_star_raw)}
            x = item["x"]

            # Student: [x + y*]
            try:
                s_prompt_text = tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True)
                s_full_text = tokenizer.apply_chat_template(list(x) + [y_star_msg], tokenize=False, add_generation_prompt=False)
            except Exception:
                skipped += 1; continue

            s_prompt_len = len(tokenizer(s_prompt_text, add_special_tokens=False)["input_ids"])
            s_enc = tokenizer(s_full_text, add_special_tokens=False, truncation=True, max_length=max_length)
            s_input_ids = s_enc["input_ids"]
            s_labels = [-100] * len(s_input_ids)
            if s_prompt_len < len(s_input_ids):
                s_labels[s_prompt_len:] = s_input_ids[s_prompt_len:]
            if all(l == -100 for l in s_labels):
                skipped += 1; continue

            # Teacher: [x + prefix(y) + o + y*]
            o_content = get_content(item.get("o", ""))
            y_content = get_content(item.get("y", ""))
            teacher_ctx = list(x)
            if self.condition == "prefix30":
                y_ids = tokenizer(y_content, add_special_tokens=False)["input_ids"]
                prefix_text = tokenizer.decode(y_ids[:30], skip_special_tokens=True)
                if prefix_text.strip():
                    teacher_ctx.append({"role": "assistant", "content": prefix_text})
            elif self.condition == "full":
                if y_content:
                    teacher_ctx.append({"role": "assistant", "content": y_content})
            if o_content:
                teacher_ctx.append({"role": "user", "content": o_content})

            try:
                t_prompt_text = tokenizer.apply_chat_template(teacher_ctx, tokenize=False, add_generation_prompt=True)
                t_full_text = tokenizer.apply_chat_template(teacher_ctx + [y_star_msg], tokenize=False, add_generation_prompt=False)
            except Exception:
                skipped += 1; continue

            t_prompt_len = len(tokenizer(t_prompt_text, add_special_tokens=False)["input_ids"])
            t_enc = tokenizer(t_full_text, add_special_tokens=False, truncation=True, max_length=self.teacher_max_length)
            t_input_ids = t_enc["input_ids"]
            t_labels = [-100] * len(t_input_ids)
            if t_prompt_len < len(t_input_ids):
                t_labels[t_prompt_len:] = t_input_ids[t_prompt_len:]
            if all(l == -100 for l in t_labels):
                skipped += 1; continue

            items.append({
                "student_input_ids": s_input_ids,
                "student_attention_mask": s_enc["attention_mask"],
                "student_labels": s_labels,
                "teacher_input_ids": t_input_ids,
                "teacher_attention_mask": t_enc["attention_mask"],
                "teacher_labels": t_labels,
                "orig_idx": orig_idx,
            })

        print(f"[PrefixDataset] {len(items)} usable, {skipped} skipped")
        return items

    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]


class DPODataset(Dataset):
    """Tokenize both y* (chosen) and y (rejected) with student prompt [x]."""

    def __init__(self, data, tokenizer, max_length, y_star_field):
        self.items = []
        skipped = 0
        for item in tqdm(data, desc="Tokenizing DPO"):
            y_star_raw = item.get(y_star_field)
            y_raw = item.get("y")
            if not y_star_raw or not y_raw:
                skipped += 1; continue

            y_star_msg = {"role": "assistant", "content": get_content(y_star_raw)}
            y_rej_msg = {"role": "assistant", "content": get_content(y_raw)}
            x = item["x"]

            try:
                prompt_text = tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True)
                chosen_text = tokenizer.apply_chat_template(list(x) + [y_star_msg], tokenize=False, add_generation_prompt=False)
                rejected_text = tokenizer.apply_chat_template(list(x) + [y_rej_msg], tokenize=False, add_generation_prompt=False)
            except Exception:
                skipped += 1; continue

            prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])

            # Chosen
            c_enc = tokenizer(chosen_text, add_special_tokens=False, truncation=True, max_length=max_length)
            c_ids = c_enc["input_ids"]
            c_labels = [-100] * len(c_ids)
            if prompt_len < len(c_ids):
                c_labels[prompt_len:] = c_ids[prompt_len:]

            # Rejected
            r_enc = tokenizer(rejected_text, add_special_tokens=False, truncation=True, max_length=max_length)
            r_ids = r_enc["input_ids"]
            r_labels = [-100] * len(r_ids)
            if prompt_len < len(r_ids):
                r_labels[prompt_len:] = r_ids[prompt_len:]

            if all(l == -100 for l in c_labels) or all(l == -100 for l in r_labels):
                skipped += 1; continue

            self.items.append({
                "chosen_input_ids": c_ids,
                "chosen_attention_mask": c_enc["attention_mask"],
                "chosen_labels": c_labels,
                "rejected_input_ids": r_ids,
                "rejected_attention_mask": r_enc["attention_mask"],
                "rejected_labels": r_labels,
            })

        print(f"[DPODataset] {len(self.items)} usable, {skipped} skipped")

    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]


class IndexedDataset(Dataset):
    """Wraps PrefixDataset to pass original JSONL index for IS weight lookup."""

    def __init__(self, prefix_dataset: PrefixDataset):
        self.ds = prefix_dataset

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        item["sample_idx"] = item.get("orig_idx", idx)
        return item


# ─────────────────────────────────────────────────────────────────────────────
# Collation
# ─────────────────────────────────────────────────────────────────────────────

def pad_field(seqs, pad_val=0):
    ml = max(len(s) for s in seqs)
    return [s + [pad_val] * (ml - len(s)) for s in seqs]


def collate(batch, pad_id):
    out = {
        "student_input_ids":      torch.tensor(pad_field([x["student_input_ids"] for x in batch]), dtype=torch.long),
        "student_attention_mask": torch.tensor(pad_field([x["student_attention_mask"] for x in batch]), dtype=torch.long),
        "student_labels":         torch.tensor(pad_field([x["student_labels"] for x in batch], -100), dtype=torch.long),
        "teacher_input_ids":      torch.tensor(pad_field([x["teacher_input_ids"] for x in batch]), dtype=torch.long),
        "teacher_attention_mask": torch.tensor(pad_field([x["teacher_attention_mask"] for x in batch]), dtype=torch.long),
        "teacher_labels":         torch.tensor(pad_field([x["teacher_labels"] for x in batch], -100), dtype=torch.long),
    }
    if "sample_idx" in batch[0]:
        out["sample_idx"] = torch.tensor([x["sample_idx"] for x in batch], dtype=torch.long)
    return out


def collate_dpo(batch, pad_id):
    return {
        "chosen_input_ids":       torch.tensor(pad_field([x["chosen_input_ids"] for x in batch]), dtype=torch.long),
        "chosen_attention_mask":  torch.tensor(pad_field([x["chosen_attention_mask"] for x in batch]), dtype=torch.long),
        "chosen_labels":          torch.tensor(pad_field([x["chosen_labels"] for x in batch], -100), dtype=torch.long),
        "rejected_input_ids":     torch.tensor(pad_field([x["rejected_input_ids"] for x in batch]), dtype=torch.long),
        "rejected_attention_mask":torch.tensor(pad_field([x["rejected_attention_mask"] for x in batch]), dtype=torch.long),
        "rejected_labels":        torch.tensor(pad_field([x["rejected_labels"] for x in batch], -100), dtype=torch.long),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

def sft_loss(student_logits, labels):
    shift_logits = student_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    B, T, V = shift_logits.shape
    return F.cross_entropy(shift_logits.reshape(B * T, V), shift_labels.reshape(B * T),
                           ignore_index=-100, reduction="mean")


def fkl_loss(student_ystar, teacher_ystar):
    log_p = torch.log_softmax(student_ystar, dim=-1)
    q = torch.softmax(teacher_ystar, dim=-1).detach()
    return -(q * log_p).sum(dim=-1).mean()


def jsd_loss(student_ystar, teacher_ystar, beta=0.5):
    p_T = torch.softmax(teacher_ystar, dim=-1).detach()
    p_S = torch.softmax(student_ystar, dim=-1)
    m = (beta * p_T + (1 - beta) * p_S).clamp(min=1e-10)
    kl_T = (p_T * (p_T.clamp(min=1e-10).log() - m.log())).sum(dim=-1)
    kl_S = (p_S * (p_S.clamp(min=1e-10).log() - m.log())).sum(dim=-1)
    return (beta * kl_T + (1 - beta) * kl_S).mean()


def jsd_loss_per_token(student_ystar, teacher_ystar, beta=0.5):
    """Returns per-token JSD values (not reduced)."""
    p_T = torch.softmax(teacher_ystar, dim=-1).detach()
    p_S = torch.softmax(student_ystar, dim=-1)
    m = (beta * p_T + (1 - beta) * p_S).clamp(min=1e-10)
    kl_T = (p_T * (p_T.clamp(min=1e-10).log() - m.log())).sum(dim=-1)
    kl_S = (p_S * (p_S.clamp(min=1e-10).log() - m.log())).sum(dim=-1)
    return beta * kl_T + (1 - beta) * kl_S


def skl_loss(student_ystar, teacher_ystar, alpha=0.1):
    """Skewed KL: KL(p_T || α·p_T + (1-α)·p_S). For DistiLLM-2."""
    p_T = torch.softmax(teacher_ystar, dim=-1).detach()
    p_S = torch.softmax(student_ystar, dim=-1)
    m = (alpha * p_T + (1 - alpha) * p_S).clamp(min=1e-10)
    kl = (p_T * (p_T.clamp(min=1e-10).log() - m.log())).sum(dim=-1)
    return kl.mean()


def extract_ystar_logits(logits, labels):
    shift_logits = logits[:, :-1, :]
    shift_mask = labels[:, 1:] != -100
    return [shift_logits[i][shift_mask[i]] for i in range(logits.shape[0])]


def extract_ystar_token_ids(labels):
    """Extract y* token IDs per sample (shifted, non -100 positions)."""
    shift_labels = labels[:, 1:]
    shift_mask = shift_labels != -100
    return [shift_labels[i][shift_mask[i]] for i in range(labels.shape[0])]


def kl_from_ystar_lists(s_list, t_list, objective, beta=0.5):
    losses = []
    for s_y, t_y in zip(s_list, t_list):
        n = min(s_y.shape[0], t_y.shape[0])
        if n == 0: continue
        if objective == "fkl":
            losses.append(fkl_loss(s_y[:n], t_y[:n]))
        else:
            losses.append(jsd_loss(s_y[:n], t_y[:n], beta=beta))
    if not losses:
        return torch.tensor(0.0, device="cuda", requires_grad=True) * 0.0
    return torch.stack(losses).mean()


def sequence_logprob(logits, labels):
    """Compute sum of log probs on labeled (non -100) tokens per sample."""
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    B = logits.shape[0]
    results = []
    for i in range(B):
        mask = shift_labels[i] != -100
        if mask.sum() == 0:
            results.append(torch.tensor(0.0, device=logits.device))
            continue
        tids = shift_labels[i][mask]
        lps = log_probs[i][mask].gather(1, tids.unsqueeze(1)).squeeze(1)
        results.append(lps.sum())
    return torch.stack(results)


# ─────────────────────────────────────────────────────────────────────────────
# DPO loss
# ─────────────────────────────────────────────────────────────────────────────

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """Standard DPO loss."""
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# IS-weighted JSD losses
# ─────────────────────────────────────────────────────────────────────────────

def load_init_logprobs(path):
    """Load precomputed per-token logprobs from .pt file."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    print(f"[IS] Loaded {len(data)} logprob tensors from {path}")
    return data


def zg_jsd_loss(s_list, t_list, sample_indices, init_logprobs,
                teacher_labels_list, beta=0.5, eps_hi=4.0):
    """
    Zone-Gated JSD: weight each token by how informative the teacher-student gap is.

    delta_t = log_p_T(y*_t) - log_p_theta0(y*_t)
    w_t = clip(delta_t, 0, eps_hi) / eps_hi

    - Blue tokens (delta < 0): w=0, student already better
    - White tokens (delta ≈ 0): w≈0, nothing to learn
    - Learnable red (0 < delta < eps_hi): w proportional
    - Extreme red (delta > eps_hi): w=1, capped

    teacher_labels_list: list of token ID tensors per sample (y* token IDs at teacher positions)
    """
    losses = []
    all_weights = []

    for i, (s_y, t_y) in enumerate(zip(s_list, t_list)):
        n = min(s_y.shape[0], t_y.shape[0])
        if n == 0:
            continue

        idx = sample_indices[i]

        # Teacher log prob on actual y* tokens (from teacher logits)
        teacher_log_p = torch.log_softmax(t_y[:n], dim=-1)
        t_token_ids = teacher_labels_list[i][:n]
        log_p_teacher = teacher_log_p.gather(1, t_token_ids.unsqueeze(1)).squeeze(1).detach()

        # Frozen init log prob (precomputed)
        if idx < len(init_logprobs) and init_logprobs[idx].numel() > 0:
            log_p_init = init_logprobs[idx].to(s_y.device)
            # Clamp to shortest of all three: student, teacher, init
            n = min(n, log_p_init.shape[0])
            log_p_init = log_p_init[:n]
            log_p_teacher = log_p_teacher[:n]
        else:
            log_p_init = torch.zeros(n, device=s_y.device)

        # Zone gate weights
        delta = log_p_teacher - log_p_init
        w = (delta.clamp(0, eps_hi) / eps_hi).detach()

        per_token = jsd_loss_per_token(s_y[:n], t_y[:n], beta=beta)

        if w.sum() > 0:
            losses.append((w * per_token).sum() / w.sum())
            all_weights.append(w.mean().item())

    if not losses:
        return torch.tensor(0.0, device="cuda", requires_grad=True) * 0.0, 0.0

    mean_weight = sum(all_weights) / len(all_weights) if all_weights else 0.0
    return torch.stack(losses).mean(), mean_weight


def jsd_is1_loss(s_list, t_list, sample_indices, init_logprobs, beta=0.5, clip_c=5.0):
    """JSD + sequence-level IS: w_i = clip(p_θ0(y*_i) / mean_p, 0, c)."""
    # Get sequence log probs for this batch
    seq_lps = []
    for idx in sample_indices:
        if idx < len(init_logprobs) and init_logprobs[idx].numel() > 0:
            seq_lps.append(init_logprobs[idx].sum().item())
        else:
            seq_lps.append(-1e6)

    seq_lps = torch.tensor(seq_lps, device=s_list[0].device)
    # Normalize by batch mean (in log space: subtract log_mean)
    log_mean = torch.logsumexp(seq_lps, dim=0) - torch.log(torch.tensor(float(len(seq_lps))))
    log_weights = seq_lps - log_mean
    weights = torch.exp(log_weights).clamp(0, clip_c)

    losses = []
    for s_y, t_y in zip(s_list, t_list):
        n = min(s_y.shape[0], t_y.shape[0])
        if n == 0: continue
        losses.append(jsd_loss(s_y[:n], t_y[:n], beta=beta))

    if not losses:
        return torch.tensor(0.0, device="cuda", requires_grad=True) * 0.0

    loss_stack = torch.stack(losses)
    weights = weights[:len(loss_stack)]
    return (weights * loss_stack).mean()


def jsd_is2_loss(s_list, t_list, sample_indices, init_logprobs, beta=0.5):
    """JSD + token-level weight: w_t = min(1, 1/(-log p_θ0(y*_t))).
    Downweights tokens the student finds very surprising (hopeless to learn).
    Easy tokens (high p) get full gradient, hard tokens get reduced gradient."""
    losses = []
    for i, (s_y, t_y) in enumerate(zip(s_list, t_list)):
        n = min(s_y.shape[0], t_y.shape[0])
        if n == 0: continue

        idx = sample_indices[i]
        if idx < len(init_logprobs) and init_logprobs[idx].numel() > 0:
            tok_lps = init_logprobs[idx].to(s_y.device)
            n = min(n, tok_lps.shape[0])  # clamp to shortest
            token_nll = (-tok_lps[:n]).clamp(min=1e-6)  # avoid div by zero
            weights = (1.0 / token_nll).clamp(max=1.0)
        else:
            weights = torch.ones(n, device=s_y.device)

        per_token = jsd_loss_per_token(s_y[:n], t_y[:n], beta=beta)
        losses.append((weights * per_token).mean())

    if not losses:
        return torch.tensor(0.0, device="cuda", requires_grad=True) * 0.0
    return torch.stack(losses).mean()


def jsd_is3_loss(s_list, t_list, sample_indices, init_logprobs, beta=0.5, tau=0.5):
    """JSD + token masking: zero grad on tokens where p_θ0 > τ."""
    losses = []
    mask_stats = {"total": 0, "masked": 0}
    for i, (s_y, t_y) in enumerate(zip(s_list, t_list)):
        n = min(s_y.shape[0], t_y.shape[0])
        if n == 0: continue

        idx = sample_indices[i]
        if idx < len(init_logprobs) and init_logprobs[idx].numel() > 0:
            tok_lps = init_logprobs[idx].to(s_y.device)
            n = min(n, tok_lps.shape[0])  # clamp to shortest
            tok_probs = torch.exp(tok_lps[:n])
            mask = (tok_probs < tau).float()  # 1 where surprising, 0 where easy
        else:
            mask = torch.ones(n, device=s_y.device)

        mask_stats["total"] += n
        mask_stats["masked"] += int((mask == 0).sum().item())

        per_token = jsd_loss_per_token(s_y[:n], t_y[:n], beta=beta)
        if mask.sum() > 0:
            losses.append((mask * per_token).sum() / mask.sum())

    if not losses:
        return torch.tensor(0.0, device="cuda", requires_grad=True) * 0.0
    return torch.stack(losses).mean()


def jsd_is4_loss(s_list, t_list, sample_indices, init_logprobs, beta=0.5):
    """JSD + length-normalized surprisal: w_i = mean_t(-log p_θ0(y*_t))."""
    seq_weights = []
    for idx in sample_indices:
        if idx < len(init_logprobs) and init_logprobs[idx].numel() > 0:
            lps = init_logprobs[idx]
            mean_surprisal = (-lps).mean().item()
            seq_weights.append(mean_surprisal)
        else:
            seq_weights.append(1.0)

    seq_weights = torch.tensor(seq_weights, device=s_list[0].device)
    # Normalize weights to mean 1
    seq_weights = seq_weights / (seq_weights.mean() + 1e-8)

    losses = []
    for s_y, t_y in zip(s_list, t_list):
        n = min(s_y.shape[0], t_y.shape[0])
        if n == 0: continue
        losses.append(jsd_loss(s_y[:n], t_y[:n], beta=beta))

    if not losses:
        return torch.tensor(0.0, device="cuda", requires_grad=True) * 0.0

    loss_stack = torch.stack(losses)
    weights = seq_weights[:len(loss_stack)]
    return (weights * loss_stack).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Teacher forward
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_teacher_ystar(model, teacher_input_ids, teacher_attention_mask, teacher_labels):
    model.disable_adapter_layers()
    t_logits = model(input_ids=teacher_input_ids, attention_mask=teacher_attention_mask).logits
    model.enable_adapter_layers()
    t_ystar = extract_ystar_logits(t_logits, teacher_labels)
    del t_logits
    return t_ystar


# ─────────────────────────────────────────────────────────────────────────────
# Rollout infrastructure (for RKL, RLAD, DistiLLM-2 SRKL)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_rollouts(model, tokenizer, batch, device, max_new_tokens=1024, temperature=0.7,
                      server_url=None):
    """
    Generate y_hat ~ p(·|x).

    If server_url is set: use external vLLM server (fast, ~10x faster).
      - Samples from base model on the server (not current LoRA).
      - Fine for RKL/RLAD — generated text is just input to the loss.
    If server_url is None: use model.generate() on training GPU (slow but exact policy).
    """
    if server_url:
        return generate_rollouts_vllm(tokenizer, batch, server_url, max_new_tokens, temperature, device)
    else:
        return generate_rollouts_local(model, tokenizer, batch, device, max_new_tokens, temperature)


def generate_rollouts_vllm(tokenizer, batch, server_url, max_new_tokens, temperature, device):
    """Generate via external vLLM server. Fast, async, batched."""
    import requests
    from concurrent.futures import ThreadPoolExecutor, as_completed

    student_ids = batch["student_input_ids"]
    student_labels = batch["student_labels"]

    # Extract prompt text for each sample
    prompts = []
    for i in range(student_ids.shape[0]):
        nonpad = (student_labels[i] != -100).nonzero(as_tuple=True)[0]
        plen = nonpad[0].item() if len(nonpad) > 0 else student_ids.shape[1]
        prompt_ids = student_ids[i, :plen].tolist()
        # Remove padding
        prompt_ids = [t for t in prompt_ids if t != tokenizer.pad_token_id]
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        prompts.append(prompt_text)

    # Auto-detect model name from server
    try:
        resp = requests.get(f"{server_url}/v1/models", timeout=5)
        model_name = resp.json()["data"][0]["id"]
    except Exception:
        model_name = "Qwen/Qwen3-8B"

    def _generate_one(prompt_text):
        payload = {
            "model": model_name,
            "prompt": prompt_text,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        for attempt in range(3):
            try:
                resp = requests.post(f"{server_url}/v1/completions", json=payload, timeout=300)
                resp.raise_for_status()
                text = resp.json()["choices"][0]["text"]
                return text
            except Exception as e:
                if attempt == 2:
                    return ""
        return ""

    # Generate all prompts concurrently
    y_hat_texts = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=min(len(prompts), 32)) as ex:
        futures = {ex.submit(_generate_one, p): i for i, p in enumerate(prompts)}
        for f in as_completed(futures):
            idx = futures[f]
            y_hat_texts[idx] = f.result()

    # Tokenize back to IDs
    y_hat_ids = []
    for text in y_hat_texts:
        ids = tokenizer.encode(text or "", add_special_tokens=False)
        y_hat_ids.append(torch.tensor(ids, device=device))

    return y_hat_ids, y_hat_texts


def generate_rollouts_local(model, tokenizer, batch, device, max_new_tokens, temperature):
    """Generate using model.generate() on the training GPU."""
    model.eval()

    student_ids = batch["student_input_ids"].to(device)
    student_mask = batch["student_attention_mask"].to(device)
    student_labels = batch["student_labels"]

    prompt_lengths = []
    for i in range(student_labels.shape[0]):
        nonpad = (student_labels[i] != -100).nonzero(as_tuple=True)[0]
        if len(nonpad) > 0:
            prompt_lengths.append(nonpad[0].item())
        else:
            prompt_lengths.append(student_ids.shape[1])

    max_prompt_len = max(prompt_lengths)

    gen_ids_list = []
    gen_mask_list = []
    for i in range(student_ids.shape[0]):
        plen = prompt_lengths[i]
        prompt = student_ids[i, :plen]
        mask = student_mask[i, :plen]
        pad_len = max_prompt_len - plen
        if pad_len > 0:
            prompt = torch.cat([torch.full((pad_len,), tokenizer.pad_token_id, device=device), prompt])
            mask = torch.cat([torch.zeros(pad_len, dtype=torch.long, device=device), mask])
        gen_ids_list.append(prompt)
        gen_mask_list.append(mask)

    gen_ids = torch.stack(gen_ids_list)
    gen_mask = torch.stack(gen_mask_list)

    outputs = model.generate(
        input_ids=gen_ids,
        attention_mask=gen_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    y_hat_ids = []
    y_hat_texts = []
    for i in range(outputs.shape[0]):
        generated = outputs[i, max_prompt_len:]
        eos_positions = (generated == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            generated = generated[:eos_positions[0] + 1]
        y_hat_ids.append(generated)
        y_hat_texts.append(tokenizer.decode(generated, skip_special_tokens=True))

    model.train()
    return y_hat_ids, y_hat_texts


def build_teacher_rollout_inputs(batch, y_hat_ids, tokenizer, device, max_length=4096):
    """
    Build teacher inputs for scoring y_hat: [x + prefix(y) + o + y_hat]

    Uses the teacher prompt from the batch (already has x + prefix(y) + o),
    and appends y_hat as the completion to score.

    Returns:
        input_ids: (B, max_len) padded
        attention_mask: (B, max_len)
        completion_mask: (B, max_len) — 1 where y_hat tokens are, 0 elsewhere
        completion_ids: (B, max_len) — token IDs at completion positions, -100 elsewhere
    """
    teacher_ids = batch["teacher_input_ids"]
    teacher_labels = batch["teacher_labels"]

    # Find teacher prompt length (before y* tokens start)
    all_input_ids = []
    all_attn_masks = []
    all_completion_masks = []
    all_completion_ids = []

    for i in range(teacher_ids.shape[0]):
        # Find where y* starts in teacher sequence (first non-(-100) label)
        nonpad = (teacher_labels[i] != -100).nonzero(as_tuple=True)[0]
        if len(nonpad) > 0:
            t_prompt_len = nonpad[0].item()
        else:
            t_prompt_len = teacher_ids.shape[1]

        # Teacher prompt (x + prefix(y) + o)
        t_prompt = teacher_ids[i, :t_prompt_len]
        # Remove any padding from end of prompt
        real_len = (t_prompt != tokenizer.pad_token_id).sum().item()
        t_prompt = t_prompt[:real_len]

        # Append y_hat
        yhat = y_hat_ids[i]
        full_ids = torch.cat([t_prompt.to(device), yhat.to(device)])

        # Truncate if too long
        if len(full_ids) > max_length:
            full_ids = full_ids[:max_length]

        prompt_len = len(t_prompt)
        seq_len = len(full_ids)
        yhat_len = seq_len - prompt_len

        attn_mask = torch.ones(seq_len, dtype=torch.long, device=device)

        # Completion mask: 1 at y_hat positions
        comp_mask = torch.zeros(seq_len, dtype=torch.long, device=device)
        comp_mask[prompt_len:] = 1

        # Completion IDs for loss: -100 at prompt, actual IDs at y_hat
        comp_ids = torch.full((seq_len,), -100, dtype=torch.long, device=device)
        comp_ids[prompt_len:] = full_ids[prompt_len:]

        all_input_ids.append(full_ids)
        all_attn_masks.append(attn_mask)
        all_completion_masks.append(comp_mask)
        all_completion_ids.append(comp_ids)

    # Right-pad to max length in batch
    max_len = max(len(ids) for ids in all_input_ids)

    def pad_to(tensor, length, val=0):
        if len(tensor) >= length:
            return tensor[:length]
        return torch.cat([tensor, torch.full((length - len(tensor),), val, dtype=tensor.dtype, device=tensor.device)])

    input_ids = torch.stack([pad_to(ids, max_len, tokenizer.pad_token_id) for ids in all_input_ids])
    attn_masks = torch.stack([pad_to(m, max_len, 0) for m in all_attn_masks])
    comp_masks = torch.stack([pad_to(m, max_len, 0) for m in all_completion_masks])
    comp_ids = torch.stack([pad_to(ids, max_len, -100) for ids in all_completion_ids])

    return input_ids, attn_masks, comp_masks, comp_ids


def build_student_rollout_inputs(batch, y_hat_ids, tokenizer, device, max_length=2048):
    """
    Build student inputs for scoring y_hat: [x + y_hat]
    Same as teacher but with student prompt (x only).
    """
    student_ids = batch["student_input_ids"]
    student_labels = batch["student_labels"]

    all_input_ids = []
    all_attn_masks = []
    all_completion_ids = []

    for i in range(student_ids.shape[0]):
        nonpad = (student_labels[i] != -100).nonzero(as_tuple=True)[0]
        if len(nonpad) > 0:
            s_prompt_len = nonpad[0].item()
        else:
            s_prompt_len = student_ids.shape[1]

        s_prompt = student_ids[i, :s_prompt_len]
        real_len = (s_prompt != tokenizer.pad_token_id).sum().item()
        s_prompt = s_prompt[:real_len]

        yhat = y_hat_ids[i]
        full_ids = torch.cat([s_prompt.to(device), yhat.to(device)])

        if len(full_ids) > max_length:
            full_ids = full_ids[:max_length]

        prompt_len = len(s_prompt)
        seq_len = len(full_ids)

        attn_mask = torch.ones(seq_len, dtype=torch.long, device=device)
        comp_ids = torch.full((seq_len,), -100, dtype=torch.long, device=device)
        comp_ids[prompt_len:] = full_ids[prompt_len:]

        all_input_ids.append(full_ids)
        all_attn_masks.append(attn_mask)
        all_completion_ids.append(comp_ids)

    max_len = max(len(ids) for ids in all_input_ids)

    def pad_to(tensor, length, val=0):
        if len(tensor) >= length:
            return tensor[:length]
        return torch.cat([tensor, torch.full((length - len(tensor),), val, dtype=tensor.dtype, device=tensor.device)])

    input_ids = torch.stack([pad_to(ids, max_len, tokenizer.pad_token_id) for ids in all_input_ids])
    attn_masks = torch.stack([pad_to(m, max_len, 0) for m in all_attn_masks])
    comp_ids = torch.stack([pad_to(ids, max_len, -100) for ids in all_completion_ids])

    return input_ids, attn_masks, comp_ids


def extract_completion_logprobs(logits, completion_ids):
    """Extract per-token log probs at completion positions."""
    shift_logits = logits[:, :-1, :]
    shift_labels = completion_ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)

    B = logits.shape[0]
    results = []
    for i in range(B):
        mask = shift_labels[i] != -100
        if mask.sum() == 0:
            results.append(torch.zeros(1, device=logits.device))
            continue
        tids = shift_labels[i][mask]
        lps = log_probs[i][mask].gather(1, tids.unsqueeze(1)).squeeze(1)
        results.append(lps)
    return results


def extract_completion_logits(logits, completion_ids):
    """Extract full logit vectors at completion positions (for SRKL)."""
    shift_logits = logits[:, :-1, :]
    shift_labels = completion_ids[:, 1:]

    results = []
    for i in range(logits.shape[0]):
        mask = shift_labels[i] != -100
        if mask.sum() == 0:
            results.append(shift_logits[i, :1])  # dummy
            continue
        results.append(shift_logits[i][mask])
    return results


def rkl_loss(student_logprobs_list, teacher_logprobs_list):
    """
    RKL with REINFORCE: L = E_{y~p_θ}[Σ_t log(p_θ(y_t)/p_T(y_t))]

    student_logprobs have gradients, teacher_logprobs are detached.
    Uses per-token advantage: a_t = (log p_T - log p_θ_old).detach()
    Policy gradient: loss = -Σ_t log p_θ(y_t) * a_t
    """
    losses = []
    mean_lengths = []
    for s_lps, t_lps in zip(student_logprobs_list, teacher_logprobs_list):
        n = min(len(s_lps), len(t_lps))
        if n == 0:
            continue
        s = s_lps[:n]
        t = t_lps[:n].detach()

        # Advantage: how much better teacher thinks each token is
        advantage = t - s.detach()  # positive = teacher prefers this token

        # Policy gradient loss (maximize tokens teacher likes)
        pg_loss = -(s * advantage).mean()
        losses.append(pg_loss)
        mean_lengths.append(n)

    if not losses:
        return torch.tensor(0.0, device="cuda", requires_grad=True) * 0.0, 0.0

    mean_len = sum(mean_lengths) / len(mean_lengths)
    return torch.stack(losses).mean(), mean_len


def rlad_loss(student_logprobs_list, teacher_logprobs_list, epsilon=0.2):
    """
    RLAD: RKL with per-token clipped trust region.
    r_t = p_θ(y_t) / p_T(y_t), clip(r_t, 1-ε, 1+ε)
    """
    losses = []
    mean_lengths = []
    for s_lps, t_lps in zip(student_logprobs_list, teacher_logprobs_list):
        n = min(len(s_lps), len(t_lps))
        if n == 0:
            continue
        s = s_lps[:n]
        t = t_lps[:n].detach()

        # Log ratio and clip
        log_ratio = s - t
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)

        # Advantage
        advantage = t - s.detach()

        # Clipped surrogate (PPO-style)
        surr1 = ratio * advantage
        surr2 = clipped_ratio * advantage
        pg_loss = -torch.min(surr1, surr2).mean()

        losses.append(pg_loss)
        mean_lengths.append(n)

    if not losses:
        return torch.tensor(0.0, device="cuda", requires_grad=True) * 0.0, 0.0

    mean_len = sum(mean_lengths) / len(mean_lengths)
    return torch.stack(losses).mean(), mean_len


def srkl_loss(student_logits_list, teacher_logits_list, alpha=0.1):
    """
    SRKL for DistiLLM-2: KL(p_θ || (1-α)p_T + α·p_θ) on rollout y_hat.
    Pushes student away from its own outputs toward teacher.
    """
    losses = []
    for s_logits, t_logits in zip(student_logits_list, teacher_logits_list):
        n = min(s_logits.shape[0], t_logits.shape[0])
        if n == 0:
            continue
        p_S = torch.softmax(s_logits[:n], dim=-1)
        p_T = torch.softmax(t_logits[:n], dim=-1).detach()
        m = ((1 - alpha) * p_T + alpha * p_S).clamp(min=1e-10)
        kl = (p_S * (p_S.clamp(min=1e-10).log() - m.log())).sum(dim=-1)
        losses.append(kl.mean())

    if not losses:
        return torch.tensor(0.0, device="cuda", requires_grad=True) * 0.0
    return torch.stack(losses).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(model, tokenizer, dataset, args, init_logprobs=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    is_dpo = args.objective == "dpo"
    is_is_method = args.objective.startswith("jsd_is") or args.objective == "zg_jsd"
    needs_teacher = args.objective in ("fkl", "jsd", "jsd_is1", "jsd_is2", "jsd_is3", "jsd_is4", "zg_jsd", "distillm2")
    is_rollout = args.objective in ("rkl", "rlad", "distillm2")

    # Resolve rollout server URL
    rollout_server_url = None
    if is_rollout:
        if args.rollout_server_url:
            rollout_server_url = args.rollout_server_url
        elif args.rollout_server_url_file:
            import time as _time
            print(f"Waiting for server URL file: {args.rollout_server_url_file}")
            while not os.path.exists(args.rollout_server_url_file):
                _time.sleep(10)
            with open(args.rollout_server_url_file) as f:
                rollout_server_url = f.read().strip()
        if rollout_server_url:
            print(f"Using vLLM server for generation: {rollout_server_url}")
        else:
            print("No rollout server — generating locally on training GPU (slow)")

    if is_is_method and init_logprobs is None:
        raise ValueError(f"--logprobs_file required for {args.objective}")

    # Wrap dataset for IS methods to pass sample indices
    if is_is_method:
        dataset = IndexedDataset(dataset)

    collate_fn = (lambda b: collate_dpo(b, tokenizer.pad_token_id)) if is_dpo \
                 else (lambda b: collate(b, tokenizer.pad_token_id))

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_fn, num_workers=4, pin_memory=True,
                        persistent_workers=True)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=args.lr, weight_decay=0.01)
    total_steps = (len(loader) * args.epochs) // args.grad_accum
    warmup_steps = int(0.05 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    save_every = args.save_every if args.save_every > 0 else max(1, total_steps // args.num_ckpts)

    print(f"Steps/epoch     : {len(loader)}")
    print(f"Total opt steps : {total_steps}")
    print(f"Effective batch : {args.batch_size * args.grad_accum}")
    print(f"Saving every    : {save_every} steps")

    if HAS_WANDB and not args.no_wandb:
        wandb.init(project=args.wandb_project, name=args.run_name or Path(args.output_dir).name,
                   config=vars(args))

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for step, batch in enumerate(pbar):

            # ── DPO ──────────────────────────────────────────────────────
            if is_dpo:
                c_ids = batch["chosen_input_ids"].to(device)
                c_mask = batch["chosen_attention_mask"].to(device)
                c_lbl = batch["chosen_labels"].to(device)
                r_ids = batch["rejected_input_ids"].to(device)
                r_mask = batch["rejected_attention_mask"].to(device)
                r_lbl = batch["rejected_labels"].to(device)

                # Ref logprobs first (no grad, LoRA disabled) — compute and detach immediately
                with torch.no_grad():
                    model.disable_adapter_layers()
                    ref_c_logits = model(input_ids=c_ids, attention_mask=c_mask).logits
                    ref_c_lps = sequence_logprob(ref_c_logits, c_lbl).detach()
                    del ref_c_logits
                    ref_r_logits = model(input_ids=r_ids, attention_mask=r_mask).logits
                    ref_r_lps = sequence_logprob(ref_r_logits, r_lbl).detach()
                    del ref_r_logits
                    model.enable_adapter_layers()
                torch.cuda.empty_cache()

                # Policy logprobs (LoRA enabled, with grad) — one at a time
                policy_c_logits = model(input_ids=c_ids, attention_mask=c_mask).logits
                policy_c_lps = sequence_logprob(policy_c_logits, c_lbl)
                del policy_c_logits
                policy_r_logits = model(input_ids=r_ids, attention_mask=r_mask).logits
                policy_r_lps = sequence_logprob(policy_r_logits, r_lbl)
                del policy_r_logits

                loss = dpo_loss(policy_c_lps, policy_r_lps,
                                ref_c_lps, ref_r_lps, beta=args.dpo_beta)

                log_dict = {"loss": loss.item(), "dpo_chosen_reward": (policy_c_lps - ref_c_lps).mean().item(),
                            "dpo_rejected_reward": (policy_r_lps - ref_r_lps).mean().item()}

            # ── RKL / RLAD (offline, SDPO-style on fixed y*) ────────────────
            # Same as Eric's offline SDPO:
            #   teacher: log p(y*|x, prefix(y), o) — privileged context, no grad
            #   student: log p(y*|x) — test-time context, with grad
            #   signal: (teacher_logp - student_logp).detach()
            #   loss: -signal * student_logp (per-token, length-normalized)
            elif args.objective in ("rkl", "rlad"):
                s_ids = batch["student_input_ids"].to(device)
                s_mask = batch["student_attention_mask"].to(device)
                s_lbl = batch["student_labels"].to(device)
                t_ids = batch["teacher_input_ids"].to(device)
                t_mask = batch["teacher_attention_mask"].to(device)
                t_lbl = batch["teacher_labels"].to(device)

                # Teacher logprobs on y* (no grad, LoRA disabled)
                with torch.no_grad():
                    model.disable_adapter_layers()
                    t_logits = model(input_ids=t_ids, attention_mask=t_mask).logits
                    model.enable_adapter_layers()
                    # Extract per-token logprobs at y* positions
                    t_shift = t_logits[:, :-1, :]
                    t_labels_shift = t_lbl[:, 1:]
                    t_log_probs = F.log_softmax(t_shift, dim=-1)
                    del t_logits, t_shift
                torch.cuda.empty_cache()

                # Student logprobs on y* (with grad)
                s_logits = model(input_ids=s_ids, attention_mask=s_mask).logits
                s_shift = s_logits[:, :-1, :]
                s_labels_shift = s_lbl[:, 1:]
                s_log_probs = F.log_softmax(s_shift, dim=-1)
                del s_logits, s_shift

                # Compute per-token loss
                B = s_ids.shape[0]
                batch_losses = []
                for i in range(B):
                    s_mask_i = s_labels_shift[i] != -100
                    t_mask_i = t_labels_shift[i] != -100
                    n = min(s_mask_i.sum().item(), t_mask_i.sum().item())
                    if n == 0:
                        continue

                    s_pos = s_mask_i.nonzero(as_tuple=True)[0][:n]
                    t_pos = t_mask_i.nonzero(as_tuple=True)[0][:n]
                    s_tids = s_labels_shift[i, s_pos]

                    # Per-token logprobs
                    logp_x = s_log_probs[i, s_pos].gather(1, s_tids.unsqueeze(1)).squeeze(1)
                    logp_xo = t_log_probs[i, t_pos].gather(1, s_tids.unsqueeze(1)).squeeze(1).detach()

                    # Signal: how much better teacher is
                    signal = (logp_xo - logp_x.detach())

                    if args.objective == "rlad":
                        # Clip signal to trust region
                        signal = signal.clamp(-args.rlad_epsilon, args.rlad_epsilon)

                    # Policy gradient loss (length-normalized)
                    per_token_loss = -(signal * logp_x)
                    batch_losses.append(per_token_loss.sum() / n)

                loss = torch.stack(batch_losses).mean() if batch_losses else torch.tensor(0.0, device=device, requires_grad=True)

                with torch.no_grad():
                    mean_signal = signal.mean().item() if batch_losses else 0.0
                log_dict = {"loss": loss.item(), "mean_signal": mean_signal}

            # ── DistiLLM-2 (SKL part only, SRKL stub) ───────────────────
            elif args.objective == "distillm2":
                s_ids = batch["student_input_ids"].to(device)
                s_mask = batch["student_attention_mask"].to(device)
                s_lbl = batch["student_labels"].to(device)
                t_ids = batch["teacher_input_ids"].to(device)
                t_mask = batch["teacher_attention_mask"].to(device)
                t_lbl = batch["teacher_labels"].to(device)

                t_ystar = get_teacher_ystar(model, t_ids, t_mask, t_lbl)
                s_logits = model(input_ids=s_ids, attention_mask=s_mask).logits
                s_ystar = extract_ystar_logits(s_logits, s_lbl)

                # SKL on y* (implemented)
                skl_losses = []
                for s_y, t_y in zip(s_ystar, t_ystar):
                    n = min(s_y.shape[0], t_y.shape[0])
                    if n > 0:
                        skl_losses.append(skl_loss(s_y[:n], t_y[:n], alpha=args.skl_alpha))
                skl_l = torch.stack(skl_losses).mean() if skl_losses else torch.tensor(0.0, device=device)

                # SRKL on y* (offline): KL(p_student || (1-α)p_teacher + α·p_student)
                # Uses the same y* logits already computed above
                srkl_l = srkl_loss(s_ystar, t_ystar, alpha=args.skl_alpha)

                loss = skl_l + args.distillm2_lambda * srkl_l
                log_dict = {"loss": loss.item(), "skl_loss": skl_l.item(), "srkl_loss": srkl_l.item()}

            # ── SFT / FKL / JSD / JSD+IS ────────────────────────────────
            else:
                s_ids = batch["student_input_ids"].to(device)
                s_mask = batch["student_attention_mask"].to(device)
                s_lbl = batch["student_labels"].to(device)

                if args.objective == "sft":
                    logits = model(input_ids=s_ids, attention_mask=s_mask).logits
                    loss = sft_loss(logits, s_lbl)
                    log_dict = {"loss": loss.item()}

                else:
                    t_ids = batch["teacher_input_ids"].to(device)
                    t_mask = batch["teacher_attention_mask"].to(device)
                    t_lbl = batch["teacher_labels"].to(device)

                    t_ystar = get_teacher_ystar(model, t_ids, t_mask, t_lbl)
                    s_logits = model(input_ids=s_ids, attention_mask=s_mask).logits
                    s_ystar = extract_ystar_logits(s_logits, s_lbl)

                    sample_indices = batch.get("sample_idx", None)
                    if sample_indices is not None:
                        sample_indices = sample_indices.tolist()

                    if args.objective == "fkl":
                        kl_l = kl_from_ystar_lists(s_ystar, t_ystar, "fkl")
                    elif args.objective == "jsd":
                        kl_l = kl_from_ystar_lists(s_ystar, t_ystar, "jsd", beta=args.beta)
                    elif args.objective == "jsd_is1":
                        kl_l = jsd_is1_loss(s_ystar, t_ystar, sample_indices, init_logprobs,
                                            beta=args.beta, clip_c=args.is_clip)
                    elif args.objective == "jsd_is2":
                        kl_l = jsd_is2_loss(s_ystar, t_ystar, sample_indices, init_logprobs, beta=args.beta)
                    elif args.objective == "jsd_is3":
                        kl_l = jsd_is3_loss(s_ystar, t_ystar, sample_indices, init_logprobs,
                                            beta=args.beta, tau=args.is_tau)
                    elif args.objective == "jsd_is4":
                        kl_l = jsd_is4_loss(s_ystar, t_ystar, sample_indices, init_logprobs, beta=args.beta)
                    elif args.objective == "zg_jsd":
                        t_token_ids = extract_ystar_token_ids(t_lbl)
                        kl_l, mean_w = zg_jsd_loss(
                            s_ystar, t_ystar, sample_indices, init_logprobs,
                            t_token_ids, beta=args.beta, eps_hi=args.zg_eps_hi)
                    else:
                        raise ValueError(f"Unknown objective: {args.objective}")

                    if args.pure_kl:
                        loss = kl_l
                    else:
                        sft_l = sft_loss(s_logits, s_lbl)
                        loss = sft_l + args.kl_alpha * kl_l

                    log_dict = {"loss": loss.item(), "kl_loss": kl_l.item()}
                    if args.objective == "zg_jsd":
                        log_dict["mean_gate_weight"] = mean_w
                    if not args.pure_kl:
                        log_dict["sft_loss"] = sft_l.item()

            # ── Backward + step ──────────────────────────────────────────
            (loss / args.grad_accum).backward()

            if (step + 1) % args.grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                lr = scheduler.get_last_lr()[0]
                log_dict.update({"lr": lr, "grad_norm": grad_norm.item(), "global_step": global_step})

                display = {}
                for k, v in log_dict.items():
                    if k not in ("loss", "kl_loss", "lr", "grad_norm", "mean_gate_weight", "mean_gen_len"):
                        continue
                    if k == "lr":
                        display[k] = f"{v:.1e}"
                    elif isinstance(v, float):
                        display[k] = f"{v:.6f}"
                    else:
                        display[k] = v
                pbar.set_postfix(display)

                if global_step % save_every == 0:
                    _save(model, tokenizer, os.path.join(args.output_dir, f"step-{global_step}"))

                if args.max_steps > 0 and global_step >= args.max_steps:
                    print(f"Reached max_steps={args.max_steps}, stopping.")
                    _save(model, tokenizer, os.path.join(args.output_dir, "final"))
                    return

                if HAS_WANDB and not args.no_wandb:
                    wandb.log(log_dict)

    _save(model, tokenizer, os.path.join(args.output_dir, "final"))
    print("Training complete.")


def _save(model, tokenizer, path):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"[Saved] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

ALL_OBJECTIVES = ["sft", "fkl", "jsd", "dpo",
                  "jsd_is1", "jsd_is2", "jsd_is3", "jsd_is4",
                  "zg_jsd",
                  "rkl", "rlad", "distillm2"]


def main():
    parser = argparse.ArgumentParser(description="Extended LoRA training with 8+ methods")

    # Data
    parser.add_argument("--input", required=True)
    parser.add_argument("--y_star_field", default="y_star_prefix30",
                        choices=["y_star_prefix30", "y_star_full", "y_star_noprefix"])
    parser.add_argument("--output_dir", required=True)

    # Model
    parser.add_argument("--model", default="Qwen/Qwen3-8B")

    # Objective
    parser.add_argument("--objective", required=True, choices=ALL_OBJECTIVES)
    parser.add_argument("--beta", type=float, default=0.5, help="β for JSD")
    parser.add_argument("--kl_alpha", type=float, default=1.0)
    parser.add_argument("--pure_kl", action="store_true")

    # DPO
    parser.add_argument("--dpo_beta", type=float, default=0.1)

    # IS methods
    parser.add_argument("--logprobs_file", default=None, help=".pt file with precomputed init logprobs")
    parser.add_argument("--is_clip", type=float, default=5.0, help="IS-1 clip threshold")
    parser.add_argument("--is_tau", type=float, default=0.3, help="IS-3 masking threshold (data-driven: 18% active at 0.3)")
    parser.add_argument("--zg_eps_hi", type=float, default=4.0, help="ZG-JSD upper clip for delta")

    # Rollout methods (RKL, RLAD, DistiLLM-2)
    parser.add_argument("--rollout_max_tokens", type=int, default=1024, help="Max tokens to generate per rollout")
    parser.add_argument("--rollout_temperature", type=float, default=0.7, help="Temperature for rollout generation")
    parser.add_argument("--rollout_server_url", default=None,
                        help="vLLM server URL for fast generation (e.g. http://gh140:8001). "
                             "If not set, generates locally on training GPU (slow).")
    parser.add_argument("--rollout_server_url_file", default=None,
                        help="File containing server URL (e.g. tmp/qwen3_8b_server_url.txt)")
    parser.add_argument("--rlad_epsilon", type=float, default=0.2, help="RLAD clipping epsilon")

    # DistiLLM-2
    parser.add_argument("--skl_alpha", type=float, default=0.1)
    parser.add_argument("--distillm2_lambda", type=float, default=1.0)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target", type=str, default="all-linear")

    # Training
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--teacher_max_length", type=int, default=None)
    parser.add_argument("--num_ckpts", type=int, default=8)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=0)

    # WandB
    parser.add_argument("--wandb_project", default="distillation-methods")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--no_wandb", action="store_true")

    args = parser.parse_args()

    if not HAS_PEFT:
        raise ImportError("pip install peft")

    print(f"{'='*60}")
    print(f"Objective    : {args.objective}")
    print(f"y* field     : {args.y_star_field}")
    print(f"Model        : {args.model}")
    print(f"LoRA r/alpha : {args.lora_r}/{args.lora_alpha}")
    print(f"Save every   : {args.save_every} steps")
    print(f"Output       : {args.output_dir}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    data = load_jsonl(args.input)

    # Build dataset
    if args.objective == "dpo":
        dataset = DPODataset(data, tokenizer, args.max_length, args.y_star_field)
    else:
        dataset = PrefixDataset(data, tokenizer, args.max_length, args.y_star_field,
                                teacher_max_length=args.teacher_max_length)

    # Load init logprobs for IS methods
    init_logprobs = None
    if args.logprobs_file:
        init_logprobs = load_init_logprobs(args.logprobs_file)

    # Load model
    try:
        import flash_attn
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"
        print("[Info] flash_attn not found, using sdpa")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl, trust_remote_code=True,
    )

    target_modules = "all-linear" if args.lora_target == "all-linear" \
                     else [m.strip() for m in args.lora_target.split(",")]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules, bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train(model, tokenizer, dataset, args, init_logprobs=init_logprobs)


if __name__ == "__main__":
    main()
