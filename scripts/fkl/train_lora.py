"""
LoRA training for offline distillation from y* targets.

Supports three training objectives:
  sft  — standard cross-entropy on y* (one-hot targets)
  fkl  — Forward KL: KL(p_teacher || p_student), full-vocab distillation
  jsd  — β-JSD (OPSD paper Eq. 7): β·KL(p_T||m) + (1-β)·KL(p_S||m)

For fkl and jsd, teacher and student see DIFFERENT inputs:
  Teacher (base model, LoRA disabled): x + prefix(y) + o + y*  ← privileged oracle context
  Student (base + LoRA):               x + y*                  ← test-time context only

This makes p_T ≠ p_S from step 0 — no SFT bootstrap required when using --pure_kl.
KL is computed over the full vocabulary (not top-k).

Dataset: JSONL with fields x (conversation), y (original response), o (oracle feedback),
and a y* field selected via --y_star_field (e.g. y_star_prefix30 or y_star_full).

Usage:
  python scripts/fkl/train_lora.py \
      --input        datasets/wildfeedback/ystar_prefix_qwen3_8b.jsonl \
      --y_star_field y_star_full \
      --objective    fkl \
      --pure_kl \
      --model        Qwen/Qwen3-8B \
      --output_dir   checkpoints/qwen3_8b_fkl_wfbest
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
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PrefixDataset(Dataset):
    """
    Builds TWO tokenized sequences per sample:

      student_input_ids : [x + y*]                — test-time context (no oracle)
      teacher_input_ids : [x + prefix(y) + o + y*] — privileged oracle context

    prefix(y) depends on condition:
      prefix30  → first 30 tokens of y (decoded back to text)
      full      → full y
      noprefix  → nothing (teacher still gets o)

    Labels: -100 on prompt tokens, actual ids on y* tokens.
    """

    def __init__(
        self,
        data: list[dict],
        tokenizer,
        max_length: int,
        y_star_field: str,
        teacher_max_length: int | None = None,
    ):
        self.condition = infer_condition(y_star_field)
        self.teacher_max_length = teacher_max_length or min(max_length * 2, 8192)
        print(f"[Dataset] condition={self.condition}  student_max={max_length}  teacher_max={self.teacher_max_length}")
        self.items = self._tokenize(data, tokenizer, max_length, y_star_field)

    def _tokenize(self, data, tokenizer, max_length, y_star_field) -> list[dict]:
        items, skipped = [], 0
        for item in tqdm(data, desc="Tokenizing"):
            y_star_raw = item.get(y_star_field)
            if not y_star_raw:
                skipped += 1
                continue

            y_star_msg = {"role": "assistant", "content": get_content(y_star_raw)}
            x = item["x"]

            # ── Student: [x + y*] ─────────────────────────────────────────
            try:
                s_prompt_text = tokenizer.apply_chat_template(
                    x, tokenize=False, add_generation_prompt=True)
                s_full_text = tokenizer.apply_chat_template(
                    list(x) + [y_star_msg], tokenize=False, add_generation_prompt=False)
            except Exception:
                skipped += 1; continue

            s_prompt_len = len(tokenizer(s_prompt_text, add_special_tokens=False)["input_ids"])
            s_enc = tokenizer(s_full_text, add_special_tokens=False,
                              truncation=True, max_length=max_length)
            s_input_ids = s_enc["input_ids"]
            s_labels = [-100] * len(s_input_ids)
            if s_prompt_len < len(s_input_ids):
                s_labels[s_prompt_len:] = s_input_ids[s_prompt_len:]
            if all(l == -100 for l in s_labels):
                skipped += 1; continue

            # ── Teacher: [x + prefix(y) + o + y*] ────────────────────────
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
                t_prompt_text = tokenizer.apply_chat_template(
                    teacher_ctx, tokenize=False, add_generation_prompt=True)
                t_full_text = tokenizer.apply_chat_template(
                    teacher_ctx + [y_star_msg], tokenize=False, add_generation_prompt=False)
            except Exception:
                skipped += 1; continue

            t_prompt_len = len(tokenizer(t_prompt_text, add_special_tokens=False)["input_ids"])
            t_enc = tokenizer(t_full_text, add_special_tokens=False,
                              truncation=True, max_length=self.teacher_max_length)
            t_input_ids = t_enc["input_ids"]
            t_labels = [-100] * len(t_input_ids)
            if t_prompt_len < len(t_input_ids):
                t_labels[t_prompt_len:] = t_input_ids[t_prompt_len:]
            if all(l == -100 for l in t_labels):
                skipped += 1; continue

            items.append({
                "student_input_ids":      s_input_ids,
                "student_attention_mask": s_enc["attention_mask"],
                "student_labels":         s_labels,
                "teacher_input_ids":      t_input_ids,
                "teacher_attention_mask": t_enc["attention_mask"],
                "teacher_labels":         t_labels,
            })

        print(f"[Dataset] {len(items)} usable, {skipped} skipped")
        return items

    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]


def collate(batch: list[dict], pad_id: int) -> dict:
    def pad_field(seqs, pad_val=0):
        ml = max(len(s) for s in seqs)
        return [s + [pad_val] * (ml - len(s)) for s in seqs]

    return {
        "student_input_ids":      torch.tensor(pad_field([x["student_input_ids"]      for x in batch]),       dtype=torch.long),
        "student_attention_mask": torch.tensor(pad_field([x["student_attention_mask"]  for x in batch]),       dtype=torch.long),
        "student_labels":         torch.tensor(pad_field([x["student_labels"]          for x in batch], -100), dtype=torch.long),
        "teacher_input_ids":      torch.tensor(pad_field([x["teacher_input_ids"]       for x in batch]),       dtype=torch.long),
        "teacher_attention_mask": torch.tensor(pad_field([x["teacher_attention_mask"]  for x in batch]),       dtype=torch.long),
        "teacher_labels":         torch.tensor(pad_field([x["teacher_labels"]          for x in batch], -100), dtype=torch.long),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

def sft_loss(student_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = student_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    B, T, V = shift_logits.shape
    return F.cross_entropy(
        shift_logits.reshape(B * T, V), shift_labels.reshape(B * T),
        ignore_index=-100, reduction="mean",
    )


def fkl_loss(student_ystar: torch.Tensor, teacher_ystar: torch.Tensor) -> torch.Tensor:
    """KL(p_T || p_S) over full vocab. p_T detached."""
    log_p = torch.log_softmax(student_ystar, dim=-1)
    q     = torch.softmax(teacher_ystar, dim=-1).detach()
    return -(q * log_p).sum(dim=-1).mean()


def jsd_loss(student_ystar: torch.Tensor, teacher_ystar: torch.Tensor, beta: float = 0.5) -> torch.Tensor:
    """β-JSD (OPSD Eq. 7) over full vocab. p_T detached."""
    p_T = torch.softmax(teacher_ystar, dim=-1).detach()
    p_S = torch.softmax(student_ystar, dim=-1)
    m   = (beta * p_T + (1 - beta) * p_S).clamp(min=1e-10)
    kl_T = (p_T * (p_T.clamp(min=1e-10).log() - m.log())).sum(dim=-1)
    kl_S = (p_S * (p_S.clamp(min=1e-10).log() - m.log())).sum(dim=-1)
    return (beta * kl_T + (1 - beta) * kl_S).mean()


def extract_ystar_logits(logits: torch.Tensor, labels: torch.Tensor) -> list[torch.Tensor]:
    """Extract logits at y* positions per batch item (after causal shift)."""
    shift_logits = logits[:, :-1, :]
    shift_mask   = labels[:, 1:] != -100
    return [shift_logits[i][shift_mask[i]] for i in range(logits.shape[0])]


def kl_from_ystar_lists(
    s_list: list[torch.Tensor],
    t_list: list[torch.Tensor],
    objective: str,
    beta: float = 0.5,
) -> torch.Tensor:
    """Compute per-sample FKL or JSD from pre-extracted y* logit lists, then average."""
    losses = []
    for s_y, t_y in zip(s_list, t_list):
        n = min(s_y.shape[0], t_y.shape[0])
        if n == 0:
            continue
        if objective == "fkl":
            losses.append(fkl_loss(s_y[:n], t_y[:n]))
        else:
            losses.append(jsd_loss(s_y[:n], t_y[:n], beta=beta))
    if not losses:
        return (s_list[0].sum() if s_list else torch.tensor(0.0, device="cuda")) * 0.0
    return torch.stack(losses).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Teacher forward
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_teacher_ystar(
    model,
    teacher_input_ids: torch.Tensor,
    teacher_attention_mask: torch.Tensor,
    teacher_labels: torch.Tensor,
) -> list[torch.Tensor]:
    """
    Base model (LoRA disabled) forward, then immediately extract y* logits
    and delete the full logits tensor to free GPU memory.
    """
    model.disable_adapter_layers()
    t_logits = model(input_ids=teacher_input_ids, attention_mask=teacher_attention_mask).logits
    model.enable_adapter_layers()
    t_ystar = extract_ystar_logits(t_logits, teacher_labels)
    del t_logits
    return t_ystar


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(model, tokenizer, dataset: Dataset, args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    model.train()

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    optimizer    = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                         lr=args.lr, weight_decay=0.01)
    total_steps  = (len(loader) * args.epochs) // args.grad_accum
    warmup_steps = int(0.05 * total_steps)
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Evenly space args.num_ckpts checkpoints across training
    save_every = max(1, total_steps // args.num_ckpts)
    effective_batch = args.batch_size * args.grad_accum

    print(f"Steps/epoch     : {len(loader)}")
    print(f"Total opt steps : {total_steps}")
    print(f"Effective batch : {effective_batch}")
    print(f"Saving every    : {save_every} steps  ({args.num_ckpts} checkpoints)")

    if HAS_WANDB and not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name or Path(args.output_dir).name,
            config=vars(args),
        )

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(pbar):
            s_ids  = batch["student_input_ids"].to(device)
            s_mask = batch["student_attention_mask"].to(device)
            s_lbl  = batch["student_labels"].to(device)

            if args.objective == "sft":
                logits = model(input_ids=s_ids, attention_mask=s_mask).logits
                loss   = sft_loss(logits, s_lbl)

            else:  # fkl or jsd
                t_ids  = batch["teacher_input_ids"].to(device)
                t_mask = batch["teacher_attention_mask"].to(device)
                t_lbl  = batch["teacher_labels"].to(device)

                # Teacher forward: extract y* logits immediately, free full tensor
                t_ystar = get_teacher_ystar(model, t_ids, t_mask, t_lbl)

                # Student forward
                s_logits = model(input_ids=s_ids, attention_mask=s_mask).logits
                s_ystar  = extract_ystar_logits(s_logits, s_lbl)

                kl_l = kl_from_ystar_lists(s_ystar, t_ystar, args.objective, beta=args.beta)

                if args.pure_kl:
                    loss  = kl_l
                    sft_l = torch.tensor(0.0)
                else:
                    sft_l = sft_loss(s_logits, s_lbl)
                    loss  = sft_l + args.kl_alpha * kl_l

            (loss / args.grad_accum).backward()

            if (step + 1) % args.grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                lr      = scheduler.get_last_lr()[0]
                postfix = {"loss": f"{loss.item():.4f}", "lr": f"{lr:.1e}", "gnorm": f"{grad_norm:.2f}"}
                if args.objective in ("fkl", "jsd"):
                    postfix["kl"] = f"{kl_l.item():.4f}"
                    if not args.pure_kl:
                        postfix["sft"] = f"{sft_l.item():.4f}"
                pbar.set_postfix(postfix)

                log_dict = {"loss": loss.item(), "lr": lr,
                            "grad_norm": grad_norm.item(), "global_step": global_step}
                if args.objective in ("fkl", "jsd"):
                    log_dict["kl_loss"] = kl_l.item()
                    if not args.pure_kl:
                        log_dict["sft_loss"] = sft_l.item()

                if global_step % save_every == 0:
                    _save(model, tokenizer, os.path.join(args.output_dir, f"step-{global_step}"))

                if args.max_steps > 0 and global_step >= args.max_steps:
                    print(f"[DryRun] Reached max_steps={args.max_steps}, stopping.")
                    return

                if HAS_WANDB and not args.no_wandb:
                    wandb.log(log_dict)

    _save(model, tokenizer, os.path.join(args.output_dir, "final"))
    print("Training complete.")


def _save(model, tokenizer, path: str) -> None:
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"[Saved] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoRA training: SFT / FKL / JSD with teacher/student split inputs."
    )
    # Data
    parser.add_argument("--input",        required=True)
    parser.add_argument("--y_star_field", default="y_star_full",
                        choices=["y_star_prefix30", "y_star_full", "y_star_noprefix"])
    parser.add_argument("--output_dir",   required=True)

    # Model
    parser.add_argument("--model", default="Qwen/Qwen3-8B")

    # Objective
    parser.add_argument("--objective", default="sft", choices=["sft", "fkl", "jsd"])
    parser.add_argument("--beta",      type=float, default=0.5,
                        help="β for JSD")
    parser.add_argument("--kl_alpha",  type=float, default=1.0,
                        help="Weight on KL term when not using --pure_kl")
    parser.add_argument("--pure_kl",   action="store_true",
                        help="Skip SFT term: loss = KL only")

    # LoRA
    parser.add_argument("--lora_r",       type=int,   default=64)
    parser.add_argument("--lora_alpha",   type=int,   default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target",  type=str,   default="all-linear")

    # Training
    parser.add_argument("--batch_size",  type=int,   default=2)
    parser.add_argument("--grad_accum",  type=int,   default=64)
    parser.add_argument("--epochs",      type=int,   default=1)
    parser.add_argument("--lr",          type=float, default=2e-6)
    parser.add_argument("--max_length",  type=int,   default=2048)
    parser.add_argument("--teacher_max_length", type=int, default=None,
                        help="Max length for teacher sequences (default: min(2*max_length, 8192))")
    parser.add_argument("--num_ckpts",   type=int,   default=8,
                        help="Number of evenly-spaced checkpoints to save during training")
    parser.add_argument("--max_steps",   type=int,   default=0,
                        help="Stop after this many optimizer steps (0 = no limit, for dry runs)")

    # WandB
    parser.add_argument("--wandb_project", default="prefix-ablation-lora")
    parser.add_argument("--run_name",      default=None)
    parser.add_argument("--no_wandb",      action="store_true")

    args = parser.parse_args()

    if not HAS_PEFT:
        raise ImportError("pip install peft")

    print(f"Objective    : {args.objective}{'  (pure_kl)' if args.pure_kl else ''}")
    print(f"y* field     : {args.y_star_field}  (condition={infer_condition(args.y_star_field)})")
    print(f"Model        : {args.model}")
    print(f"LoRA r/alpha : {args.lora_r}/{args.lora_alpha}")
    print(f"Output       : {args.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    data    = load_jsonl(args.input)
    dataset = PrefixDataset(
        data, tokenizer,
        max_length=args.max_length,
        y_star_field=args.y_star_field,
        teacher_max_length=args.teacher_max_length,
    )

    try:
        import flash_attn
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"
        print("[Info] flash_attn not found, using sdpa")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16,
        attn_implementation=attn_impl, trust_remote_code=True,
    )

    target_modules = "all-linear" if args.lora_target == "all-linear" \
                     else [m.strip() for m in args.lora_target.split(",")]

    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, target_modules=target_modules, bias="none",
    ))
    model = model.to(torch.bfloat16)
    model.print_trainable_parameters()

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    if args.objective in ("fkl", "jsd"):
        model.enable_adapter_layers()

    train(model, tokenizer, dataset, args)


if __name__ == "__main__":
    main()
