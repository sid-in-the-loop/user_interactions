#!/usr/bin/env python3
"""
OLMo-3-7B FKL training — T-2 (soft FKL), T-4 (contrastive FKL), T-5 (JSD β=0.5).

Student : OLMo-3-7B-Instruct-SFT, full params, FSDP ZeRO-3 across 2 GPUs.
Teacher : same checkpoint, frozen (eval mode, no grad), one copy per GPU.
Data    : ystar_olmo_xo_C.jsonl — Prompt C teacher, "Note: o" inline injection.

Loss modes
──────────
T2  Soft FKL:
      p_T = frozen_OLMo(· | x_o, y*_{<n})   teacher sees x with "Note: o"
      p_S = student(·    | x,   y*_{<n})     student sees x only
      loss = Σ_v p_T(v) · log[p_T(v) / p_S(v)]

T4  Contrastive FKL:
      p_T   = frozen_OLMo(· | x_o, y*_{<n})
      p_ref = frozen_OLMo(· | x,   y*_{<n})  reference: teacher without feedback
      loss  = Σ_v [p_T(v) − p_ref(v)] · log[p_T(v) / p_S(v)]
      Memory: two frozen 7B models on 2 GPUs. Reduce --batch_size if OOM.

T5  JSD β=0.5:
      m    = 0.5·p_T + 0.5·p_S   (stop-gradient)
      loss = 0.5·KL(p_T ‖ m) + 0.5·KL(p_S ‖ m)
      Gradients only through p_S terms; p_T and m are stopped.

AlpacaEval validation every --eval_steps optimizer steps (default 40):
  → student switches to eval mode on rank 0
  → batched generation directly from FSDP model (no vLLM, no checkpoint merge)
  → 500 concurrent OpenAI threads for GPT-4o-mini judge
  → LC win rate logged to wandb, student switches back to train mode

Run (one variant):
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29500 \\
        scripts/fkl/train_olmo_fkl.py \\
        --input      /data/.../ystar_olmo_xo_C_full.jsonl \\
        --output_dir /data/group_data/cx_group/ssmurali/offpolicy/fkl/olmo3/T2 \\
        --model      allenai/OLMo-3-7B-Instruct-SFT \\
        --mode       T2 \\
        --run_name   olmo_fkl_T2
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
try:
    from torch.distributed.fsdp import ShardedStateDictConfig
except ImportError:
    from torch.distributed.fsdp.api import ShardedStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ── System prompts (mirrors generate_olmo.py) ────────────────────────────────
SYSTEM_STUDENT = (
    "You are a helpful assistant. "
    "Respond directly and helpfully to the user's request."
)
SYSTEM_TEACHER = "You are a helpful assistant."
HINDSIGHT_C    = "Note: {o}"   # Prompt C injection text


# ── Distributed helpers ──────────────────────────────────────────────────────

def setup():
    dist.init_process_group("nccl")
    rank       = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def cleanup():
    dist.destroy_process_group()

def is_main(rank: int) -> bool:
    return rank == 0


# ── Dataset ──────────────────────────────────────────────────────────────────

def _inject_o(x: list, o: dict | str) -> list:
    """Inject 'Note: {o_content}' into the last user turn of x (Prompt C)."""
    o_content = o.get("content", "") if isinstance(o, dict) else str(o)
    turns = list(x)
    for i in range(len(turns) - 1, -1, -1):
        if turns[i].get("role") == "user":
            turns[i] = {**turns[i], "content": turns[i]["content"] + "\n\n" + HINDSIGHT_C.format(o=o_content)}
            break
    return turns


class OlmoFKLDataset(Dataset):
    """
    Produces per-sample:
      teacher_input_ids / teacher_labels  — frozen OLMo sees x_with_o + y*
      student_input_ids / student_labels  — student (and reference) sees x_only + y*

    Label tensors: -100 for prefix tokens, actual token ids for y* positions.
    Both sequences share the same y* token sequence, enabling aligned logit extraction.
    """

    def __init__(self, path: str, tokenizer, max_prompt_len: int, max_compl_len: int, rank: int):
        self.tokenizer     = tokenizer
        self.max_prompt_len = max_prompt_len
        self.max_compl_len  = max_compl_len
        self.items = self._load(path, rank)

    def _encode_pair(self, prefix_msgs: list, full_msgs: list):
        tok = self.tokenizer
        prefix_text = tok.apply_chat_template(prefix_msgs, tokenize=False, add_generation_prompt=True)
        full_text   = tok.apply_chat_template(full_msgs,   tokenize=False, add_generation_prompt=False)
        prefix_ids  = tok(prefix_text, add_special_tokens=False)["input_ids"]
        full_enc    = tok(full_text, add_special_tokens=False,
                         truncation=True, max_length=self.max_prompt_len + self.max_compl_len)
        input_ids   = full_enc["input_ids"]
        prompt_len  = len(prefix_ids)
        labels      = [-100] * len(input_ids)
        if prompt_len < len(input_ids):
            labels[prompt_len:] = input_ids[prompt_len:]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def _build_item(self, item: dict):
        x  = item.get("x", [])
        o  = item.get("o")
        y_star_text = item["y_star"] if isinstance(item["y_star"], str) else item["y_star"]["content"]
        y_star_msg  = {"role": "assistant", "content": y_star_text}

        x_o  = _inject_o(x, o) if o is not None else list(x)
        x_plain = list(x)

        teacher_prefix = [{"role": "system", "content": SYSTEM_TEACHER}] + x_o
        student_prefix = [{"role": "system", "content": SYSTEM_STUDENT}] + x_plain

        t_ids, t_labels = self._encode_pair(teacher_prefix, teacher_prefix + [y_star_msg])
        s_ids, s_labels = self._encode_pair(student_prefix, student_prefix + [y_star_msg])
        return t_ids, t_labels, s_ids, s_labels

    def _load(self, path: str, rank: int):
        raw = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    raw.append(json.loads(line))

        items, skipped = [], 0
        for item in tqdm(raw, desc="Tokenizing", disable=not is_main(rank)):
            try:
                t_ids, t_labels, s_ids, s_labels = self._build_item(item)
            except Exception:
                skipped += 1
                continue
            if all(l == -100 for l in t_labels.tolist()) or all(l == -100 for l in s_labels.tolist()):
                skipped += 1
                continue
            items.append({
                "teacher_input_ids": t_ids,
                "teacher_labels":    t_labels,
                "student_input_ids": s_ids,
                "student_labels":    s_labels,
            })

        if is_main(rank):
            print(f"[Dataset] {len(items)} usable, {skipped} skipped", flush=True)
        return items

    def __len__(self):  return len(self.items)
    def __getitem__(self, idx):  return self.items[idx]


def collate(batch: list, pad_id: int) -> dict:
    def pad(tensors, val):
        return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=val)
    t_ids    = pad([x["teacher_input_ids"] for x in batch], pad_id)
    t_labels = pad([x["teacher_labels"]    for x in batch], -100)
    s_ids    = pad([x["student_input_ids"] for x in batch], pad_id)
    s_labels = pad([x["student_labels"]    for x in batch], -100)
    return {
        "teacher_input_ids":      t_ids,
        "teacher_attention_mask": t_ids.ne(pad_id).long(),
        "teacher_labels":         t_labels,
        "student_input_ids":      s_ids,
        "student_attention_mask": s_ids.ne(pad_id).long(),
        "student_labels":         s_labels,
    }


# ── Loss functions ────────────────────────────────────────────────────────────

def _extract_ystar(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Extract logits at y* positions (shifted by 1).

    For each sequence in the batch, teacher_labels[b, t] != -100 marks y* positions.
    The shifted logit at t-1 predicts token t.

    Returns (N_total_ystar_tokens, V) — ordering is aligned across teacher/student
    because both use the same y* token sequence (just different prefixes).
    """
    shifted_logits = logits[:, :-1, :]       # (B, T-1, V)
    shifted_labels = labels[:, 1:]           # (B, T-1)
    mask           = shifted_labels != -100  # (B, T-1)
    return shifted_logits[mask]              # (N, V)


def loss_T2(t_logits, s_logits, t_labels, s_labels):
    """Soft FKL: KL(p_T ‖ p_S) per y* token, averaged."""
    with torch.no_grad():
        p_T     = torch.softmax(_extract_ystar(t_logits, t_labels), dim=-1)  # (N, V)
        log_p_T = torch.log(p_T.clamp(min=1e-10))

    log_p_S      = torch.log_softmax(_extract_ystar(s_logits, s_labels), dim=-1)
    kl_per_token = (p_T * (log_p_T - log_p_S)).sum(dim=-1)   # (N,)
    loss         = kl_per_token.mean()
    signal_mean  = kl_per_token.detach().mean().item()
    return loss, signal_mean


def loss_T4(t_logits, r_logits, s_logits, t_labels, s_labels):
    """Contrastive FKL: Σ_v (p_T − p_ref) · log(p_T / p_S), averaged over y* tokens."""
    with torch.no_grad():
        p_T     = torch.softmax(_extract_ystar(t_logits, t_labels), dim=-1)  # (N, V)
        p_ref   = torch.softmax(_extract_ystar(r_logits, s_labels), dim=-1)  # ref uses x-only context
        weight  = p_T - p_ref                                                  # (N, V) can be negative
        log_p_T = torch.log(p_T.clamp(min=1e-10))

    log_p_S      = torch.log_softmax(_extract_ystar(s_logits, s_labels), dim=-1)
    signal_part  = (weight * log_p_T).sum(dim=-1)   # (N,) — no grad
    student_part = (weight * log_p_S).sum(dim=-1)   # (N,) — grad through log_p_S
    loss         = (signal_part - student_part).mean()
    signal_mean  = weight.abs().mean().item()        # mean |p_T − p_ref| (signal strength)
    return loss, signal_mean


def loss_T5(t_logits, s_logits, t_labels, s_labels):
    """JSD β=0.5: 0.5·KL(p_T ‖ m) + 0.5·KL(p_S ‖ m), m = 0.5(p_T + p_S), m stop-gradient."""
    with torch.no_grad():
        p_T     = torch.softmax(_extract_ystar(t_logits, t_labels), dim=-1)  # (N, V)
        log_p_T = torch.log(p_T.clamp(min=1e-10))

    s_ystar = _extract_ystar(s_logits, s_labels)   # (N, V) — gradient flows here

    with torch.no_grad():
        p_S_sg = torch.softmax(s_ystar.detach(), dim=-1)
        m      = 0.5 * p_T + 0.5 * p_S_sg
        log_m  = torch.log(m.clamp(min=1e-10))

    # KL(p_T ‖ m): no student gradient (p_T and m are stopped)
    kl_T_m = (p_T * (log_p_T - log_m)).sum(dim=-1)   # (N,)

    # KL(p_S ‖ m): student gradient through p_S * (log_p_S − log_m_sg)
    p_S_live    = torch.softmax(s_ystar, dim=-1)           # gradient
    log_p_S_live = torch.log_softmax(s_ystar, dim=-1)      # gradient
    kl_S_m      = (p_S_live * (log_p_S_live - log_m)).sum(dim=-1)  # (N,)

    loss        = (0.5 * kl_T_m + 0.5 * kl_S_m).mean()
    signal_mean = (0.5 * kl_T_m + 0.5 * kl_S_m).detach().mean().item()  # JSD value per token
    return loss, signal_mean


# ── FSDP helpers ─────────────────────────────────────────────────────────────

def _get_layer_cls():
    for module_path, cls_name in [
        ("transformers.models.olmo3.modeling_olmo3", "Olmo3DecoderLayer"),
        ("transformers.models.olmo2.modeling_olmo2", "Olmo2DecoderLayer"),
        ("transformers.models.olmo.modeling_olmo",   "OlmoDecoderLayer"),
    ]:
        try:
            mod = importlib.import_module(module_path)
            return {getattr(mod, cls_name)}
        except (ImportError, AttributeError):
            pass
    return None   # fallback to size-based policy


def wrap_fsdp(model, fp32: bool):
    layer_cls = _get_layer_cls()
    if layer_cls is not None:
        wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls=layer_cls)
    else:
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=1_000_000)

    mp = None if fp32 else MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    return FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mp,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        sync_module_states=False,  # both ranks load same checkpoint independently
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
    )


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_sharded(model, tokenizer, path: str, rank: int):
    dist.barrier()
    os.makedirs(path, exist_ok=True)
    save_policy = ShardedStateDictConfig(offload_to_cpu=True)
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, save_policy):
        state_dict = model.state_dict()
    torch.save(state_dict, os.path.join(path, f"rank_{rank}.pt"))
    if is_main(rank):
        tokenizer.save_pretrained(path)
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump({"world_size": dist.get_world_size()}, f)
        print(f"\n[Saved sharded] {path}", flush=True)
    dist.barrier()


def save_full(model, tokenizer, path: str, rank: int, base_model: str, fp32: bool = False):
    dist.barrier()
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        state_dict = model.state_dict()
    if is_main(rank):
        os.makedirs(path, exist_ok=True)
        torch_dtype = torch.float32 if fp32 else torch.bfloat16
        save_m = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch_dtype,
                                                       trust_remote_code=True)
        save_m.load_state_dict(state_dict)
        save_m.save_pretrained(path)
        tokenizer.save_pretrained(path)
        print(f"\n[Saved full HF] {path}", flush=True)
        del save_m
    dist.barrier()


# ── Inline AlpacaEval validation ─────────────────────────────────────────────
# Runs entirely on the training GPUs — no separate eval job needed.
# Generation: FSDP student in eval mode, batched model.generate().
# Judging:    500 concurrent OpenAI threads (GPT-4o-mini, weighted_alpaca_eval).

ALPACA_PROMPTS_PATH = "arena-hard-auto/arena-hard-auto/alpaca_eval_data/alpaca_eval_prompts.jsonl"
ALPACA_JUDGE_TEMPLATE = (
    "I want you to evaluate which of the following two responses to the user's question is better.\n\n"
    "User question: {instruction}\n\n"
    "Response A:\n{output_a}\n\n"
    "Response B:\n{output_b}\n\n"
    "Reference response (GPT-4):\n{reference}\n\n"
    "Which response is better? Answer with 'A' or 'B' only."
)


def _generate_alpaca_responses(model, tokenizer, prompts_path: str, local_rank: int,
                                max_new_tokens: int = 2048, batch_size: int = 16) -> list[dict]:
    """Batched generation from FSDP model in eval mode. Rank 0 only."""
    with open(prompts_path, encoding="utf-8") as f:
        questions = [json.loads(l) for l in f if l.strip()]

    results = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(questions), batch_size):
            batch_qs = questions[i: i + batch_size]
            # Build prompts
            texts = []
            for q in batch_qs:
                msgs = [{"role": "user", "content": q["instruction"]}]
                texts.append(tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                ))
            enc = tokenizer(texts, return_tensors="pt", padding=True,
                            truncation=True, max_length=4096).to(local_rank)
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
            )
            for q, inp_ids, gen_ids in zip(batch_qs, enc["input_ids"], out):
                text = tokenizer.decode(gen_ids[inp_ids.shape[0]:], skip_special_tokens=True).strip()
                results.append({"instruction": q["instruction"],
                                 "dataset":     q.get("dataset", ""),
                                 "output":      text,
                                 "generator":   "student"})
    return results


def _judge_one(item: dict, reference_output: str, client) -> str:
    """Single GPT-4o-mini judge call. Returns 'A', 'B', or 'tie'."""
    prompt = ALPACA_JUDGE_TEMPLATE.format(
        instruction=item["instruction"],
        output_a=item["output"],
        output_b=reference_output,
        reference=reference_output,
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4,
            temperature=0.0,
        )
        verdict = resp.choices[0].message.content.strip().upper()
        if verdict.startswith("A"):
            return "win"
        elif verdict.startswith("B"):
            return "loss"
        return "tie"
    except Exception:
        return "tie"


def _load_alpaca_references(prompts_path: str) -> dict[str, str]:
    """Load reference (GPT-4 Turbo) outputs from the alpaca_eval dataset."""
    try:
        import alpaca_eval
        from alpaca_eval import utils as ae_utils
        ref_path = Path(alpaca_eval.__file__).parent / "models_configs" / "gpt4_turbo" / "outputs.json"
        if ref_path.exists():
            refs = json.loads(ref_path.read_text())
            return {r["instruction"]: r["output"] for r in refs}
    except Exception:
        pass
    return {}


def run_inline_eval(model, tokenizer, args, rank, local_rank, global_step: int) -> float | None:
    """
    Run AlpacaEval inline on the training GPUs.
    Only rank 0 runs generation and judging; result is broadcast to all ranks.
    Returns LC win rate (float) or None if eval failed.
    """
    import time as _time
    t0 = _time.perf_counter()

    lc_wr_tensor = torch.zeros(1, device=local_rank)

    if is_main(rank):
        print(f"\n[Eval] step={global_step} — generating AlpacaEval responses...", flush=True)
        try:
            outputs = _generate_alpaca_responses(
                model, tokenizer,
                prompts_path=os.path.join(args.repo_dir, ALPACA_PROMPTS_PATH),
                local_rank=local_rank,
                max_new_tokens=args.eval_max_tokens,
                batch_size=args.eval_gen_batch,
            )

            # Save outputs
            out_file = os.path.join(args.output_dir, f"alpaca_eval_step{global_step:06d}.json")
            with open(out_file, "w") as f:
                json.dump(outputs, f, indent=2)

            # Judge with concurrent OpenAI calls
            try:
                from openai import OpenAI
                client = OpenAI()
            except ImportError:
                print("[Eval] openai not installed, skipping judge", flush=True)
                raise

            refs = _load_alpaca_references(os.path.join(args.repo_dir, ALPACA_PROMPTS_PATH))
            wins = losses = ties = 0

            print(f"[Eval] judging {len(outputs)} outputs with {args.eval_judge_workers} threads...", flush=True)
            with ThreadPoolExecutor(max_workers=args.eval_judge_workers) as ex:
                futures = {
                    ex.submit(_judge_one, item, refs.get(item["instruction"], ""), client): item
                    for item in outputs
                }
                for fut in as_completed(futures):
                    v = fut.result()
                    if v == "win":   wins   += 1
                    elif v == "loss": losses += 1
                    else:             ties   += 1

            # Raw win rate (ties excluded) as LC proxy
            total = wins + losses
            lc_wr = wins / total if total > 0 else 0.0
            elapsed = _time.perf_counter() - t0
            print(f"[Eval] step={global_step} | WR={lc_wr*100:.1f}% "
                  f"(W={wins} L={losses} T={ties}) | {elapsed:.0f}s", flush=True)
            lc_wr_tensor[0] = lc_wr

        except Exception as e:
            print(f"[Eval] FAILED: {e}", flush=True)

    # Broadcast result to all ranks so training can resume in sync
    dist.broadcast(lc_wr_tensor, src=0)
    model.train()
    return lc_wr_tensor[0].item() if lc_wr_tensor[0].item() > 0 else None


# ── Training loop ─────────────────────────────────────────────────────────────

def train(student, teacher, tokenizer, dataset, args, rank, local_rank, world_size):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    optimizer = AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)

    total_steps  = (len(loader) * args.epochs) // args.grad_accum
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    wandb_run_id = None
    if is_main(rank) and HAS_WANDB and not args.no_wandb:
        wrun = wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
        wandb_run_id = wrun.id

    if is_main(rank):
        eff_bs = args.batch_size * world_size * args.grad_accum
        print(f"Mode={args.mode} | steps/epoch={len(loader)} | total={total_steps} | warmup={warmup_steps} | eff_batch={eff_bs}", flush=True)

    global_step = 0
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        student.train()
        optimizer.zero_grad()

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=not is_main(rank))
        for step, batch in enumerate(pbar):
            t_ids   = batch["teacher_input_ids"].to(local_rank)
            t_mask  = batch["teacher_attention_mask"].to(local_rank)
            t_lbls  = batch["teacher_labels"].to(local_rank)
            s_ids   = batch["student_input_ids"].to(local_rank)
            s_mask  = batch["student_attention_mask"].to(local_rank)
            s_lbls  = batch["student_labels"].to(local_rank)

            with torch.no_grad():
                t_logits = teacher(input_ids=t_ids, attention_mask=t_mask).logits
                # T4: ref is the same frozen model but sees x-only (no "Note: o")
                r_logits = teacher(input_ids=s_ids, attention_mask=s_mask).logits if args.mode == "T4" else None

            s_logits = student(input_ids=s_ids, attention_mask=s_mask).logits

            if args.mode == "T2":
                loss, signal_mean = loss_T2(t_logits, s_logits, t_lbls, s_lbls)
            elif args.mode == "T4":
                loss, signal_mean = loss_T4(t_logits, r_logits, s_logits, t_lbls, s_lbls)
            else:  # T5
                loss, signal_mean = loss_T5(t_logits, s_logits, t_lbls, s_lbls)

            (loss / args.grad_accum).backward()

            if (step + 1) % args.grad_accum == 0:
                student.clip_grad_norm_(1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if is_main(rank):
                    lr = scheduler.get_last_lr()[0]
                    pbar.set_postfix({"loss": f"{loss.item():.4f}", "sig": f"{signal_mean:.4f}", "lr": f"{lr:.1e}"})
                    if HAS_WANDB and not args.no_wandb:
                        wandb.log({
                            "loss":        loss.item(),
                            "signal_mean": signal_mean,
                            "lr":          lr,
                            "global_step": global_step,
                        }, step=global_step)

                # AlpacaEval validation every eval_steps
                if global_step % args.eval_steps == 0:
                    lc_wr = run_inline_eval(student, tokenizer, args, rank, local_rank, global_step)
                    if is_main(rank) and lc_wr is not None and HAS_WANDB and not args.no_wandb:
                        wandb.log({"alpaca_lc_winrate": lc_wr, "global_step": global_step},
                                  step=global_step)

        # End-of-epoch sharded checkpoint
        ep_dir = os.path.join(args.output_dir, f"epoch-{epoch + 1}")
        save_sharded(student, tokenizer, ep_dir, rank)
        student.train()

    # Final full HF checkpoint (vLLM-ready)
    save_full(student, tokenizer, os.path.join(args.output_dir, "final"), rank, args.model, args.fp32)

    if is_main(rank):
        print("Training complete.", flush=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",          required=True,
                        help="ystar_olmo_xo_C.jsonl (full WildFeedback)")
    parser.add_argument("--output_dir",     default="./checkpoints")
    parser.add_argument("--model",          default="allenai/OLMo-3-7B-Instruct-SFT")
    parser.add_argument("--mode",           choices=["T2", "T4", "T5"], required=True,
                        help="T2=soft FKL, T4=contrastive FKL, T5=JSD β=0.5")
    parser.add_argument("--batch_size",     type=int,   default=32,
                        help="Per-GPU batch size. Effective = batch_size × world_size × grad_accum. "
                             "Default 32 × 2 GPUs × 1 = 64.")
    parser.add_argument("--grad_accum",     type=int,   default=1)
    parser.add_argument("--epochs",         type=int,   default=2)
    parser.add_argument("--lr",             type=float, default=5e-6)
    parser.add_argument("--warmup_ratio",   type=float, default=0.05)
    parser.add_argument("--max_prompt_len", type=int,   default=2048)
    parser.add_argument("--max_compl_len",  type=int,   default=2048)
    parser.add_argument("--fp32",           action="store_true")
    parser.add_argument("--eval_steps",        type=int, default=40,
                        help="Run inline AlpacaEval every N optimizer steps.")
    parser.add_argument("--eval_gen_batch",    type=int, default=16,
                        help="Batch size for AlpacaEval generation (model.generate).")
    parser.add_argument("--eval_max_tokens",   type=int, default=2048)
    parser.add_argument("--eval_judge_workers",type=int, default=500,
                        help="Concurrent OpenAI threads for judge calls.")
    parser.add_argument("--repo_dir",          default="/home/ssmurali/user_interactions",
                        help="Repo root for locating alpaca_eval prompts file.")
    parser.add_argument("--wandb_project",  default="olmo-fkl")
    parser.add_argument("--run_name",       required=True,
                        help="WandB run name and prefix for eval queue files.")
    parser.add_argument("--no_wandb",       action="store_true")
    args = parser.parse_args()

    rank, local_rank, world_size = setup()
    os.makedirs(args.output_dir, exist_ok=True)

    torch_dtype = torch.float32 if args.fp32 else torch.bfloat16

    if is_main(rank):
        print(f"=== OLMo FKL | mode={args.mode} | world_size={world_size} | eff_batch={args.batch_size * world_size * args.grad_accum} ===", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = OlmoFKLDataset(args.input, tokenizer, args.max_prompt_len, args.max_compl_len, rank)

    # ── Student — FSDP ZeRO-3, sharded first while GPU is empty ─────────────
    # Must load before teacher: FSDP shard clone needs ~7GB free on each GPU.
    # Loading teacher first (~14GB) leaves too little headroom.
    if is_main(rank):
        print("Loading student on CPU (FSDP will shard)...", flush=True)
    student = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, attn_implementation="sdpa", trust_remote_code=True,
    )
    student.gradient_checkpointing_enable()
    student = wrap_fsdp(student, args.fp32)
    # Trigger FSDP lazy init (_full_param_padded ~13.6 GiB all-gather workspace)
    # before the teacher occupies GPU memory. Without this, the buffer is allocated
    # on the first real forward when teacher (~14 GiB) is already loaded → OOM.
    _dummy = torch.zeros(1, 4, dtype=torch.long, device=local_rank)
    with torch.no_grad():
        student(input_ids=_dummy, attention_mask=torch.ones_like(_dummy))
    del _dummy
    torch.cuda.empty_cache()
    if is_main(rank):
        print("FSDP student ready (lazy init done).", flush=True)

    # ── Frozen teacher — loaded after FSDP sharding is done ──────────────────
    if is_main(rank):
        print(f"Loading frozen teacher on GPU {local_rank}...", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, attn_implementation="sdpa", trust_remote_code=True,
    ).to(local_rank).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    train(student, teacher, tokenizer, dataset, args, rank, local_rank, world_size)
    cleanup()


if __name__ == "__main__":
    main()
