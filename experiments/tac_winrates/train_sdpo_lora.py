"""Offline SDPO (= offline RKL) with LoRA.

Thin wrapper around scripts/sdpo/offline_sdpo_trainer.py. Adds:
  - PEFT LoRA wrapping (user-chosen r/alpha/dropout/target)
  - Dynamic save_steps = total_steps // num_ckpts (so we get ~num_ckpts evenly
    spaced intermediate ckpts plus 'final_model')
  - Pushes OUTPUT_DIR via CLI flag instead of env var

Expected input JSONL schema (produced by build_training_inputs.py):
  {prompt: list[msg], user_response: {content}, completion: {content}}
"""

import argparse
import math
import os
import sys
from pathlib import Path

# $REPO/datasets/ is a data dir (3GB of jsonl), not a Python package. With
# CWD=$REPO it becomes an implicit namespace package and shadows the pip
# `datasets` pkg. Drop "" and $REPO from sys.path before importing.
_REPO = "/u/ssredharan/user_interactions"
sys.path[:] = [p for p in sys.path if p not in ("", _REPO) and Path(p).resolve() != Path(_REPO).resolve()]

# Make the existing sdpo trainer importable
sys.path.insert(
    0,
    str(Path("/u/ssredharan/user_interactions/scripts/sdpo").resolve()),
)

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model

from offline_sdpo_trainer import OfflineSDPOCollator, OfflineSDPOTrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_epochs", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=2e-6)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_completion_length", type=int, default=2048)
    p.add_argument("--num_ckpts", type=int, default=4,
                   help="Intermediate ckpts evenly spaced across training.")
    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target", type=str, default="all-linear")
    p.add_argument("--wandb_project", default="tac-winrates-sdpo")
    p.add_argument("--run_name", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.run_name:
        os.environ["WANDB_NAME"] = args.run_name
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    print(f"Loading train data: {args.train_jsonl}", flush=True)
    dataset = load_dataset("json", data_files=args.train_jsonl, split="train")
    print(f"  n_rows = {len(dataset)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    try:
        from transformers.utils import is_flash_attn_2_available
        attn_impl = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
    except ImportError:
        attn_impl = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )
    model.generation_config.do_sample = True
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0

    # Required for LoRA + gradient_checkpointing: the base model's input
    # activations must carry requires_grad for the backward graph to connect
    # through the (frozen) base to the (trainable) LoRA adapters.
    model.enable_input_require_grads()

    # LoRA wrap
    target_modules = ("all-linear" if args.lora_target == "all-linear"
                      else [m.strip() for m in args.lora_target.split(",")])
    lora_cfg = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Compute dynamic save_steps: roughly num_ckpts evenly across training
    steps_per_epoch = math.ceil(len(dataset) / (args.batch_size * args.grad_accum))
    total_steps = max(1, steps_per_epoch * args.num_epochs)
    save_steps = max(1, total_steps // args.num_ckpts)
    print(f"steps_per_epoch={steps_per_epoch}  total_steps={total_steps}  "
          f"save_steps={save_steps}", flush=True)

    collator = OfflineSDPOCollator(
        tokenizer=tokenizer,
        max_completion_length=args.max_completion_length,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        logging_steps=10,
        save_strategy="steps",
        save_steps=save_steps,
        report_to=["wandb"],
        warmup_ratio=0.05,
        max_grad_norm=10.0,
        lr_scheduler_type="cosine",
        remove_unused_columns=False,
        dataloader_num_workers=4,
        seed=42,
    )

    trainer = OfflineSDPOTrainer(
        ignore_first_k=2,
        model=model,
        ref_model=None,
        kl_beta=0,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    final = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final)
    tokenizer.save_pretrained(final)
    print(f"done -> {final}", flush=True)


if __name__ == "__main__":
    main()
