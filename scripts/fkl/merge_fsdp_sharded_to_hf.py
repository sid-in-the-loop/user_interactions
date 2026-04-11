#!/usr/bin/env python3
"""
Merge FSDP sharded checkpoints (rank_0.pt, rank_1.pt, ...) written by train_fkl_fsdp.py
into a single HuggingFace directory (config.json + weights) for vLLM eval.

Must run with torchrun with nproc_per_node == meta.json world_size (typically 2).

Usage:
  torchrun --nproc_per_node=2 scripts/fkl/merge_fsdp_sharded_to_hf.py \\
    --tasks_file eval_results/merge_fsdp_tasks.txt \\
    --base_model Qwen/Qwen3-4B \\
    --fp32
"""
from __future__ import annotations

import argparse
import json
import os
from functools import partial

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
from transformers import AutoModelForCausalLM, AutoTokenizer


def setup():
    dist.init_process_group("nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def build_fsdp_model(base_model: str, fp32: bool, rank: int):
    torch_dtype = torch.float32 if fp32 else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        attn_implementation="sdpa",
    )
    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

        layer_cls = {Qwen3DecoderLayer}
    except ImportError:
        try:
            from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

            layer_cls = {Qwen2DecoderLayer}
        except ImportError:
            from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

            wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=1_000_000)
            layer_cls = None

    if layer_cls is not None:
        wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=layer_cls,
        )
    if fp32:
        mp = None
    else:
        mp = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mp,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
    )
    return model


def merge_one(
    shard_dir: str,
    out_dir: str,
    base_model: str,
    fp32: bool,
    rank: int,
    world_size: int,
    skip_existing: bool,
) -> bool:
    shard_dir = os.path.abspath(shard_dir)
    out_dir = os.path.abspath(out_dir)

    if is_main(rank) and skip_existing and os.path.isfile(os.path.join(out_dir, "config.json")):
        print(f"[skip exists] {out_dir}")
        dist.barrier()
        return True

    meta_path = os.path.join(shard_dir, "meta.json")
    if not os.path.isfile(meta_path):
        if is_main(rank):
            print(f"[skip] no meta.json: {shard_dir}")
        dist.barrier()
        return False
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    ws = int(meta.get("world_size", 0))
    if ws != world_size:
        if is_main(rank):
            print(f"[skip] meta world_size={ws} != WORLD_SIZE={world_size}: {shard_dir}")
        dist.barrier()
        return False

    rank_pt = os.path.join(shard_dir, f"rank_{rank}.pt")
    if not os.path.isfile(rank_pt):
        if is_main(rank):
            print(f"[skip] missing {rank_pt}")
        dist.barrier()
        return False

    if is_main(rank):
        print(f"\n=== merge {shard_dir} -> {out_dir} ===")

    tokenizer = AutoTokenizer.from_pretrained(shard_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_fsdp_model(base_model, fp32, rank)
    save_policy = ShardedStateDictConfig(offload_to_cpu=True)
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, save_policy):
        # weights_only=False needed for ShardedTensor objects (PyTorch 2.6+ default changed)
        sd = torch.load(rank_pt, map_location="cpu", weights_only=False)
        model.load_state_dict(sd, strict=True)
    del sd
    dist.barrier()

    full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_cfg):
        full_sd = model.state_dict()
    dist.barrier()

    if is_main(rank):
        os.makedirs(out_dir, exist_ok=True)
        torch_dtype = torch.float32 if fp32 else torch.bfloat16
        save_m = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch_dtype,
            attn_implementation="sdpa",
        )
        save_m.load_state_dict(full_sd, strict=True)
        save_m.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        print(f"[ok] wrote {out_dir}")
    del full_sd
    del model
    torch.cuda.empty_cache()
    dist.barrier()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tasks_file",
        required=True,
        help="Lines: shard_dir|out_dir  (optional third field: base_model override)",
    )
    ap.add_argument("--base_model", default="Qwen/Qwen3-4B")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--no_skip_existing", action="store_true")
    args = ap.parse_args()

    rank, _local_rank, world_size = setup()
    skip_existing = not args.no_skip_existing

    with open(args.tasks_file, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]

    ok = 0
    for ln in lines:
        parts = [p.strip() for p in ln.split("|")]
        if len(parts) < 2:
            if is_main(rank):
                print(f"[skip bad line] {ln}")
            continue
        shard_dir, out_dir = parts[0], parts[1]
        base = parts[2] if len(parts) > 2 else args.base_model
        try:
            if merge_one(shard_dir, out_dir, base, args.fp32, rank, world_size, skip_existing):
                ok += 1
        except Exception as e:
            if is_main(rank):
                print(f"[FAIL] {shard_dir}: {e}")
            raise
    if is_main(rank):
        print(f"\nDone. Merged {ok} checkpoint(s).")
    cleanup()


if __name__ == "__main__":
    main()
