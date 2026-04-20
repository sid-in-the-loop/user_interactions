#!/usr/bin/env python3
"""
Push all LoRA adapter checkpoints to HuggingFace Hub.

One repo per method, all checkpoints as subfolders:
  ssmurali/user-interactions-jsd
    ├── step-10/
    ├── step-20/
    ├── ...
    └── final/

Usage:
  # Dry run:
  python scripts/eval/push_checkpoints_to_hf.py --dry_run

  # Push all:
  HF_TOKEN=... python scripts/eval/push_checkpoints_to_hf.py

  # Push one method:
  HF_TOKEN=... python scripts/eval/push_checkpoints_to_hf.py --methods jsd_p30
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo


METHOD_NAMES = {
    "sft_p30": "sft",
    "fkl_p30": "fkl",
    "jsd_p30": "jsd",
    "dpo_p30": "dpo",
    "jsd_is1_p30": "jsd-is1",
    "jsd_is2_p30": "jsd-is2",
    "jsd_is3_p30": "jsd-is3",
    "jsd_is4_p30": "jsd-is4",
    "zg_jsd_p30": "zg-jsd",
    "rkl_p30": "rkl",
    "rlad_p30": "rlad",
    "distillm2_p30": "distillm2",
}


def main():
    parser = argparse.ArgumentParser(description="Push LoRA checkpoints to HuggingFace Hub")
    parser.add_argument("--checkpoints_root", default="/projects/bgtw/ssredharan/checkpoints")
    parser.add_argument("--hf_user", default="ssmurali")
    parser.add_argument("--repo_prefix", default="user-interactions")
    parser.add_argument("--methods", nargs="+", default=None, help="Only push these method dirs")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    root = Path(args.checkpoints_root)

    # Discover methods
    method_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if args.methods:
        method_dirs = [d for d in method_dirs if d.name in args.methods]

    total_size = 0
    total_ckpts = 0

    for method_dir in method_dirs:
        method_key = method_dir.name
        clean_name = METHOD_NAMES.get(method_key, method_key.replace("_p30", "").replace("_", "-"))
        repo_id = f"{args.hf_user}/{args.repo_prefix}-{clean_name}"

        # Find checkpoints
        ckpts = sorted([
            d for d in method_dir.iterdir()
            if d.is_dir() and (d / "adapter_config.json").exists()
        ])

        if not ckpts:
            print(f"  {method_key}: no checkpoints found, skipping")
            continue

        size_mb = sum(
            f.stat().st_size for ckpt in ckpts for f in ckpt.rglob("*") if f.is_file()
        ) / 1e6

        print(f"\n{repo_id}")
        print(f"  {len(ckpts)} checkpoints, {size_mb:.0f} MB total")
        for c in ckpts:
            print(f"    {c.name}/")

        total_size += size_mb
        total_ckpts += len(ckpts)

        if args.dry_run:
            continue

        token = os.environ.get("HF_TOKEN")
        if not token:
            print("ERROR: Set HF_TOKEN environment variable")
            return

        api = HfApi(token=token)

        # Create repo
        try:
            create_repo(repo_id, token=token, private=args.private, exist_ok=True, repo_type="model")
            print(f"  Created/verified repo: {repo_id}")
        except Exception as e:
            print(f"  ERROR creating repo: {e}")
            continue

        # Upload checkpoint by checkpoint to avoid large folder issues
        uploaded = 0
        for ckpt in ckpts:
            try:
                api.upload_folder(
                    folder_path=str(ckpt),
                    path_in_repo=ckpt.name,
                    repo_id=repo_id,
                    token=token,
                )
                uploaded += 1
                print(f"    {ckpt.name} ✓ ({uploaded}/{len(ckpts)})")
            except Exception as e:
                print(f"    {ckpt.name} ERROR: {e}")
        print(f"  Done: {uploaded}/{len(ckpts)} checkpoints uploaded")

    print(f"\n{'='*50}")
    print(f"Total: {len(method_dirs)} methods, {total_ckpts} checkpoints, {total_size:.0f} MB")
    if args.dry_run:
        print("(dry run — nothing was pushed)")


if __name__ == "__main__":
    main()
