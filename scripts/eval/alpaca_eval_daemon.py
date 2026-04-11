#!/usr/bin/env python3
"""
AlpacaEval validation daemon for OLMo FKL training.

Watches --queue_dir/pending/ for JSON eval requests written by train_olmo_fkl.py.
Processes each request sequentially:
  1. Merge FSDP sharded checkpoint → HF format  (torchrun on daemon's GPUs)
  2. Generate AlpacaEval 2.0 outputs             (vLLM, 805 prompts)
  3. Run GPT-4o-mini judge                       (alpaca_eval CLI)
  4. Parse LC win rate from leaderboard.csv
  5. Log to wandb (resume by run_id)

Exits when queue is empty AND all expected training runs have written their
sentinel files ({queue_dir}/sentinel_{run_name}.done).

Run:
    CUDA_VISIBLE_DEVICES=6,7 python scripts/eval/alpaca_eval_daemon.py \\
        --queue_dir       /data/group_data/cx_group/ssmurali/offpolicy/fkl/olmo3/eval_queue \\
        --base_model      allenai/OLMo-3-7B-Instruct-SFT \\
        --run_names       olmo_fkl_T2 olmo_fkl_T4 olmo_fkl_T5
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

ALPACA_PROMPTS = "arena-hard-auto/arena-hard-auto/alpaca_eval_data/alpaca_eval_prompts.jsonl"


# ── Step helpers ──────────────────────────────────────────────────────────────

def merge_checkpoint(shard_dir: str, hf_dir: str, base_model: str, repo_dir: str, tp_size: int) -> bool:
    """Merge FSDP shards to a single HF directory using merge_fsdp_sharded_to_hf.py."""
    if Path(hf_dir, "config.json").exists():
        print(f"[Merge] already merged, skipping: {hf_dir}", flush=True)
        return True

    tasks_file = str(Path(shard_dir).parent / f"merge_{Path(shard_dir).name}.txt")
    Path(tasks_file).write_text(f"{shard_dir}|{hf_dir}\n")

    cmd = [
        "torchrun",
        f"--nproc_per_node={tp_size}",
        "--master_port=29600",
        os.path.join(repo_dir, "scripts/fkl/merge_fsdp_sharded_to_hf.py"),
        "--tasks_file", tasks_file,
        "--base_model",  base_model,
    ]
    print(f"[Merge] {Path(shard_dir).name} → {hf_dir}", flush=True)
    result = subprocess.run(cmd, cwd=repo_dir)
    return result.returncode == 0


def generate_outputs(hf_dir: str, model_name: str, output_file: str, repo_dir: str, tp_size: int) -> bool:
    """Generate AlpacaEval model outputs with vLLM."""
    if os.path.exists(output_file):
        print(f"[Gen] already exists: {output_file}", flush=True)
        return True

    cmd = [
        sys.executable,
        os.path.join(repo_dir, "scripts/eval/alpaca_eval_gen.py"),
        "--model_name",          model_name,
        "--model_path",          hf_dir,
        "--input_file",          os.path.join(repo_dir, ALPACA_PROMPTS),
        "--output_file",         output_file,
        "--tensor_parallel_size", str(tp_size),
        "--mode",                "nonthinking",
    ]
    print(f"[Gen] {model_name}", flush=True)
    result = subprocess.run(cmd, cwd=repo_dir)
    return result.returncode == 0


def run_judge(output_file: str, output_dir: str) -> bool:
    """Run weighted AlpacaEval gpt-4o-mini judge."""
    leaderboard = Path(output_dir) / "weighted_alpaca_eval_gpt4_turbo" / "leaderboard.csv"
    if leaderboard.exists():
        print(f"[Judge] already done: {leaderboard}", flush=True)
        return True

    cmd = [
        "alpaca_eval",
        "--model_outputs",     output_file,
        "--annotators_config", "weighted_alpaca_eval_gpt4_turbo",
        "--output_path",       output_dir,
    ]
    print(f"[Judge] {Path(output_file).parent.name}", flush=True)
    result = subprocess.run(cmd)
    return result.returncode == 0


def parse_lc_winrate(output_dir: str, model_name: str) -> float | None:
    """Extract length_controlled_winrate for model_name from leaderboard.csv."""
    leaderboard = Path(output_dir) / "weighted_alpaca_eval_gpt4_turbo" / "leaderboard.csv"
    if not leaderboard.exists():
        return None

    lines = leaderboard.read_text().splitlines()
    if len(lines) < 2:
        return None

    header = [h.strip().strip('"') for h in lines[0].split(",")]
    try:
        lc_idx = header.index("length_controlled_winrate")
    except ValueError:
        try:
            lc_idx = header.index("win_rate")
        except ValueError:
            return None

    for line in lines[1:]:
        parts = [p.strip().strip('"') for p in line.split(",")]
        if parts and parts[0] == model_name:
            try:
                return float(parts[lc_idx])
            except (ValueError, IndexError):
                pass
    return None


def log_to_wandb(wandb_run_id: str | None, run_name: str, step: int, lc_winrate: float, project: str):
    if not HAS_WANDB or not wandb_run_id:
        print(f"[WandB] {run_name} step={step} lc_winrate={lc_winrate:.4f} (wandb not available)", flush=True)
        return
    run = wandb.init(project=project, id=wandb_run_id, resume="allow")
    run.log({"alpaca_lc_winrate": lc_winrate, "global_step": step})
    run.finish()
    print(f"[WandB] logged {run_name} step={step} lc_winrate={lc_winrate:.4f}", flush=True)


# ── Request processing ────────────────────────────────────────────────────────

def process_request(req: dict, args: argparse.Namespace) -> bool:
    run_name     = req["run_name"]
    step         = req["step"]
    shard_dir    = req["shard_dir"]
    wandb_run_id = req.get("wandb_run_id")

    hf_dir      = shard_dir + "_hf"
    model_name  = f"{run_name}_step{step}"
    output_file = os.path.join(args.eval_output_dir, run_name, f"step-{step:06d}", "model_outputs.json")
    output_dir  = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    if not merge_checkpoint(shard_dir, hf_dir, args.base_model, args.repo_dir, args.tp_size):
        print(f"[ERROR] Merge failed: {shard_dir}", flush=True)
        return False

    if not generate_outputs(hf_dir, model_name, output_file, args.repo_dir, args.tp_size):
        print(f"[ERROR] Generation failed: {model_name}", flush=True)
        return False

    if not run_judge(output_file, output_dir):
        print(f"[ERROR] Judge failed: {model_name}", flush=True)
        return False

    lc_winrate = parse_lc_winrate(output_dir, model_name)
    if lc_winrate is None:
        print(f"[WARN] Could not parse LC winrate for {model_name}", flush=True)
        lc_winrate = float("nan")

    log_to_wandb(wandb_run_id, run_name, step, lc_winrate, args.wandb_project)
    print(f"[Done] {model_name} | LC winrate: {lc_winrate:.4f}", flush=True)
    return True


# ── Main daemon loop ──────────────────────────────────────────────────────────

def all_sentinels_present(queue_dir: str, run_names: list[str]) -> bool:
    return all(Path(queue_dir, f"sentinel_{rn}.done").exists() for rn in run_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue_dir",       required=True,
                        help="Root queue directory (contains pending/, in_progress/, done/ subdirs).")
    parser.add_argument("--base_model",      default="allenai/OLMo-3-7B-Instruct-SFT")
    parser.add_argument("--repo_dir",        default="/home/ssmurali/user_interactions")
    parser.add_argument("--eval_output_dir", default=None,
                        help="Where to write AlpacaEval outputs per run/step. "
                             "Defaults to {queue_dir}/../alpaca_eval")
    parser.add_argument("--run_names",       nargs="+",
                        default=["olmo_fkl_T2", "olmo_fkl_T4", "olmo_fkl_T5"],
                        help="Expected run names. Daemon exits when all sentinels present and queue empty.")
    parser.add_argument("--tp_size",         type=int, default=2,
                        help="Tensor parallel size for merge and generation (= number of daemon GPUs).")
    parser.add_argument("--poll_interval",   type=int, default=30,
                        help="Seconds between queue polls when idle.")
    parser.add_argument("--wandb_project",   default="olmo-fkl")
    args = parser.parse_args()

    if args.eval_output_dir is None:
        args.eval_output_dir = str(Path(args.queue_dir).parent / "alpaca_eval")

    pending_dir     = Path(args.queue_dir) / "pending"
    in_progress_dir = Path(args.queue_dir) / "in_progress"
    done_dir        = Path(args.queue_dir) / "done"
    for d in [pending_dir, in_progress_dir, done_dir, Path(args.eval_output_dir)]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"[Daemon] started | queue={pending_dir} | tp_size={args.tp_size}", flush=True)
    print(f"[Daemon] waiting for runs: {args.run_names}", flush=True)

    while True:
        pending = sorted(pending_dir.glob("*.json"))

        if pending:
            req_file  = pending[0]
            in_prog   = in_progress_dir / req_file.name
            shutil.move(str(req_file), str(in_prog))

            req = json.loads(in_prog.read_text())
            print(f"\n[Daemon] processing {req['run_name']} step={req['step']}", flush=True)

            try:
                success = process_request(req, args)
            except Exception as e:
                print(f"[ERROR] {e}", flush=True)
                success = False

            dest = done_dir / req_file.name if success else pending_dir / ("FAILED_" + req_file.name)
            shutil.move(str(in_prog), str(dest))

        else:
            if all_sentinels_present(args.queue_dir, args.run_names):
                # One final check after grace period
                time.sleep(args.poll_interval)
                if not list(pending_dir.glob("*.json")):
                    print("[Daemon] All training runs complete and queue empty. Exiting.", flush=True)
                    break
            time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
