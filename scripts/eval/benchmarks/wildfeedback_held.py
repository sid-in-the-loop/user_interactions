"""WildFeedback held-out — 500 samples NOT in training, model(x) vs GPT-4(y)."""

import json
import random
from pathlib import Path
from vllm import SamplingParams
from .base import Benchmark


def get_content(obj):
    if obj is None: return ""
    if isinstance(obj, str): return obj
    if isinstance(obj, dict): return obj.get("content") or ""
    return str(obj)


class WildFeedbackHeldOutBenchmark(Benchmark):
    name = "wildfeedback_held"

    def __init__(
        self,
        full_data_path="datasets/wildfeedback/filtered_BEST.jsonl",
        training_data_path="datasets/wildchat/ystar_prefix_wildchat_qwen3_8b.jsonl",
        held_out_ids_path="data/benchmark_data/wildfeedback_held/held_out_ids.json",
        n_samples=500,
        seed=42,
    ):
        self.full_data_path = full_data_path
        self.training_data_path = training_data_path
        self.held_out_ids_path = held_out_ids_path
        self.n_samples = n_samples
        self.seed = seed
        self.data = []

    def load_data(self):
        ids_path = Path(self.held_out_ids_path)
        if ids_path.exists():
            with open(ids_path) as f:
                held_out_ids = set(tuple(x) for x in json.load(f))
            # Load full data and filter to held-out
            all_data = []
            with open(self.full_data_path) as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        key = (row.get("conversation_id", ""), row.get("turn_index"))
                        if key in held_out_ids:
                            all_data.append(row)
            self.data = all_data[:self.n_samples]
        else:
            # Compute held-out set
            self._compute_held_out()

        print(f"  [{self.name}] Loaded {len(self.data)} held-out samples")

    def _compute_held_out(self):
        """Compute held-out IDs by diffing full dataset vs training set."""
        # Training set IDs
        training_ids = set()
        with open(self.training_data_path) as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    training_ids.add((row.get("conversation_id", ""), row.get("turn_index")))

        # Full dataset minus training
        candidates = []
        with open(self.full_data_path) as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    key = (row.get("conversation_id", ""), row.get("turn_index"))
                    if key not in training_ids:
                        candidates.append(row)

        random.seed(self.seed)
        if len(candidates) > self.n_samples:
            self.data = random.sample(candidates, self.n_samples)
        else:
            self.data = candidates

        # Save held-out IDs for reproducibility
        ids_path = Path(self.held_out_ids_path)
        ids_path.parent.mkdir(parents=True, exist_ok=True)
        ids = [(row.get("conversation_id", ""), row.get("turn_index")) for row in self.data]
        with open(ids_path, "w") as f:
            json.dump(ids, f)

        print(f"  [{self.name}] Computed {len(self.data)} held-out samples "
              f"({len(candidates)} candidates, {len(training_ids)} training)")

    def format_prompts(self):
        """Model sees only x (conversation history)."""
        prompts = []
        for row in self.data:
            x = row.get("x", [])
            if isinstance(x, list):
                msgs = [{"role": m["role"], "content": m["content"]} for m in x]
            else:
                msgs = [{"role": "user", "content": str(x)}]
            prompts.append(msgs)
        return prompts

    def sampling_params(self):
        return SamplingParams(temperature=0.7, max_tokens=2048, skip_special_tokens=True)

    def save_outputs(self, vllm_outputs, path: Path):
        records = []
        for row, o in zip(self.data, vllm_outputs):
            text = o.outputs[0].text
            records.append({
                "conversation_id": row.get("conversation_id", ""),
                "turn_index": row.get("turn_index"),
                "x": row.get("x", []),
                "model_output": text,
                "y_gpt4": get_content(row.get("y", "")),
                "o_oracle": get_content(row.get("o", "")),
                "generator": "checkpoint",
            })
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(records, f, indent=2)

    def judge(self, outputs_path, scores_path, **kwargs):
        raise NotImplementedError("Use judge_all.py for API judging")
