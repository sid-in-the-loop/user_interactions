"""AlpacaEval 2.0 — 805 instruction-following prompts vs GPT-4 Turbo reference."""

import json
from pathlib import Path
from vllm import SamplingParams
from .base import Benchmark


class AlpacaEvalBenchmark(Benchmark):
    name = "alpaca_eval"

    def __init__(self, reference_path="alpaca_eval_data/gpt4_turbo_reference.json"):
        self.reference_path = reference_path
        self.data = []

    def load_data(self):
        with open(self.reference_path) as f:
            self.data = json.load(f)
        print(f"  [{self.name}] Loaded {len(self.data)} prompts")

    def format_prompts(self):
        return [
            [{"role": "user", "content": d["instruction"]}]
            for d in self.data
        ]

    def sampling_params(self):
        return SamplingParams(temperature=0.7, max_tokens=2048, skip_special_tokens=True)

    def save_outputs(self, vllm_outputs, path: Path):
        records = []
        for d, o in zip(self.data, vllm_outputs):
            text = o.outputs[0].text
            records.append({
                "instruction": d["instruction"],
                "output": text,
                "dataset": d.get("dataset", ""),
                "generator": "checkpoint",
            })
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(records, f, indent=2)

    def judge(self, outputs_path, scores_path, **kwargs):
        """Pairwise judging vs GPT-4 Turbo reference. Implemented in judge_all.py."""
        raise NotImplementedError("Use judge_all.py for API judging")
