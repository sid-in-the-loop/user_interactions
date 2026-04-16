"""MT-Bench — 80 multi-turn questions, 2 turns each."""

import json
from pathlib import Path
from vllm import SamplingParams
from .base import Benchmark


class MTBenchBenchmark(Benchmark):
    name = "mt_bench"

    def __init__(self, questions_path="data/benchmark_data/mt_bench/question.jsonl"):
        self.questions_path = questions_path
        self.data = []

    def load_data(self):
        with open(self.questions_path) as f:
            self.data = [json.loads(l) for l in f if l.strip()]
        print(f"  [{self.name}] Loaded {len(self.data)} questions (2 turns each)")

    def needs_multi_turn(self):
        return True

    def format_prompts(self):
        """Not used directly — use format_turn1_prompts / format_turn2_prompts."""
        return self.format_turn1_prompts()

    def format_turn1_prompts(self):
        return [
            [{"role": "user", "content": q["turns"][0]}]
            for q in self.data
        ]

    def format_turn2_prompts(self, turn1_responses):
        """Build turn-2 prompts using turn-1 model responses."""
        prompts = []
        for q, t1_resp in zip(self.data, turn1_responses):
            prompts.append([
                {"role": "user", "content": q["turns"][0]},
                {"role": "assistant", "content": t1_resp},
                {"role": "user", "content": q["turns"][1]},
            ])
        return prompts

    def sampling_params(self):
        return SamplingParams(temperature=0.0, max_tokens=2048, skip_special_tokens=True)

    def save_outputs(self, vllm_outputs, path):
        """Not used for multi-turn."""
        pass

    def save_multiturn_outputs(self, turn1_texts, turn2_texts, path: Path):
        records = []
        for q, t1, t2 in zip(self.data, turn1_texts, turn2_texts):
            records.append({
                "question_id": q["question_id"],
                "category": q.get("category", ""),
                "turns": [t1, t2],
                "reference": q.get("reference", []),
                "generator": "checkpoint",
            })
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(records, f, indent=2)

    def judge(self, outputs_path, scores_path, **kwargs):
        raise NotImplementedError("Use judge_all.py for API judging")
