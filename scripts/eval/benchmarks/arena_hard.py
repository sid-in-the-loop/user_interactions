"""Arena-Hard v2.0 — 500 hard + 250 creative writing prompts."""

import json
from pathlib import Path
from vllm import SamplingParams
from .base import Benchmark


class ArenaHardBenchmark(Benchmark):
    name = "arena_hard"

    def __init__(self, questions_path="data/benchmark_data/arena_hard/question.jsonl"):
        self.questions_path = questions_path
        self.data = []

    def load_data(self):
        with open(self.questions_path) as f:
            self.data = [json.loads(l) for l in f if l.strip()]
        print(f"  [{self.name}] Loaded {len(self.data)} prompts")

    def format_prompts(self, max_chars=30000):
        prompts = []
        for q in self.data:
            turns = q.get("turns", [])
            if turns:
                text = turns[0]
                if len(text) > max_chars:
                    text = text[:max_chars] + "\n\n[Content truncated for length]"
                prompts.append([{"role": "user", "content": text}])
            else:
                prompts.append([{"role": "user", "content": q.get("prompt", "")[:max_chars]}])
        return prompts

    def sampling_params(self):
        return SamplingParams(temperature=0.0, max_tokens=4096, skip_special_tokens=True)

    def truncate_long_prompts(self, tokenizer, max_input_tokens=12000):
        """Truncate prompts that would exceed model context. Call after load_data."""
        truncated = 0
        for q in self.data:
            turns = q.get("turns", [])
            if turns:
                text = turns[0]
                tokens = tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) > max_input_tokens:
                    q["turns"] = [tokenizer.decode(tokens[:max_input_tokens], skip_special_tokens=True)]
                    truncated += 1
        if truncated:
            print(f"  [{self.name}] Truncated {truncated} long prompts")

    def save_outputs(self, vllm_outputs, path: Path):
        records = []
        for q, o in zip(self.data, vllm_outputs):
            text = o.outputs[0].text
            records.append({
                "question_id": q.get("question_id", q.get("id", "")),
                "category": q.get("category", q.get("cluster", "")),
                "turns": q.get("turns", []),
                "output": text,
                "generator": "checkpoint",
            })
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(records, f, indent=2)

    def judge(self, outputs_path, scores_path, **kwargs):
        raise NotImplementedError("Use judge_all.py for API judging")
