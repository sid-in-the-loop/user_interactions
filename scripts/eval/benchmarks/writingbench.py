"""WritingBench — writing quality evaluation with rubric-based scoring."""

import json
from pathlib import Path
from vllm import SamplingParams
from .base import Benchmark


class WritingBenchBenchmark(Benchmark):
    name = "writingbench"

    def __init__(self, data_path="data/benchmark_data/writingbench/benchmark_all.jsonl"):
        self.data_path = data_path
        self.data = []

    def load_data(self):
        if self.data_path.endswith(".jsonl"):
            with open(self.data_path) as f:
                self.data = [json.loads(l) for l in f if l.strip()]
        else:
            with open(self.data_path) as f:
                self.data = json.load(f)
        print(f"  [{self.name}] Loaded {len(self.data)} writing tasks")

    def format_prompts(self, max_chars=40000):
        """Format prompts, truncating very long queries to fit context window."""
        prompts = []
        for d in self.data:
            text = d.get("query", d.get("prompt", ""))
            if len(text) > max_chars:
                text = text[:max_chars] + "\n\n[Content truncated for length]"
            prompts.append([{"role": "user", "content": text}])
        return prompts

    def sampling_params(self):
        return SamplingParams(temperature=0.7, max_tokens=4096, skip_special_tokens=True)

    def save_outputs(self, vllm_outputs, path: Path):
        records = []
        for d, o in zip(self.data, vllm_outputs):
            text = o.outputs[0].text
            records.append({
                "index": d.get("index", ""),
                "query": d.get("query", d.get("prompt", "")),
                "output": text,
                "category": d.get("category", d.get("domain", "")),
                "generator": "checkpoint",
            })
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(records, f, indent=2)

    def judge(self, outputs_path, scores_path, **kwargs):
        raise NotImplementedError("Use judge_all.py for rubric-based judging")
