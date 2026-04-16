"""AIME 2025+2026 — 60 competition math problems with programmatic evaluation."""

import json
import re
from pathlib import Path
from vllm import SamplingParams
from .base import Benchmark

MATH_SYSTEM = (
    "Solve the following math problem step by step. "
    "Put your final numerical answer in \\boxed{}. "
    "The answer is an integer between 0 and 999."
)


def extract_boxed(text):
    """Extract the last \\boxed{...} from model output."""
    # Find all boxed expressions (handle nested braces)
    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    return matches[-1].strip() if matches else ""


def normalize_answer(answer):
    """Normalize math answer for comparison."""
    answer = answer.strip()
    # Remove surrounding $ signs
    answer = answer.strip("$")
    # Remove spaces
    answer = answer.replace(" ", "")
    # Normalize fractions
    answer = answer.replace("\\frac", "\\frac")
    return answer


def answers_match(predicted, gold):
    """Check if predicted answer matches gold. For AIME, answers are integers 0-999."""
    if not predicted or not gold:
        return False
    p = normalize_answer(predicted)
    g = normalize_answer(gold)
    if p == g:
        return True
    # Try integer comparison (AIME answers are integers)
    try:
        return int(float(p)) == int(float(g))
    except (ValueError, TypeError):
        pass
    return False


class MATH500Benchmark(Benchmark):
    name = "aime"

    def __init__(self, data_paths=None):
        self.data_paths = data_paths or [
            "data/benchmark_data/math/aime25.jsonl",
            "data/benchmark_data/math/aime26.jsonl",
        ]
        self.data = []

    def load_data(self):
        for path in self.data_paths:
            with open(path) as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        row["source"] = Path(path).stem  # aime25 or aime26
                        self.data.append(row)
        print(f"  [{self.name}] Loaded {len(self.data)} problems from {len(self.data_paths)} files")

    def format_prompts(self):
        return [
            [
                {"role": "system", "content": MATH_SYSTEM},
                {"role": "user", "content": d["problem"]},
            ]
            for d in self.data
        ]

    def sampling_params(self):
        return SamplingParams(temperature=0.0, max_tokens=2048, skip_special_tokens=True)

    def save_outputs(self, vllm_outputs, path: Path):
        records = []
        for d, o in zip(self.data, vllm_outputs):
            text = o.outputs[0].text
            records.append({
                "problem": d["problem"],
                "output": text,
                "gold_answer": str(d.get("answer", "")),
                "id": d.get("id", ""),
                "source": d.get("source", ""),
                "generator": "checkpoint",
            })
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(records, f, indent=2)

    def judge(self, outputs_path, scores_path, **kwargs):
        """Programmatic evaluation — no API needed."""
        with open(outputs_path) as f:
            outputs = json.load(f)

        correct, total = 0, 0
        results = []
        for item in outputs:
            predicted = extract_boxed(item["output"])
            gold = extract_boxed(item["gold_answer"]) or item["gold_answer"]
            is_correct = answers_match(predicted, gold)
            if is_correct:
                correct += 1
            total += 1
            results.append({
                "predicted": predicted,
                "gold": gold,
                "correct": is_correct,
            })

        scores = {
            "accuracy": correct / total * 100 if total > 0 else 0,
            "correct": correct,
            "total": total,
            "results": results,
        }

        scores_path.parent.mkdir(parents=True, exist_ok=True)
        with open(scores_path, "w") as f:
            json.dump(scores, f, indent=2)

        return scores
