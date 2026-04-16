"""Reasoning Gym — 3 tasks × 200 problems, programmatic verification."""

import json
from pathlib import Path
from vllm import SamplingParams
from .base import Benchmark

REASONING_SYSTEM = "Solve this reasoning problem. Show your work and provide your final answer clearly."

TASKS = ["knights_knaves", "binary_matrix", "shortest_path"]
PROBLEMS_PER_TASK = 200
SEED = 42


class ReasoningGymBenchmark(Benchmark):
    name = "reasoning_gym"

    def __init__(self, problems_path="data/benchmark_data/reasoning_gym/problems.json"):
        self.problems_path = problems_path
        self.data = []

    def load_data(self):
        """Load pre-generated problems. If not cached, generate them."""
        path = Path(self.problems_path)
        if path.exists():
            with open(path) as f:
                self.data = json.load(f)
        else:
            self.data = self._generate_problems()
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(self.data, f, indent=2)
        print(f"  [{self.name}] Loaded {len(self.data)} problems ({len(TASKS)} tasks)")

    def _generate_problems(self):
        """Generate problems using reasoning-gym package."""
        try:
            import reasoning_gym
        except ImportError:
            raise ImportError("pip install reasoning-gym")

        problems = []
        for task in TASKS:
            data = reasoning_gym.create_dataset(task, size=PROBLEMS_PER_TASK, seed=SEED)
            for i, entry in enumerate(data):
                problems.append({
                    "task": task,
                    "problem_index": i,
                    "problem": entry["question"],
                    "answer": entry["answer"],
                    "seed": SEED,
                })
        return problems

    def format_prompts(self):
        return [
            [
                {"role": "system", "content": REASONING_SYSTEM},
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
                "task": d["task"],
                "problem_index": d["problem_index"],
                "problem": d["problem"],
                "output": text,
                "seed": d["seed"],
                "generator": "checkpoint",
            })
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(records, f, indent=2)

    def judge(self, outputs_path, scores_path, **kwargs):
        """Programmatic evaluation using reasoning-gym score_answer()."""
        try:
            import reasoning_gym
        except ImportError:
            raise ImportError("pip install reasoning-gym")

        with open(outputs_path) as f:
            outputs = json.load(f)

        # Build datasets
        datasets = {}
        for task in TASKS:
            datasets[task] = reasoning_gym.create_dataset(task, size=PROBLEMS_PER_TASK, seed=SEED)

        results_by_task = {task: {"correct": 0, "total": 0} for task in TASKS}
        results = []

        for item in outputs:
            task = item["task"]
            idx = item["problem_index"]
            dataset = datasets[task]
            entry = dataset[idx]

            try:
                score = dataset.score_answer(answer=item["output"], entry=entry)
                is_correct = score >= 1.0
            except Exception:
                is_correct = False

            results_by_task[task]["total"] += 1
            if is_correct:
                results_by_task[task]["correct"] += 1
            results.append({"task": task, "index": idx, "correct": is_correct})

        # Compute per-task accuracy
        task_scores = {}
        for task, counts in results_by_task.items():
            acc = counts["correct"] / counts["total"] * 100 if counts["total"] > 0 else 0
            task_scores[task] = {"accuracy": acc, **counts}

        total_correct = sum(c["correct"] for c in results_by_task.values())
        total = sum(c["total"] for c in results_by_task.values())

        scores = {
            "overall_accuracy": total_correct / total * 100 if total > 0 else 0,
            "total_correct": total_correct,
            "total": total,
            "per_task": task_scores,
            "results": results,
        }

        scores_path.parent.mkdir(parents=True, exist_ok=True)
        with open(scores_path, "w") as f:
            json.dump(scores, f, indent=2)

        return scores
