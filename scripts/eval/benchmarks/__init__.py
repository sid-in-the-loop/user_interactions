"""Benchmark registry."""

from .alpaca_eval import AlpacaEvalBenchmark
from .arena_hard import ArenaHardBenchmark
from .mt_bench import MTBenchBenchmark
from .math500 import MATH500Benchmark
from .reasoning_gym_bench import ReasoningGymBenchmark
from .wildfeedback_held import WildFeedbackHeldOutBenchmark
from .writingbench import WritingBenchBenchmark

BENCHMARKS = {
    "alpaca_eval": AlpacaEvalBenchmark,
    "arena_hard": ArenaHardBenchmark,
    "mt_bench": MTBenchBenchmark,
    "aime": MATH500Benchmark,
    "reasoning_gym": ReasoningGymBenchmark,
    "wildfeedback_held": WildFeedbackHeldOutBenchmark,
    "writingbench": WritingBenchBenchmark,
}

ALL_BENCHMARK_NAMES = list(BENCHMARKS.keys())
