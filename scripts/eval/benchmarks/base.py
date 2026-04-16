"""Abstract base class for all evaluation benchmarks."""

from abc import ABC, abstractmethod
from pathlib import Path
from vllm import SamplingParams


class Benchmark(ABC):
    """
    Each benchmark implements:
      - load_data(): load prompts/problems into memory
      - format_prompts(): return list of chat message lists for vLLM.chat()
      - save_outputs(): post-process vLLM outputs and save as JSON
      - sampling_params(): benchmark-specific generation params
      - judge(): score outputs (API or programmatic)

    Multi-turn benchmarks override needs_multi_turn() and provide
    format_turn1_prompts(), format_turn2_prompts(), save_multiturn_outputs().
    """

    name: str = ""

    @abstractmethod
    def load_data(self) -> None:
        """Load prompts/problems from disk or HF."""

    @abstractmethod
    def format_prompts(self) -> list[list[dict]]:
        """Return list of chat message lists for vLLM.chat()."""

    @abstractmethod
    def save_outputs(self, vllm_outputs: list, path: Path) -> None:
        """Post-process vLLM outputs and save as JSON."""

    def sampling_params(self) -> SamplingParams:
        """Override for benchmark-specific params."""
        return SamplingParams(temperature=0.7, max_tokens=2048, skip_special_tokens=True)

    def needs_multi_turn(self) -> bool:
        return False

    @abstractmethod
    def judge(self, outputs_path: Path, scores_path: Path, **kwargs) -> dict:
        """Score outputs. Returns dict with metric name → value."""

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"
