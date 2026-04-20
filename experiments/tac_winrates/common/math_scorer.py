"""math_verify-based scorer. Wraps gold answer in \\boxed{} and returns 0/1.

Based on the pattern supplied by the user:
    ground_truth_boxed = f"\\boxed{{{ground_truth}}}"
    score, _ = math_metric(...)(ground_truth_boxed, model_output)
"""

from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig


class MathVerifyScorer:
    """One instance per process (math_metric holds state)."""

    def __init__(self):
        self._verify_func = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        )

    def score(self, model_output: str, ground_truth: str, timeout_score: float = 0.0) -> float:
        gt_boxed = f"\\boxed{{{ground_truth}}}"
        try:
            s, _ = self._verify_func([gt_boxed], [model_output])
            return float(s)
        except TimeoutException:
            return float(timeout_score)
        except Exception:
            return 0.0
