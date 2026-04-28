"""Wilson 95% confidence interval for a binomial proportion."""

import math


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Return (point_estimate, lower_95, upper_95). n=0 → (0,0,0)."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (p, max(0.0, center - margin), min(1.0, center + margin))
