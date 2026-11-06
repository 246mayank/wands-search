"""Paired bootstrap over queries: is one ranking's mean really different from another's?"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RESAMPLES = 10_000
SEED = 20261106


@dataclass(frozen=True)
class Delta:
    delta: float  # mean(b) - mean(a)
    low: float
    high: float
    n: int

    @property
    def significant(self) -> bool:
        return self.low > 0 or self.high < 0


def paired_bootstrap(a: np.ndarray, b: np.ndarray, resamples: int = RESAMPLES, seed: int = SEED) -> Delta:
    """Resample whole queries, so a query's two scores stay together. 95% percentile interval."""
    if a.shape != b.shape:
        raise ValueError(f"paired arrays differ in shape: {a.shape} vs {b.shape}")
    keep = ~(np.isnan(a) | np.isnan(b))
    diffs = b[keep] - a[keep]
    n = diffs.size
    if n == 0:
        raise ValueError("no query has a score in both arrays")
    rng = np.random.default_rng(seed)
    means = diffs[rng.integers(0, n, size=(resamples, n))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return Delta(float(diffs.mean()), float(low), float(high), n)
