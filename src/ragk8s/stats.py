"""Bootstrap confidence intervals.

With ~55 evaluation questions, point estimates alone are not enough to rank
configurations. Two tools are used throughout the evaluation:

- ``bootstrap_ci``: percentile interval of a mean over questions.
- ``paired_bootstrap``: interval of the difference between two configurations
  scored on the *same* questions. Resampling questions jointly keeps the pairing,
  which is much more sensitive than comparing two independent intervals.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from . import config

N_RESAMPLES = 10_000


@dataclass(frozen=True)
class Estimate:
    value: float
    low: float
    high: float
    n: int

    def fmt(self, digits: int = 3) -> str:
        return f"{self.value:.{digits}f} [{self.low:.{digits}f}, {self.high:.{digits}f}]"


@dataclass(frozen=True)
class PairedDifference:
    diff: float
    """mean(a) - mean(b)"""
    low: float
    high: float
    p_value: float
    """Two-sided bootstrap p-value for a zero difference."""
    n: int

    @property
    def significant(self) -> bool:
        return self.low > 0 or self.high < 0


def _clean(values: Sequence[float | None]) -> np.ndarray:
    arr = np.array([np.nan if v is None else float(v) for v in values], dtype=float)
    return arr[~np.isnan(arr)]


def bootstrap_ci(
    values: Sequence[float | None],
    n_resamples: int = N_RESAMPLES,
    alpha: float = 0.05,
    seed: int = config.SEED,
) -> Estimate:
    """Mean and percentile bootstrap interval. ``None``/NaN values are dropped."""
    arr = _clean(values)
    if arr.size == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, arr.size, size=(n_resamples, arr.size))].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return Estimate(float(arr.mean()), float(low), float(high), int(arr.size))


def paired_bootstrap(
    a: Sequence[float | None],
    b: Sequence[float | None],
    n_resamples: int = N_RESAMPLES,
    alpha: float = 0.05,
    seed: int = config.SEED,
) -> PairedDifference:
    """Bootstrap of mean(a - b) over questions scored in both ``a`` and ``b``."""
    if len(a) != len(b):
        raise ValueError("paired samples must have the same length")
    pairs = [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None]
    diffs = np.array([float(x) - float(y) for x, y in pairs], dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    if diffs.size == 0:
        return PairedDifference(float("nan"), float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    means = diffs[rng.integers(0, diffs.size, size=(n_resamples, diffs.size))].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    tail = min(float((means <= 0).mean()), float((means >= 0).mean()))
    return PairedDifference(float(diffs.mean()), float(low), float(high), min(1.0, 2 * tail), int(diffs.size))
