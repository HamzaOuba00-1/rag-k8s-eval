from __future__ import annotations

import math

import pytest

from ragk8s.stats import bootstrap_ci, paired_bootstrap


def test_bootstrap_ci_contains_mean_and_is_deterministic() -> None:
    values = [0, 1, 1, 0, 1, 1, 1, 0, 1, 1]
    a, b = bootstrap_ci(values), bootstrap_ci(values)
    assert a == b
    assert a.value == pytest.approx(0.7)
    assert a.low <= 0.7 <= a.high and 0.0 <= a.low < a.high <= 1.0
    assert a.n == 10


def test_bootstrap_ci_constant_and_missing_values() -> None:
    est = bootstrap_ci([1.0, None, 1.0, float("nan")])
    assert (est.value, est.low, est.high, est.n) == (1.0, 1.0, 1.0, 2)
    assert math.isnan(bootstrap_ci([]).value)


def test_paired_bootstrap_detects_consistent_improvement() -> None:
    a = [0.9, 0.8, 1.0, 0.7, 0.9, 1.0, 0.8, 0.9] * 4
    b = [x - 0.2 for x in a]
    d = paired_bootstrap(a, b)
    assert d.diff == pytest.approx(0.2)
    assert d.significant and d.p_value < 0.01


def test_paired_bootstrap_no_difference() -> None:
    a = [1, 0, 1, 0, 1, 0, 1, 0]
    b = [0, 1, 0, 1, 1, 0, 1, 0]
    d = paired_bootstrap(a, b)
    assert d.diff == 0.0 and not d.significant and d.p_value > 0.5


def test_paired_bootstrap_skips_incomplete_pairs_and_checks_length() -> None:
    assert paired_bootstrap([1, None, 1], [0, 0, None]).n == 1
    with pytest.raises(ValueError):
        paired_bootstrap([1, 2], [1])
