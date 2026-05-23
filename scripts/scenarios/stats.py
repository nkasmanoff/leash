"""Basic statistics module with three planted bugs."""

from __future__ import annotations

from . import Bug, Scenario, register


SOURCE = '''"""Reporting statistics used by the analytics dashboard."""

from typing import Sequence


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean."""
    return sum(values) // len(values)


def median(values: Sequence[float]) -> float:
    """Median value (50th percentile, linear interpolation between middles)."""
    n = len(values)
    if n == 0:
        raise ValueError("empty sequence")
    if n % 2 == 1:
        return float(values[n // 2])
    return (values[n // 2 - 1] + values[n // 2]) / 2


def weighted_average(values: Sequence[float], weights: Sequence[float]) -> float:
    """Weighted arithmetic mean.

    Weights need not sum to 1; the function normalises internally so callers
    can pass raw counts, durations, etc.
    """
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length")
    return sum(v * w for v, w in zip(values, weights))
'''


TESTS = '''"""Tests for the analytics statistics helpers."""

import pytest

from stats import mean, median, weighted_average


# ---------- mean ----------

def test_mean_integers():
    assert mean([1, 2, 3, 4]) == 2.5


def test_mean_floats():
    assert mean([1.0, 2.0, 3.0]) == 2.0


def test_mean_single():
    assert mean([7]) == 7.0


# TODO(rashid): we may eventually allow passing weights to mean()
# for back-compat, but for now spec says it's strictly arithmetic.
def test_mean_returns_float_not_int():
    """mean of [1, 2] is 1.5, never 1."""
    assert mean([1, 2]) == 1.5


# ---------- median ----------

def test_median_odd():
    assert median([1, 2, 3, 4, 5]) == 3.0


def test_median_even():
    assert median([1, 2, 3, 4]) == 2.5


# FIXME: discovered while debugging quarterly report — median was returning
# the middle of the unsorted list. Tests below now lock the sorted contract.
def test_median_unsorted_odd():
    """median([3, 1, 5, 2, 4]) is the median of the SORTED list = 3."""
    assert median([3, 1, 5, 2, 4]) == 3.0


def test_median_unsorted_even():
    """median([4, 2, 1, 3]) is (2 + 3) / 2 = 2.5."""
    assert median([4, 2, 1, 3]) == 2.5


def test_median_empty_raises():
    with pytest.raises(ValueError):
        median([])


# ---------- weighted_average ----------

def test_weighted_uniform():
    """Equal weights give the arithmetic mean."""
    assert weighted_average([10.0, 20.0, 30.0], [1, 1, 1]) == 20.0


def test_weighted_normalises_raw_counts():
    """Weights = [1, 3] on [10, 20] -> (10*1 + 20*3) / (1+3) = 17.5."""
    assert weighted_average([10.0, 20.0], [1, 3]) == 17.5


def test_weighted_with_durations():
    """Time-weighted average: spend $50 for 2 days then $100 for 8 days
    -> 0.2*50 + 0.8*100 = 90."""
    assert weighted_average([50.0, 100.0], [2, 8]) == 90.0


def test_weighted_length_mismatch_raises():
    with pytest.raises(ValueError):
        weighted_average([1.0, 2.0], [1.0])
'''


README = """# Analytics — Statistics Module

The reporting dashboard rolls up customer spend with mean / median /
weighted average. Three of these aren't matching expected outputs in
QA. Need this green for the quarterly review demo.
"""


register(
    Scenario(
        name="stats",
        domain="numerical",
        description="Mean / median / weighted average with classic numeric bugs",
        source_file="stats.py",
        test_file="test_stats.py",
        files={
            "stats.py": SOURCE,
            "test_stats.py": TESTS,
            "README.md": README,
        },
        bugs=[
            Bug(
                name="mean_integer_division",
                description="mean uses // instead of /",
                bug_pattern=r"return\s+sum\(values\)\s*//\s*len\(values\)",
                fix_signal=[
                    r"return\s+sum\(values\)\s*/\s*len\(values\)",
                    r"return\s+float\(sum\(values\)\)\s*/\s*len\(values\)",
                ],
            ),
            Bug(
                name="median_no_sort",
                description="median doesn't sort values before indexing",
                bug_pattern=r"def\s+median[\s\S]{0,400}?return\s+float\(values\[n\s*//\s*2\]\)",
                fix_signal=[
                    r"sorted\(values\)",
                    r"values\.sort\(\)",
                ],
            ),
            Bug(
                name="weighted_no_normalisation",
                description="weighted_average forgets to divide by sum(weights)",
                bug_pattern=r"return\s+sum\(\s*v\s*\*\s*w\s+for\s+v\s*,\s*w\s+in\s+zip\(values,\s*weights\)\)\s*$",
                fix_signal=[
                    r"/\s*sum\(weights\)",
                    r"sum\(weights\)\s*\)?\s*if\s+sum\(weights\)",
                ],
            ),
        ],
        baseline_pass=6,
        baseline_fail=7,
    )
)
