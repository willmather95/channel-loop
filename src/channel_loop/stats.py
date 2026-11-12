"""Small, dependency-light statistics for comparing policies across seeds."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairedComparison:
    n: int
    mean: float
    ci_low: float
    ci_high: float
    p_value: float  # one-sided, H1: mean difference > 0
    wins: int  # seeds where the difference is positive

    def to_dict(self) -> dict:
        return {"n": self.n, "mean": round(self.mean, 2), "ci95": [round(self.ci_low, 2), round(self.ci_high, 2)],
                "p_one_sided": self.p_value, "wins": self.wins}


def bootstrap_ci(x: np.ndarray, reps: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    means = x[rng.integers(0, len(x), size=(reps, len(x)))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def sign_flip_p(diff: np.ndarray, reps: int = 50_000, seed: int = 1) -> float:
    """One-sided paired permutation test: under H0 each seed's difference is equally likely +/-.

    Exact enumeration up to 16 seeds, Monte Carlo above (with the +1 correction).
    """
    d = np.asarray(diff, dtype=float)
    obs = d.mean()
    n = len(d)
    if n <= 16:
        signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
        return float(((signs * d).mean(axis=1) >= obs - 1e-12).mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(reps, n))
    return float((((signs * d).mean(axis=1) >= obs - 1e-12).sum() + 1) / (reps + 1))


def compare(diff: np.ndarray) -> PairedComparison:
    d = np.asarray(diff, dtype=float)
    lo, hi = bootstrap_ci(d)
    return PairedComparison(len(d), float(d.mean()), lo, hi, sign_flip_p(d), int((d > 0).sum()))


def mean_band(rows: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-column mean and 95% bootstrap band across seeds (rows)."""
    rows = np.asarray(rows, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, rows.shape[0], size=(4000, rows.shape[0]))
    boots = rows[idx].mean(axis=1)
    return rows.mean(axis=0), np.quantile(boots, 0.025, axis=0), np.quantile(boots, 0.975, axis=0)


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 3:
        return float("nan")
    a, b = np.asarray(x), np.asarray(y)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def fmt_money(v: float) -> str:
    s = "-" if v < 0 else ""
    return f"{s}${abs(v):,.0f}"


def nan_to_none(v: float) -> float | None:
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v
