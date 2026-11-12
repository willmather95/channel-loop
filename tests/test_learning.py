"""The loop learns: a statistical test across seeds, on the SIMULATED market.

Eight seeds, three policies, 12 weeks each, run through the real ticks, guardrails and
attribution. Pairs share a seed, so they face the same hidden market and the same noise.
One-sided sign-flip permutation tests, exact for 8 seeds (smallest possible p = 1/256).

What is asserted is only what the full 30-seed run in the README supports:
1. the loop earns more than doing nothing,
2. each ad change the loop makes is worth more, by the hidden truth, than a random one,
3. the loop's beliefs move toward the hidden truth for the arms its estimates get most wrong.
Total profit vs random is NOT asserted: over 12 weeks it is a tie inside the noise.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from channel_loop.config import load_loop_config
from channel_loop.experiment import AD_ARMS, run_matrix
from channel_loop.stats import compare

from conftest import BRIEF, GUARD, LOOPCFG

SEEDS = list(range(1, 9))


@pytest.fixture(scope="module")
def runs():
    res = run_matrix(SEEDS, ("loop", "random", "static"), load_loop_config(LOOPCFG), GUARD, BRIEF,
                     workers=min(8, os.cpu_count() or 2))
    return {(r["policy"], r["seed"]): r for r in res}


def _per_change(r):
    t = [d["truth"] for d in r["decisions"] if d["arm"] in AD_ARMS and d["truth"] is not None]
    return float(np.mean(t))


@pytest.mark.slow
def test_loop_beats_doing_nothing(runs):
    diff = np.array([runs[("loop", s)]["total_profit"] - runs[("static", s)]["total_profit"] for s in SEEDS])
    c = compare(diff)
    assert c.p_value < 0.01, c
    assert c.ci_low > 0


@pytest.mark.slow
def test_loop_picks_better_ad_changes_than_random(runs):
    diff = np.array([_per_change(runs[("loop", s)]) - _per_change(runs[("random", s)]) for s in SEEDS])
    c = compare(diff)
    assert c.p_value < 0.01, c
    assert c.ci_low > 0


@pytest.mark.slow
@pytest.mark.parametrize("arm,x,direction", [
    ("budget_shift", 39.0, -1),   # estimates run high (true value is about a third)
    ("bid_up", 27.0, -1),         # estimates say +$27/day; the true effect is about zero
    ("creative_refresh", 46.0, +1),  # estimates run low (true value is about 3x)
])
def test_beliefs_move_toward_the_hidden_truth(runs, arm, x, direction):
    """Predicted effect for a typical estimate x, after 12 weeks of evidence, vs the prior's x."""
    learned = []
    for s in SEEDS:
        b = runs[("loop", s)]["beliefs"][arm]
        if b["n"] >= 1:
            learned.append((b["w0"] + b["w1"] * x / 100.0) * 100.0)
    assert len(learned) >= 6
    moved = [direction * (v - x) > 0 for v in learned]
    assert np.mean(moved) >= 0.7
    assert direction * (np.mean(learned) - x) > 10


@pytest.mark.slow
def test_every_journal_in_the_matrix_is_intact(runs):
    assert all(r["journal_ok"] for r in runs.values())
