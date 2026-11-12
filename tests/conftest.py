from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pytest

from channel_loop.config import load_loop_config
from channel_loop.experiment import START
from channel_loop.loop import Loop, load_guard_config
from channel_loop.sim.source import SimSource

ROOT = Path(__file__).resolve().parents[1]
GUARD = str(ROOT / "config" / "guardrail.toml")
BRIEF = str(ROOT / "config" / "brief.yaml")
LOOPCFG = str(ROOT / "config" / "loop.toml")


def make_loop(root: Path, seed: int = 3, policy: str = "loop", guard=None, cfg=None, weeks: int = 12) -> Loop:
    cfg = (cfg or load_loop_config(LOOPCFG)).with_policy(policy)
    guard = guard or load_guard_config(GUARD)
    src = SimSource(root, seed, START, weeks * 7, cfg.agent.value_per_conversion)
    return Loop(root, cfg, guard, BRIEF, seed=seed, source=src)


def run_days(loop: Loop, n: int, start: date = START) -> list:
    return [loop.tick(start + timedelta(days=i)) for i in range(n)]


@pytest.fixture(scope="session")
def five_week_runs(tmp_path_factory):
    """Loop and random policy, two seeds each, 5 simulated weeks. Shared by the end-to-end tests."""
    out = {}
    for policy in ("loop", "random"):
        for seed in (3, 11):
            root = tmp_path_factory.mktemp(f"{policy}-{seed}")
            lp = make_loop(root, seed, policy)
            run_days(lp, 35)
            out[(policy, seed)] = lp
    return out


@pytest.fixture
def guard():
    return load_guard_config(GUARD)


@pytest.fixture
def cfg():
    return load_loop_config(LOOPCFG)


def no_writes(g):
    return replace(g, writes_enabled=False)
