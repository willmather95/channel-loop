"""HOME: how the loop lives on a schedule.

- `catch_up(...)` runs every missing tick in date order. A scheduler that missed three
  days runs three ticks; one that fires twice runs zero extra (finished ticks are no-ops).
- A process lock (`flock` on `<state>/runner.lock`) stops two schedulers, say cron and a
  long-running runner, from ticking the same state at once. The second one exits with
  code 5 and touches nothing.
- `serve(...)` is a simple long-running runner: catch up, sleep, repeat. In SIMULATED mode
  each wake-up advances the simulated calendar by one day.

The state directory holds `loop-meta.json` (seed, start date, horizon, policy) so a
cron line only needs `--state`.
"""
from __future__ import annotations

import fcntl
import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Iterator

from ..config import LoopConfig
from ..loop import Loop, TickResult, load_guard_config
from ..sim.source import SimSource


class RunnerBusy(RuntimeError):
    pass


@dataclass(frozen=True)
class Meta:
    seed: int
    start: str
    horizon_days: int
    policy: str
    guard_config: str
    brief: str

    @classmethod
    def load(cls, state: Path) -> "Meta":
        return cls(**json.loads((state / "loop-meta.json").read_text()))

    def save(self, state: Path) -> None:
        state.mkdir(parents=True, exist_ok=True)
        (state / "loop-meta.json").write_text(json.dumps(asdict(self), indent=2) + "\n")

    @property
    def start_date(self) -> date:
        return date.fromisoformat(self.start)

    @property
    def end_date(self) -> date:
        return self.start_date + timedelta(days=self.horizon_days - 1)


@contextmanager
def runner_lock(state: Path) -> Iterator[None]:
    state.mkdir(parents=True, exist_ok=True)
    fd = os.open(state / "runner.lock", os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunnerBusy(f"another runner holds {state / 'runner.lock'}") from exc
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def build_loop(state: Path, cfg: LoopConfig, meta: Meta, build_sites: bool = False) -> Loop:
    guard = load_guard_config(meta.guard_config)
    src = SimSource(state, meta.seed, meta.start_date, meta.horizon_days, cfg.agent.value_per_conversion)
    return Loop(state, cfg.with_policy(meta.policy), guard, meta.brief, seed=meta.seed, source=src,
                build_sites=build_sites)


def catch_up(state: Path, cfg: LoopConfig, through: date, on_tick: Callable[[TickResult], None] | None = None,
             build_sites: bool = False) -> list[TickResult]:
    meta = Meta.load(state)
    through = min(through, meta.end_date)
    out: list[TickResult] = []
    with runner_lock(state):
        loop = build_loop(state, cfg, meta, build_sites)
        js = loop.journal.state()
        day = meta.start_date if js.last_tick is None else js.last_tick + timedelta(days=1)
        while day <= through:
            r = loop.tick(day)
            out.append(r)
            if on_tick:
                on_tick(r)
            day += timedelta(days=1)
    return out


def serve(state: Path, cfg: LoopConfig, interval_s: float, max_wakeups: int | None = None,
          on_tick: Callable[[TickResult], None] | None = None) -> int:
    """Long-running runner for the SIMULATED calendar: one simulated day per wake-up."""
    meta = Meta.load(state)
    wakeups = 0
    while max_wakeups is None or wakeups < max_wakeups:
        loop = build_loop(state, cfg, meta)
        last = loop.journal.state().last_tick
        nxt = meta.start_date if last is None else last + timedelta(days=1)
        if nxt > meta.end_date:
            return 0
        catch_up(state, cfg, nxt, on_tick)
        wakeups += 1
        if max_wakeups is None or wakeups < max_wakeups:
            time.sleep(interval_s)
    return 0
