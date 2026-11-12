"""The decision journal: an append-only, hash-chained JSONL log, and the loop's only state.

It reuses ads-guardrail-agent's AuditLog format (seq, prev_hash, SHA-256 per line, file
lock, fsync), so `verify()` catches any edit, deletion or reorder.

Nothing else holds loop state. Each tick rebuilds what it needs (open decisions, locks,
the evidence behind every belief, the live test) by replaying the journal. That is why a
crashed tick can resume, a finished tick is a no-op, and `replay` can re-derive every past
choice from the log alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Iterator

from ads_guardrail import AuditLog



@dataclass
class DecisionRecord:
    decision_id: str
    day: date
    arm: str
    target: str
    hypothesis: str
    x: float  # the hypothesis' own estimate, $/day
    treated: list[str] = field(default_factory=list)
    outcome: str = "decided"  # decided -> applied | clamped | blocked | dry_run | halted | launched | queued
    acted: bool = False
    measure_due: date | None = None
    measured: bool = False
    y: float | None = None
    se: float | None = None
    method: str | None = None
    counted: bool = False  # used as evidence for the bandit
    details: dict[str, Any] = field(default_factory=dict)

    def mirror_row(self) -> dict[str, Any]:
        return dict(decision_id=self.decision_id, date=self.day, arm=self.arm, target=self.target,
                    hypothesis=self.hypothesis, estimate_per_day=self.x, outcome=self.outcome,
                    measured_per_day=self.y, measured_se=self.se, method=self.method)


@dataclass
class Observation:
    arm: str
    x: float
    y: float
    se: float
    decision_id: str
    seq: int


@dataclass
class JournalState:
    finished_ticks: set[date] = field(default_factory=set)
    steps_done: set[tuple[date, str]] = field(default_factory=set)
    decisions: dict[str, DecisionRecord] = field(default_factory=dict)
    observations: list[Observation] = field(default_factory=list)
    open_experiment: dict[str, Any] | None = None
    last_experiment_end: date | None = None
    changes: dict[str, list[date]] = field(default_factory=dict)  # entity_key -> days it was changed
    refreshes: dict[str, date] = field(default_factory=dict)
    halts: list[dict[str, Any]] = field(default_factory=list)
    last_tick: date | None = None

    def observations_before(self, seq: int) -> list[Observation]:
        return [o for o in self.observations if o.seq < seq]

    def open_entity_decisions(self) -> list[DecisionRecord]:
        return [d for d in self.decisions.values()
                if d.acted and not d.measured and d.arm != "landing_test" and d.treated]

    def locked_entities(self) -> set[str]:
        return {k for d in self.open_entity_decisions() for k in d.treated}

    def due(self, day: date) -> list[DecisionRecord]:
        return [d for d in self.decisions.values()
                if d.acted and not d.measured and d.measure_due is not None and d.measure_due <= day]


def tick_time(day: date, hour: int = 6) -> datetime:
    return datetime.combine(day, time(hour), tzinfo=timezone.utc)


class Journal:
    def __init__(self, path: str | Path, hour: int = 6) -> None:
        self.log = AuditLog(path)
        self.path = Path(path)
        self.hour = hour
        self._cache: list[dict[str, Any]] = []
        self._cache_size = -1

    def append(self, event: str, day: date, **fields: Any) -> dict[str, Any]:
        fresh = self._size() == self._cache_size
        rec = self.log.append(event, at=tick_time(day, self.hour), date=day.isoformat(), **fields)
        if fresh:  # keep the cache in step with the file without re-reading it
            self._cache.append(rec)
            self._cache_size = self._size()
        return rec

    def _size(self) -> int:
        return self.path.stat().st_size if self.path.exists() else 0

    def records(self) -> list[dict[str, Any]]:
        """All records. Cached; re-read whenever the file size differs from what we last saw."""
        if self._size() != self._cache_size:
            self._cache = list(self.log.records())
            self._cache_size = self._size()
        return list(self._cache)

    def verify(self) -> tuple[bool, str]:
        return self.log.verify()

    def state(self) -> JournalState:
        return replay_state(self.records())


def _d(s: str | None) -> date | None:
    return date.fromisoformat(s) if s else None


def replay_state(records: Iterator[dict[str, Any]] | list[dict[str, Any]]) -> JournalState:
    """Rebuild every piece of loop state from journal records, in order."""
    st = JournalState()
    for r in records:
        ev = r["event"]
        day = date.fromisoformat(r["date"]) if "date" in r else None
        if ev == "tick_finished":
            st.finished_ticks.add(day)
            st.last_tick = max(st.last_tick or day, day)
        elif ev == "step_done":
            st.steps_done.add((day, r["step"]))
        elif ev == "decision" and r.get("arm"):
            st.decisions[r["decision_id"]] = DecisionRecord(
                decision_id=r["decision_id"], day=day, arm=r["arm"], target=r.get("target", ""),
                hypothesis=r.get("hypothesis", ""), x=float(r["x"]), treated=list(r.get("treated", [])))
        elif ev == "acted":
            d = st.decisions[r["decision_id"]]
            d.acted = r["outcome"] not in ("blocked", "dry_run", "halted")
            d.outcome = r["outcome"]
            d.treated = list(r.get("treated", d.treated))
            d.measure_due = _d(r.get("measure_due"))
            d.details = dict(r.get("details", {}))
            if not d.acted:
                d.measured = True  # nothing changed, nothing to measure
            for k in r.get("changed", []):
                st.changes.setdefault(k, []).append(date.fromisoformat(r.get("effective", r["date"])))
            if d.arm == "creative_refresh" and d.acted:
                for k in d.treated:
                    st.refreshes[k] = day
        elif ev == "lp_launched":
            st.open_experiment = dict(key=r["key"], decision_id=r["decision_id"], started=r["started"],
                                      look_days=r["look_days"], control=r["control"], arms=r["arms"],
                                      weights=r["weights"])
        elif ev == "lp_concluded":
            st.open_experiment = None
            st.last_experiment_end = day
        elif ev == "measured":
            d = st.decisions[r["decision_id"]]
            d.measured, d.y, d.se, d.method, d.counted = True, r["y"], r["se"], r["method"], bool(r["counted"])
            if r["counted"]:
                st.observations.append(Observation(d.arm, float(r["x"]), float(r["y"]), float(r["se"]),
                                                   d.decision_id, int(r["seq"])))
        elif ev == "halted":
            st.halts.append(dict(date=r["date"], reason=r["reason"]))
    return st

