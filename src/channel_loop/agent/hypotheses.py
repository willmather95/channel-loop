"""Observe and hypothesize: read the warehouse, then write one concrete, testable change per arm.

Each candidate carries the agent's own estimate of its effect in dollars of profit per day.
The estimates use deliberately simple, explainable models:

- marginal return of a campaign = elasticity_prior x (conversion value / spend) over the
  lookback window (one shared elasticity, because two weeks of data cannot pin down each
  campaign's curve),
- pacing = spend / budget,
- creative fatigue = recent CTR / best 7-day CTR since that creative launched,
- a landing-page test is worth `landing_prior_lift` of site signups until proven otherwise.

These models are often wrong in a consistent way. The bandit learns by how much, per arm.
Only warehouse data, live ad settings and the journal are used here. Never the simulator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from ..config import AgentConfig
from ..journal import JournalState
from ..warehouse import EntityWindow, Warehouse


@dataclass(frozen=True)
class Candidate:
    """One concrete change. `estimate` is rounded to 6 places so the journal replays exactly."""

    arm: str
    target: str  # human label
    treated: tuple[str, ...]  # entity keys that will change (empty for the landing page)
    estimate: float  # $/day of profit, the hypothesis' own estimate
    hypothesis: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "estimate", round(float(self.estimate), 6))

    def to_dict(self) -> dict[str, Any]:
        return {"arm": self.arm, "target": self.target, "treated": list(self.treated),
                "estimate": round(self.estimate, 6), "hypothesis": self.hypothesis, "params": self.params}


@dataclass(frozen=True)
class EntityView:
    key: str
    name: str
    account_id: str
    budget: float  # live
    bid: float | None  # live
    spend_day: float
    value_day: float
    marginal: float  # estimated marginal return per extra dollar
    utilization: float
    writable: bool


@dataclass(frozen=True)
class Snapshot:
    day: date
    entities: dict[str, EntityView]
    site_sessions_day: float
    site_signups_day: float

    def summary(self) -> dict[str, Any]:
        sp = sum(e.spend_day for e in self.entities.values())
        v = sum(e.value_day for e in self.entities.values())
        return {"ad_spend_day": round(sp, 2), "ad_value_day": round(v, 2), "ad_profit_day": round(v - sp, 2),
                "site_sessions_day": round(self.site_sessions_day, 1),
                "site_signups_day": round(self.site_signups_day, 2),
                "entities": {k: {"m": round(e.marginal, 3), "util": round(e.utilization, 3)}
                             for k, e in self.entities.items()}}


def observe(wh: Warehouse, day: date, live: dict[str, tuple[float, float | None]], writable: set[str],
            cfg: AgentConfig) -> Snapshot:
    end = day - timedelta(days=1)
    start = day - timedelta(days=cfg.lookback_days)
    win: dict[str, EntityWindow] = wh.entity_windows(start, end)
    ents = {}
    for k, w in win.items():
        if w.days == 0 or w.spend <= 0:
            continue
        budget, bid = live.get(k, (w.avg_budget, w.avg_bid))
        ents[k] = EntityView(
            key=k, name=w.name, account_id=w.account_id, budget=budget, bid=bid, spend_day=w.spend / w.days,
            value_day=w.value / w.days, marginal=cfg.elasticity_prior * w.value / w.spend,
            utilization=w.spend / max(1e-9, w.avg_budget * w.days), writable=k in writable)
    sessions, signups = wh.site_daily(day - timedelta(days=7), end)
    return Snapshot(day, ents, sessions, signups)


def _ctr_ratio(wh: Warehouse, key: str, day: date) -> tuple[float, float, float] | None:
    """(recent 7-day CTR, best 7-day CTR for the current creative, ratio) or None if too new."""
    rows = wh.ctr_series(key, day - timedelta(days=56), day - timedelta(days=1))
    if len(rows) < 14:
        return None
    current = rows[-1][3]
    rows = [r for r in rows if r[3] == current]
    if len(rows) < 14:
        return None

    def ctr(chunk: list) -> float:
        imp = sum(r[1] for r in chunk)
        return sum(r[2] for r in chunk) / imp if imp else 0.0

    weekly = [ctr(rows[i:i + 7]) for i in range(0, len(rows) - 6)]
    recent, best = weekly[-1], max(weekly)
    if best <= 0:
        return None
    return recent, best, recent / best


def hypothesize(snap: Snapshot, wh: Warehouse, js: JournalState, cfg: AgentConfig) -> list[Candidate]:
    out: list[Candidate] = []
    locked = js.locked_entities()
    open_refresh = {k for d in js.decisions.values() if d.arm == "creative_refresh" and not d.measured
                    for k in d.treated}
    elig = {k: e for k, e in snap.entities.items() if e.writable and k not in locked}
    room = len(js.open_entity_decisions()) < cfg.max_open_entity_measurements

    if room:
        # budget shift: lowest to highest estimated marginal return, inside one account
        best: Candidate | None = None
        for acct in sorted({e.account_id for e in elig.values()}):
            pool = [e for e in elig.values() if e.account_id == acct]
            srcs = [e for e in pool if e.budget >= 60]
            dsts = [e for e in pool if e.utilization >= 0.90]
            if not srcs or not dsts:
                continue
            src = min(srcs, key=lambda e: e.marginal)
            dst = max(dsts, key=lambda e: e.marginal)
            gap = dst.marginal - src.marginal
            if src.key == dst.key or gap < cfg.shift_min_gap:
                continue
            est = cfg.shift_amount * gap
            c = Candidate("budget_shift", f"{src.name} -> {dst.name}", (src.key, dst.key), est,
                          f"Move ${cfg.shift_amount:.0f}/day from {src.name} (marginal return {src.marginal:.2f}) "
                          f"to {dst.name} ({dst.marginal:.2f}). Expect about {est:+.0f} $/day.",
                          {"src": src.key, "dst": dst.key, "amount": cfg.shift_amount})
            if best is None or c.estimate > best.estimate:
                best = c
        if best:
            out.append(best)

        # bid up: under-pacing campaigns that return more than they cost at the margin
        ups = [e for e in elig.values() if e.bid and e.utilization < cfg.underpacing_utilization and e.marginal > 1.05]
        if ups:
            e = max(ups, key=lambda e: e.budget * min(1 - e.utilization, 0.25) * (e.marginal - 1))
            est = e.budget * min(1 - e.utilization, 0.25) * (e.marginal - 1)
            out.append(Candidate("bid_up", e.name, (e.key,), est,
                                 f"Raise the bid on {e.name} {cfg.bid_up_pct:.0%}. It spends {e.utilization:.0%} of "
                                 f"budget and returns {e.marginal:.2f} per marginal dollar. Expect about {est:+.0f} $/day.",
                                 {"key": e.key, "pct": cfg.bid_up_pct}))

        # bid down: fully-paced campaigns whose marginal dollar loses money
        downs = [e for e in elig.values() if e.bid and e.utilization >= cfg.saturated_utilization and e.marginal < 0.9]
        if downs:
            e = min(downs, key=lambda e: e.marginal)
            est = e.spend_day * cfg.bid_down_pct * (1 - e.marginal)
            out.append(Candidate("bid_down", e.name, (e.key,), est,
                                 f"Cut the bid on {e.name} {cfg.bid_down_pct:.0%}. It spends its whole budget and the "
                                 f"marginal dollar returns {e.marginal:.2f}. Expect about {est:+.0f} $/day.",
                                 {"key": e.key, "pct": -cfg.bid_down_pct}))

        # creative refresh: CTR has decayed against the creative's own best week
        best_r: Candidate | None = None
        for k, e in sorted(elig.items()):
            last = js.refreshes.get(k)
            if k in open_refresh or (last and (snap.day - last).days < cfg.refresh_cooldown_days):
                continue
            r = _ctr_ratio(wh, k, snap.day)
            if r is None or r[2] >= cfg.refresh_ctr_ratio:
                continue
            est = e.value_day * (1 / r[2] - 1) * 0.5
            c = Candidate("creative_refresh", e.name, (k,), est,
                          f"Refresh creative on {e.name}. CTR is {r[0]:.2%}, {r[2]:.0%} of its best week "
                          f"({r[1]:.2%}). If half the drop comes back, about {est:+.0f} $/day.",
                          {"key": k, "ctr_recent": r[0], "ctr_best": r[1]})
            if best_r is None or c.estimate > best_r.estimate:
                best_r = c
        if best_r:
            out.append(best_r)

    # landing-page test: one at a time, with a short rest between tests
    rested = js.last_experiment_end is None or (snap.day - js.last_experiment_end).days >= cfg.landing_cooldown_days
    if js.open_experiment is None and rested and snap.site_signups_day > 0:
        est = cfg.landing_prior_lift * snap.site_signups_day * cfg.value_per_conversion
        out.append(Candidate("landing_test", "/trial hero test", (), est,
                             f"Test 3 landing-page angles on /trial ({snap.site_sessions_day:,.0f} sessions/day). "
                             f"Prior worth {cfg.landing_prior_lift:.0%} of signups, about {est:+.0f} $/day.",
                             {"daily_visitors": int(snap.site_sessions_day)}))
    return out
