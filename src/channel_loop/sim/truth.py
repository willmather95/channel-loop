"""SIMULATED ground truth: what each decision really did, from the market's hidden parameters.

Only the evaluation harness calls this, after a run, to grade the loop's own measurements.
The loop never sees it. For an ad change, truth is the expected profit per day over the
same post window, with the change minus without it, holding everything else as it was
(the realized demand shocks included). For a landing test, truth is the hidden lift of the
shipped arm times the expected signups per day in the week after shipping.
"""
from __future__ import annotations

from datetime import date, timedelta

from ..config import MeasureConfig
from ..journal import DecisionRecord
from ..world import CAMPAIGNS, key_of
from .market import Market


def _age(day: date, launched_on: str) -> int:
    return (day - date.fromisoformat(launched_on)).days


def _profit(m: Market, key: str, day: date, budget: float, bid: float, creative_id: str, launched_on: str,
            eff_mult: float, learning: float) -> float:
    spend, _, signups = m.expected(key, day, budget, bid, _age(day, launched_on), m.creative_quality(creative_id),
                                   eff_mult, learning)
    return m.value * signups - spend


def true_effect(m: Market, d: DecisionRecord, cfg: MeasureConfig) -> float | None:
    if not d.acted:
        return None
    if d.arm == "landing_test":
        return _true_landing(m, d)
    eff = date.fromisoformat(d.details.get("effective", d.day.isoformat()))
    days = [eff + timedelta(days=cfg.lag_days + i) for i in range(cfg.post_days)]
    if any(x.isoformat() not in m.history for x in days):
        return None
    before = {w["entity_key"]: w["before"] for w in d.details.get("writes", [])}
    prev_creative = m.history[(eff - timedelta(days=1)).isoformat()]["entities"]
    total = 0.0
    for x in days:
        h = m.history[x.isoformat()]
        for key in d.treated:
            e = h["entities"][key]
            with_ = _profit(m, key, x, e["budget"], e["bid"], e["creative_id"], e["launched_on"], h["eff_mult"],
                            e["learning"])
            budget, bid, cid, launched = e["budget"], e["bid"], e["creative_id"], e["launched_on"]
            if d.arm == "budget_shift":
                budget = before.get(key, budget)
            elif d.arm in ("bid_up", "bid_down"):
                bid = before.get(key, bid)
            elif d.arm == "creative_refresh":
                cid, launched = prev_creative[key]["creative_id"], prev_creative[key]["launched_on"]
            without = _profit(m, key, x, budget, bid, cid, launched, h["eff_mult"], 1.0)
            total += with_ - without
    return total / len(days)


def _true_landing(m: Market, d: DecisionRecord) -> float | None:
    key = d.details.get("key")
    tag = next((t for t in m.applied_ships if t.startswith(f"{key}:")), None)
    if tag is None:
        return 0.0  # nothing shipped: the page did not change
    mult = m.exp_mults[key][tag.split(":", 1)[1]]
    ship_day = date.fromisoformat(m.ship_days[tag])
    days = [ship_day + timedelta(days=i) for i in range(7) if (ship_day + timedelta(days=i)).isoformat() in m.history]
    if not days:
        return None
    total = 0.0
    for x in days:
        h = m.history[x.isoformat()]
        sig = m.expected_organic(x, h["eff_mult"])
        for c in CAMPAIGNS:
            e = h["entities"][key_of(c)]
            sig += m.expected(key_of(c), x, e["budget"], e["bid"], _age(x, e["launched_on"]),
                              m.creative_quality(e["creative_id"]), h["eff_mult"], e["learning"])[2]
        total += m.value * sig * (1 - 1 / mult)
    return total / len(days)
