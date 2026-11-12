"""Guardrails hold end to end: every write the loop made, re-checked from the audit log alone."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta

import pytest

from channel_loop.experiment import START
from channel_loop.platform import CreativeState, SiteState
from channel_loop.world import starting_entities

from conftest import make_loop, no_writes, run_days

BRAND = "google_ads:9990001111:31000000001"


def _applied(lp):
    return [r for r in lp.audit.records() if r["event"] == "applied"]


@pytest.mark.parametrize("key", [("loop", 3), ("loop", 11), ("random", 3), ("random", 11)])
def test_every_write_is_inside_every_cap(five_week_runs, key):
    lp = five_week_runs[key]
    g = lp.guard.guardrails
    writes = _applied(lp)
    assert writes, "the run should have made some changes"
    ok, msg = lp.audit.verify()
    assert ok, msg
    budgets = {e.key: e.daily_budget_micros for e in starting_entities()}
    acct_of = {e.key: e.account_id for e in starting_entities()}
    last_write: dict[str, datetime] = {}
    moved = defaultdict(int)
    for r in writes:
        k, before, after = r["entity_key"], r["before_micros"], r["after_micros"]
        caps = g.budget if r["change_type"] == "budget" else g.bid
        assert k != BRAND, "brand search is not on the allowlist"
        assert abs(after - before) <= min(before * caps.max_change_bp // 10_000, caps.max_change_abs_micros)
        assert after >= caps.floor_micros
        assert after % 10_000 == 0, "whole cents"
        at = datetime.fromisoformat(r["at"])
        if k in last_write:
            assert at - last_write[k] >= timedelta(hours=g.cooldown_hours)
        last_write[k] = at
        if r["change_type"] == "budget":
            moved[(r["account_id"], r["at"][:10])] += abs(after - before)
            budgets[k] = after
    for (acct, _), m in moved.items():
        assert m <= g.account_daily_change_cap_micros
    for acct in {a for a in acct_of.values()}:
        total = sum(v for k, v in budgets.items() if acct_of[k] == acct)
        assert total <= g.total_cap_for(acct)


def test_budget_shifts_are_symmetric(five_week_runs):
    """A shift moves the same dollars out of one campaign and into another, or nothing at all."""
    for lp in five_week_runs.values():
        for r in lp.journal.records():
            if r["event"] == "acted" and r["arm"] == "budget_shift" and r["outcome"] in ("applied", "clamped"):
                w = r["details"]["writes"]
                assert len(w) == 2
                assert abs((w[0]["after"] - w[0]["before"]) + (w[1]["after"] - w[1]["before"])) < 1e-9


def test_only_one_landing_test_at_a_time_and_only_planned_looks(five_week_runs):
    for lp in five_week_runs.values():
        open_key = None
        launched, looks = {}, {}
        for r in lp.journal.records():
            if r["event"] == "lp_launched":
                assert open_key is None
                open_key = r["key"]
                launched[r["key"]] = r
            elif r["event"] == "lp_concluded":
                open_key = None
            elif r["event"] == "lp_readout":
                assert r["look_day"] in launched[r["key"]]["look_days"]
                looks.setdefault(r["key"], []).append(r["look_day"])
        for k, ls in looks.items():
            assert ls == sorted(set(ls))


def test_kill_switch_stops_every_kind_of_write_and_release_resumes(tmp_path):
    lp = make_loop(tmp_path, seed=3)
    before_writes = 0
    site_before = SiteState.load(lp.landing.site_path)
    creatives_before = CreativeState.load(lp.creative.path)
    lp.kill_switch.engage("test")
    res = run_days(lp, 9)
    acted_days = [r for r in res if any(d["arm"] for d in r.decisions)]
    assert acted_days and all(r.halted for r in acted_days)
    assert len(_applied(lp)) == before_writes
    assert SiteState.load(lp.landing.site_path).experiment == site_before.experiment
    assert len(CreativeState.load(lp.creative.path).requests) == len(creatives_before.requests)
    halted = [r for r in lp.journal.records() if r["event"] == "halted"]
    assert halted and all("KILL_SWITCH" in h["reason"] for h in halted)
    lp.kill_switch.release()
    run_days(lp, 10, START + timedelta(days=9))
    assert len(_applied(lp)) > before_writes or any(
        r["event"] == "acted" and r["outcome"] in ("launched", "queued") for r in lp.journal.records()
        if r["date"] >= (START + timedelta(days=9)).isoformat())


def test_kill_switch_by_environment(tmp_path, monkeypatch):
    lp = make_loop(tmp_path, seed=3)
    monkeypatch.setenv("CHANNEL_LOOP_KILL", "1")
    res = run_days(lp, 3)
    assert any(r.halted for r in res)
    assert not _applied(lp)


def test_write_switch_off_means_dry_run(tmp_path, guard):
    lp = make_loop(tmp_path, seed=3, guard=no_writes(guard))
    run_days(lp, 10)
    assert not _applied(lp)
    outcomes = {r["outcome"] for r in lp.journal.records() if r["event"] == "acted"}
    assert outcomes <= {"dry_run", "blocked"}
    assert SiteState.load(lp.landing.site_path).experiment is None
    snap = lp.account().snapshot()
    assert snap == {e.key: (e.daily_budget_micros, e.bid_micros) for e in sorted(starting_entities(), key=lambda e: e.key)}


def test_locked_campaigns_are_never_changed_again_before_measurement(five_week_runs):
    for lp in five_week_runs.values():
        js = lp.journal.state()
        for d in js.decisions.values():
            if not d.acted or d.arm == "landing_test" or d.measure_due is None:
                continue
            for other in js.decisions.values():
                if other is d or not other.acted or other.arm == "landing_test":
                    continue
                if d.day < other.day < d.measure_due:
                    assert not set(other.treated) & set(d.treated), (d.decision_id, other.decision_id)
