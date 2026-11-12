"""HOME: ticks are idempotent, resumable after a crash, ordered, and single-runner."""
from __future__ import annotations

import json
from datetime import timedelta

import pytest

from channel_loop.experiment import START
from channel_loop.home.runner import Meta, RunnerBusy, catch_up, runner_lock
from channel_loop.loop import Loop

from conftest import BRIEF, GUARD, make_loop, run_days


def _events(loop):
    return [r["event"] for r in loop.journal.records()]


def test_finished_tick_is_a_noop(tmp_path):
    lp = make_loop(tmp_path)
    run_days(lp, 3)
    n_journal = len(lp.journal.records())
    n_audit = len(list(lp.audit.records()))
    wh_rows = _count_rows(lp)
    for i in range(3):
        assert lp.tick(START + timedelta(days=i)).status == "noop"
    assert len(lp.journal.records()) == n_journal
    assert len(list(lp.audit.records())) == n_audit
    assert _count_rows(lp) == wh_rows


def _count_rows(lp):
    from channel_loop.warehouse import Warehouse
    with Warehouse(lp.paths.warehouse) as wh:
        return wh.con.execute("SELECT count(*) FROM ads_daily").fetchone()[0]


def test_a_fresh_process_sees_the_same_state(tmp_path):
    lp = make_loop(tmp_path)
    run_days(lp, 4)
    again = make_loop(tmp_path)  # new objects, same state dir: everything comes from the journal
    assert again.tick(START + timedelta(days=3)).status == "noop"
    assert again.tick(START + timedelta(days=4)).status == "ran"


def test_ticks_cannot_go_backwards(tmp_path):
    lp = make_loop(tmp_path)
    run_days(lp, 3)
    with pytest.raises(ValueError):
        lp.tick(START + timedelta(days=1) - timedelta(days=2))


@pytest.mark.parametrize("crash_in", ["readout", "measure", "_act"])
def test_crash_mid_tick_resumes_without_doing_anything_twice(tmp_path, monkeypatch, crash_in):
    ref = make_loop(tmp_path / "ref")
    run_days(ref, 12)

    lp = make_loop(tmp_path / "crash")
    run_days(lp, 11)
    target = {"readout": "_readout", "measure": "_measure", "_act": "_act"}[crash_in]
    original = getattr(Loop, target)
    calls = {"n": 0}

    def boom(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            if target == "_act":
                original(self, *a, **k)  # the write happens, then the process dies
            raise RuntimeError("simulated crash")
        return original(self, *a, **k)

    monkeypatch.setattr(Loop, target, boom)
    day = START + timedelta(days=11)
    with pytest.raises(RuntimeError):
        lp.tick(day)
    monkeypatch.setattr(Loop, target, original)
    resumed = make_loop(tmp_path / "crash")
    assert resumed.tick(day).status == "ran"

    def acts(loop):
        return [(r["decision_id"], r["outcome"]) for r in loop.journal.records() if r["event"] == "acted"]

    def decisions(loop):
        return [(r["decision_id"], r.get("arm")) for r in loop.journal.records() if r["event"] == "decision"]

    def writes(loop):
        return [(r["entity_key"], r["after_micros"]) for r in loop.audit.records() if r["event"] == "applied"]

    assert decisions(resumed) == decisions(ref)
    assert acts(resumed) == acts(ref)
    assert writes(resumed) == writes(ref)
    assert resumed.journal.verify()[0]


def test_catch_up_runs_missing_days_in_order_and_lock_blocks_a_second_runner(tmp_path, cfg):
    Meta(5, START.isoformat(), 14, "loop", GUARD, BRIEF).save(tmp_path)
    done = catch_up(tmp_path, cfg, START + timedelta(days=4))
    assert [r.day for r in done] == [START + timedelta(days=i) for i in range(5)]
    assert catch_up(tmp_path, cfg, START + timedelta(days=4)) == []  # nothing missing
    with runner_lock(tmp_path):
        with pytest.raises(RunnerBusy):
            catch_up(tmp_path, cfg, START + timedelta(days=6))
    more = catch_up(tmp_path, cfg, START + timedelta(days=40))  # capped at the horizon
    assert more[-1].day == START + timedelta(days=13)
    meta = json.loads((tmp_path / "loop-meta.json").read_text())
    assert meta["horizon_days"] == 14
