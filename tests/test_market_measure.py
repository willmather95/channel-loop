"""The SIMULATED market behaves as documented, and the attribution recovers known effects."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest
from ads_guardrail import InMemoryAdAccount
from ads_guardrail.warehouse import WarehouseReader

from channel_loop.agent.measure import measure_entity, measure_landing
from channel_loop.config import AgentConfig, MeasureConfig
from channel_loop.experiment import START
from channel_loop.journal import DecisionRecord, JournalState
from channel_loop.platform import SiteState
from channel_loop.sim.market import Market, utilization
from channel_loop.warehouse import Warehouse
from channel_loop.world import starting_entities

KEY = "google_ads:9990001111:31000000002"


def _run_market(seed: int, days: int):
    m = Market(seed, START)
    acct = InMemoryAdAccount(starting_entities())
    cr = m.initial_creatives()
    out = []
    d = m.origin
    for _ in range(days):
        out.append(m.simulate_day(d, acct, SiteState(), cr))
        d += timedelta(days=1)
    return m, out


def test_same_seed_same_world_and_days_must_be_in_order():
    a, da = _run_market(4, 10)
    b, db = _run_market(4, 10)
    assert da == db
    c, dc = _run_market(5, 10)
    assert da != dc
    with pytest.raises(ValueError):
        a.simulate_day(a.last_day + timedelta(days=2), InMemoryAdAccount(starting_entities()), SiteState(),
                       a.initial_creatives())


def test_diminishing_returns_and_pacing():
    m = Market(1, START)
    day = START + timedelta(days=2)
    sig = [m.expected(KEY, day, b, 60.0, 30, 1.0, 1.0)[2] for b in (100, 200, 300, 400)]
    gains = np.diff(sig)
    assert all(g > 0 for g in gains) and gains[0] > gains[1] > gains[2]
    assert utilization(10, 20) < utilization(22, 20) < utilization(30, 20) <= 1.0


def test_fatigue_lowers_signups_and_a_fresh_creative_restores_them():
    m = Market(1, START)
    day = START + timedelta(days=2)
    meta = "meta:act_000000000000042:23800000002"
    old = m.expected(meta, day, 260, 24, 120, 1.0, 1.0)[2]
    new = m.expected(meta, day, 260, 24, 0, 1.0, 1.0)[2]
    assert new > old * 1.3


def test_thanksgiving_dip_is_in_the_data():
    m = Market(2, START)
    assert m.demand(date(2026, 11, 26)) < 0.7 * m.demand(date(2026, 11, 19))


def test_lag_conserves_conversions():
    m, days = _run_market(6, 40)
    reported = sum(r["conversions"] for d in days for r in d["ads"])
    true_paid = sum(w["signups"] for d in days for w in d["web"] if w["source"] == "paid")
    pending = sum(sum(v.values()) for v in m.pending.values())
    assert reported + pending == true_paid


def test_warehouse_load_is_idempotent_and_readable_by_ads_guardrail(tmp_path):
    m, days = _run_market(6, 5)
    with Warehouse(tmp_path / "w.duckdb") as wh:
        for d in days:
            wh.load_day(d["ads"][0]["date"], d["ads"], d["settings"], d["web"])
        wh.load_day(days[0]["ads"][0]["date"], days[0]["ads"], days[0]["settings"], days[0]["web"])
        assert wh.con.execute("SELECT count(*) FROM ads_daily").fetchone()[0] == 5 * 8
    reader = WarehouseReader(tmp_path / "w.duckdb")
    assert len(reader.fetch_metrics(m.origin, m.origin + timedelta(days=4))) == 40
    assert {e.key for e in reader.list_entities()} == {e.key for e in starting_entities()}


def _synthetic_warehouse(tmp_path, effect: float, shock_day: date, seed: int = 0):
    """Four campaigns, 21 days. A shared demand shock hits everyone; the treated one gains `effect`."""
    rng = np.random.default_rng(seed)
    wh = Warehouse(tmp_path / f"s{seed}.duckdb")
    keys = [f"google_ads:1:{i}" for i in range(4)]
    start = shock_day - timedelta(days=10)
    for t in range(21):
        d = start + timedelta(days=t)
        common = 0.7 if d >= shock_day + timedelta(days=3) else 1.0  # demand dip after the change
        ads = []
        for i, k in enumerate(keys):
            spend = 200.0
            value = 400.0 * common * (1 + 0.03 * rng.standard_normal())
            if i == 0 and d >= shock_day:
                value += effect * common
            ads.append(dict(platform="google_ads", account_id="1", entity_id=str(i), entity_name=k, date=d,
                            spend=spend, impressions=1000, clicks=100, conversions=value / 60,
                            conversion_value=value))
        wh.load_day(d, ads, [], [])
    return wh, keys


@pytest.mark.parametrize("effect", [80.0, -50.0, 0.0])
def test_diff_in_diff_recovers_a_known_effect_through_a_demand_shock(tmp_path, effect):
    day = date(2026, 10, 1)
    wh, keys = _synthetic_warehouse(tmp_path, effect, day)
    d = DecisionRecord("d-x", day, "bid_up", "x", "", 10.0, treated=[keys[0]], acted=True,
                       details={"effective": day.isoformat()})
    m = measure_entity(wh, d, JournalState(), keys, MeasureConfig())
    # the change is worth `effect` at normal demand and 0.7x during the dip; post window is mostly in the dip
    assert m.method.startswith("diff-in-diff")
    assert len(m.controls) == 3
    expected = effect * np.mean([1.0 if day + timedelta(days=1 + i) < day + timedelta(days=3) else 0.7 for i in range(7)])
    assert abs(m.y - expected) < 3 * m.se + 5
    # naive pre/post would blame the demand dip on the change
    naive = m.detail["treated_profit_post"] - m.detail["treated_profit_pre"]
    assert abs(naive - expected) > 50
    wh.close()


def test_no_clean_control_falls_back_and_says_so(tmp_path):
    day = date(2026, 10, 1)
    wh, keys = _synthetic_warehouse(tmp_path, 50.0, day, seed=1)
    js = JournalState(changes={k: [day + timedelta(days=2)] for k in keys[1:]})  # every control changed mid-window
    d = DecisionRecord("d-y", day, "bid_up", "x", "", 10.0, treated=[keys[0]], acted=True,
                       details={"effective": day.isoformat()})
    m = measure_entity(wh, d, js, keys, MeasureConfig())
    assert m.controls == ()
    assert "pre/post only" in m.method
    wh.close()


def test_landing_measurement_uses_the_ab_lift():
    cfg = AgentConfig()
    m = measure_landing({"shipped": "c", "lift": 0.10, "lift_se": 0.04, "status": "winner"}, 100.0, cfg)
    assert m.y == pytest.approx(0.10 * 100 * cfg.value_per_conversion)
    assert m.se == pytest.approx(0.04 * 100 * cfg.value_per_conversion)
    none = measure_landing({"shipped": None, "status": "no_winner_keep_control"}, 100.0, cfg)
    assert none.y == 0.0
