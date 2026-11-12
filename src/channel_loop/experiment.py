"""SIMULATED multi-seed comparison: the loop vs a static baseline, a random policy and a
no-learning greedy policy, each run through the exact same ticks, guardrails and market.

For a given seed every policy faces the same hidden market and the same daily noise, so
the seed-level difference in profit is a paired comparison. Profit here is the realized
simulated profit: site signups x value per signup - ad spend, summed over the 12 weeks.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from .agent.bandit import posteriors
from .config import ARMS, LoopConfig
from .journal import Journal
from .loop import Loop, load_guard_config
from .sim.market import Market
from .sim.source import SimSource
from .sim.truth import true_effect
from .stats import compare, mean_band, nan_to_none, pearson
from .warehouse import Warehouse

POLICIES = ("loop", "loop_ts", "greedy", "random", "static")
START = date(2026, 9, 7)  # a Monday; 12 weeks ends Sunday 2026-11-29 (Thanksgiving week inside)


@dataclass(frozen=True)
class RunSpec:
    seed: int
    policy: str
    root: str
    guard_path: str
    brief_path: str
    cfg: LoopConfig
    weeks: int = 12
    build_sites: bool = False


def run_one(spec: RunSpec) -> dict[str, Any]:
    root = Path(spec.root)
    if root.exists():
        shutil.rmtree(root)
    cfg = spec.cfg.with_policy(spec.policy)
    guard = load_guard_config(spec.guard_path)
    horizon = 7 * spec.weeks
    src = SimSource(root, spec.seed, START, horizon, cfg.agent.value_per_conversion)
    loop = Loop(root, cfg, guard, spec.brief_path, seed=spec.seed, source=src, build_sites=spec.build_sites)
    for i in range(horizon):
        loop.tick(START + timedelta(days=i))
    # the last tick acts on day 84; ingest that day too so every day of the horizon is counted
    with Warehouse(loop.paths.warehouse) as wh:
        src.ingest_through(START + timedelta(days=horizon - 1), wh, loop.account())
        rows = [r for r in wh.totals_by_day() if START <= r[0] < START + timedelta(days=horizon)]
    value = cfg.agent.value_per_conversion
    daily = [float(sig * value - sp) for _, sp, _, sig in rows]
    js = loop.journal.state()
    market = Market.load(src.path)
    decisions = []
    for d in sorted(js.decisions.values(), key=lambda d: d.decision_id):
        t = true_effect(market, d, cfg.measure) if d.measured else None
        decisions.append(dict(id=d.decision_id, day=d.day.isoformat(), arm=d.arm, target=d.target, x=round(d.x, 2),
                              outcome=d.outcome, y=d.y, se=d.se, method=d.method, truth=t))
    audit_writes = sum(1 for r in loop.audit.records() if r["event"] == "applied")
    audit_blocked = sum(1 for r in loop.audit.records() if r["event"] in ("blocked", "blocked_at_write"))
    ok, msg = loop.journal.verify()
    beliefs = {a: p.to_dict() for a, p in posteriors(js.observations, cfg.bandit).items()}
    return dict(seed=spec.seed, policy=spec.policy, daily_profit=daily, total_profit=float(sum(daily)),
                decisions=decisions, ads_writes=audit_writes, ads_blocked=audit_blocked, journal_ok=ok,
                journal=msg, ships=market.applied_ships, beliefs=beliefs, root=str(root))


def run_matrix(seeds: list[int], policies: tuple[str, ...], cfg: LoopConfig, guard_path: str, brief_path: str,
               workdir: Path | None = None, workers: int = 4, weeks: int = 12,
               keep: tuple[int, str] | None = None) -> list[dict[str, Any]]:
    base = Path(workdir or tempfile.mkdtemp(prefix="channel-loop-"))
    specs = [RunSpec(s, p, str(base / f"{p}-seed{s}"), guard_path, brief_path, cfg, weeks,
                     build_sites=bool(keep and keep == (s, p))) for s in seeds for p in policies]
    if workers <= 1:
        results = [run_one(s) for s in specs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(run_one, specs))
    if workdir is None:
        shutil.rmtree(base, ignore_errors=True)
    return results


def summarize(results: list[dict[str, Any]], weeks: int = 12) -> dict[str, Any]:
    by = {(r["policy"], r["seed"]): r for r in results}
    seeds = sorted({r["seed"] for r in results})
    policies = [p for p in POLICIES if any(r["policy"] == p for r in results)]
    out: dict[str, Any] = {"label": "SIMULATED", "seeds": seeds, "weeks": weeks, "policies": {}, "comparisons": {}}

    def weekly(r: dict) -> np.ndarray:
        return np.array(r["daily_profit"]).reshape(weeks, 7).sum(axis=1)

    for p in policies:
        tot = np.array([by[(p, s)]["total_profit"] for s in seeds])
        out["policies"][p] = {"mean_total_profit": round(float(tot.mean()), 2),
                              "actions": float(np.mean([sum(1 for d in by[(p, s)]["decisions"]
                                                            if d["outcome"] not in ("blocked", "dry_run", "halted"))
                                                        for s in seeds])),
                              "ads_writes": float(np.mean([by[(p, s)]["ads_writes"] for s in seeds])),
                              "journals_intact": all(by[(p, s)]["journal_ok"] for s in seeds)}
    if "static" in policies:
        for p in policies:
            if p == "static":
                continue
            diff = np.array([by[(p, s)]["total_profit"] - by[("static", s)]["total_profit"] for s in seeds])
            wk = np.array([weekly(by[(p, s)]) - weekly(by[("static", s)]) for s in seeds])
            m, lo, hi = mean_band(np.cumsum(wk, axis=1))
            wm, wlo, whi = mean_band(wk)
            out["comparisons"][f"{p}_vs_static"] = {
                **compare(diff).to_dict(),
                "cumulative_weekly": {"mean": m.round(2).tolist(), "lo": lo.round(2).tolist(), "hi": hi.round(2).tolist()},
                "weekly": {"mean": wm.round(2).tolist(), "lo": wlo.round(2).tolist(), "hi": whi.round(2).tolist()},
                "first4_vs_last4_per_week": [round(float(wk[:, :4].mean()), 2), round(float(wk[:, -4:].mean()), 2)],
            }
    for a, b in (("loop", "random"), ("loop", "greedy"), ("loop", "loop_ts")):
        if a in policies and b in policies:
            diff = np.array([by[(a, s)]["total_profit"] - by[(b, s)]["total_profit"] for s in seeds])
            out["comparisons"][f"{a}_vs_{b}"] = compare(diff).to_dict()
    # attribution quality and what the loop learned to do
    if "loop" in policies:
        meas = [d for s in seeds for d in by[("loop", s)]["decisions"] if d["y"] is not None and d["truth"] is not None]
        ys, ts = [d["y"] for d in meas], [d["truth"] for d in meas]
        cover = [abs(d["y"] - d["truth"]) <= 1.96 * d["se"] for d in meas]
        out["attribution"] = {
            "n": len(meas), "pearson_r": nan_to_none(round(pearson(ys, ts), 3)),
            "mean_abs_error": round(float(np.mean(np.abs(np.array(ys) - np.array(ts)))), 2) if meas else None,
            "sign_agreement": round(float(np.mean([(y > 0) == (t > 0) for y, t in zip(ys, ts)])), 3) if meas else None,
            "ci95_coverage": round(float(np.mean(cover)), 3) if meas else None,
            "by_arm": {a: {"n": sum(1 for d in meas if d["arm"] == a),
                           "mean_truth": nan_to_none(round(float(np.mean([d["truth"] for d in meas if d["arm"] == a])), 2))
                           if any(d["arm"] == a for d in meas) else None,
                           "mean_estimate": nan_to_none(round(float(np.mean([d["x"] for d in meas if d["arm"] == a])), 2))
                           if any(d["arm"] == a for d in meas) else None}
                       for a in ARMS},
        }
        mix = {}
        for p in policies:
            half = weeks * 7 // 2
            early = [d["arm"] for s in seeds for d in by[(p, s)]["decisions"] if d["outcome"] not in ("blocked",)
                     and (date.fromisoformat(d["day"]) - START).days < half]
            late = [d["arm"] for s in seeds for d in by[(p, s)]["decisions"] if d["outcome"] not in ("blocked",)
                    and (date.fromisoformat(d["day"]) - START).days >= half]
            mix[p] = {"first_6_weeks": {a: early.count(a) for a in ARMS}, "last_6_weeks": {a: late.count(a) for a in ARMS}}
        out["action_mix"] = mix
        out["learned_beliefs_mean"] = {a: {k: round(float(np.mean([by[("loop", s)]["beliefs"][a][k] for s in seeds])), 3)
                                           for k in ("n", "w0", "w1")} for a in ARMS}
        truth_by_policy = {}
        for p in policies:
            tr = [d["truth"] for s in seeds for d in by[(p, s)]["decisions"] if d["truth"] is not None]
            truth_by_policy[p] = {"n": len(tr), "mean_true_effect": round(float(np.mean(tr)), 2) if tr else None,
                                  "share_harmful": round(float(np.mean([t < 0 for t in tr])), 3) if tr else None}
        out["true_effect_of_actions"] = truth_by_policy
    out["ad_change_quality"] = ad_change_quality(by, seeds, policies, weeks)
    return out


AD_ARMS = ("budget_shift", "bid_up", "bid_down", "creative_refresh")


def ad_change_quality(by: dict, seeds: list[int], policies: list[str], weeks: int) -> dict[str, Any]:
    """Decision quality, graded by the hidden truth: what each ad change was really worth.

    Landing tests are left out on purpose. Every policy launches them the same way, and
    their outcome (did a winner ship, how big) is mostly traffic luck. Ad changes are where
    the policies actually choose differently.
    """
    half = START + timedelta(days=weeks * 7 // 2)

    def changes(p: str, s: int) -> list[dict]:
        return [d for d in by[(p, s)]["decisions"] if d["arm"] in AD_ARMS and d["truth"] is not None]

    q: dict[str, Any] = {"policies": {}, "comparisons": {}}
    per_change = {}
    for p in policies:
        if p == "static":
            continue
        allc = [d for s in seeds for d in changes(p, s)]
        early = [d["truth"] for d in allc if date.fromisoformat(d["day"]) < half]
        late = [d["truth"] for d in allc if date.fromisoformat(d["day"]) >= half]
        per_change[p] = np.array([np.mean([d["truth"] for d in changes(p, s)]) if changes(p, s) else 0.0
                                  for s in seeds])
        q["policies"][p] = {
            "changes_per_run": round(len(allc) / len(seeds), 2),
            "true_value_per_change": round(float(np.mean([d["truth"] for d in allc])), 2) if allc else None,
            "share_harmful": round(float(np.mean([d["truth"] < 0 for d in allc])), 3) if allc else None,
            "true_value_sum_per_run": round(float(sum(d["truth"] for d in allc)) / len(seeds), 2),
            "weeks_1_6_value_per_change": round(float(np.mean(early)), 2) if early else None,
            "weeks_7_12_value_per_change": round(float(np.mean(late)), 2) if late else None,
            "weeks_1_6_share_harmful": round(float(np.mean([t < 0 for t in early])), 3) if early else None,
            "weeks_7_12_share_harmful": round(float(np.mean([t < 0 for t in late])), 3) if late else None,
        }
    for b in ("random", "greedy", "loop_ts"):
        if "loop" in per_change and b in per_change:
            q["comparisons"][f"loop_vs_{b}_value_per_change"] = compare(per_change["loop"] - per_change[b]).to_dict()
    return q


def summarize_file(runs_path: Path, weeks: int = 12) -> dict[str, Any]:
    return summarize(json.loads(Path(runs_path).read_text()), weeks)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def journal_of(root: str | Path) -> Journal:
    return Journal(Path(root) / "journal.jsonl")
