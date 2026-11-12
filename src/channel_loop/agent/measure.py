"""Attribution: how much did a change really move profit per day?

Ad changes (budget shift, bid, creative) use a ratio difference-in-differences:

    counterfactual_post = treated_pre x (controls_post / controls_pre)      (value and spend apart)
    effect = treated_post_profit - (counterfactual_value - counterfactual_spend)

Controls are campaigns that did not change anywhere in the pre or post window, so the
weekday mix, the Thanksgiving dip, landing-page wins and shared demand shocks cancel out.
Days right after the change are skipped (conversion lag). The standard error comes from a
day-level bootstrap, resampling the same days for treated and controls. With no clean
control left, it falls back to pre/post, says so, and doubles the error.

Landing-page tests are measured by their own randomized holdout: the A/B readout's lift
of the shipped arm vs control, times site signups per day, times the value of a signup.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from ..config import AgentConfig, MeasureConfig
from ..journal import DecisionRecord, JournalState
from ..warehouse import Warehouse


@dataclass(frozen=True)
class Measurement:
    y: float  # $/day
    se: float
    method: str
    controls: tuple[str, ...]
    windows: dict[str, str]
    detail: dict


def _clean_controls(all_keys: list[str], treated: list[str], js: JournalState, lo: date, hi: date) -> list[str]:
    out = []
    for k in all_keys:
        if k in treated:
            continue
        if any(lo <= d <= hi for d in js.changes.get(k, [])):
            continue
        out.append(k)
    return out


def _did(vt: np.ndarray, st: np.ndarray, vc: np.ndarray, sc: np.ndarray, pre: np.ndarray, post: np.ndarray,
         controls: bool) -> float:
    if controls and vc[pre].sum() > 0 and sc[pre].sum() > 0:
        rv = vc[post].mean() / vc[pre].mean()
        rs = sc[post].mean() / sc[pre].mean()
    else:
        rv = rs = 1.0
    cf = vt[pre].mean() * rv - st[pre].mean() * rs
    return float((vt[post] - st[post]).mean() - cf)


def measure_entity(wh: Warehouse, d: DecisionRecord, js: JournalState, all_keys: list[str], cfg: MeasureConfig) -> Measurement:
    eff = date.fromisoformat(d.details.get("effective", d.day.isoformat()))
    pre_lo, pre_hi = eff - timedelta(days=cfg.pre_days), eff - timedelta(days=1)
    post_lo, post_hi = eff + timedelta(days=cfg.lag_days), eff + timedelta(days=cfg.lag_days + cfg.post_days - 1)
    controls = _clean_controls(all_keys, d.treated, js, pre_lo, post_hi)
    data = wh.daily_entity(pre_lo, post_hi)
    days = [pre_lo + timedelta(days=i) for i in range((post_hi - pre_lo).days + 1)]
    idx_pre = np.array([i for i, x in enumerate(days) if x <= pre_hi])
    idx_post = np.array([i for i, x in enumerate(days) if x >= post_lo])

    def series(keys: list[str], j: int) -> np.ndarray:
        return np.array([sum(data.get(k, {}).get(x, (0.0, 0.0))[j] for k in keys) for x in days])

    vt, st = series(d.treated, 1), series(d.treated, 0)
    vc, sc = series(controls, 1), series(controls, 0)
    use_controls = len(controls) >= cfg.min_controls
    y = _did(vt, st, vc, sc, idx_pre, idx_post, use_controls)
    rng = np.random.default_rng(zlib.crc32(d.decision_id.encode()))
    boots = np.empty(cfg.bootstrap_reps)
    for b in range(cfg.bootstrap_reps):
        bp = rng.choice(idx_pre, size=len(idx_pre))
        bq = rng.choice(idx_post, size=len(idx_post))
        boots[b] = _did(vt, st, vc, sc, bp, bq, use_controls)
    se = float(boots.std(ddof=1))
    method = "diff-in-diff vs untouched campaigns" if use_controls else "pre/post only (no clean control), error doubled"
    if not use_controls:
        se *= 2
    return Measurement(y, max(se, 1.0), method, tuple(controls),
                       {"pre": f"{pre_lo}..{pre_hi}", "post": f"{post_lo}..{post_hi}"},
                       {"treated_profit_pre": round(float((vt[idx_pre] - st[idx_pre]).mean()), 2),
                        "treated_profit_post": round(float((vt[idx_post] - st[idx_post]).mean()), 2)})


def measure_landing(conclusion: dict, signups_day: float, cfg: AgentConfig) -> Measurement:
    """Effect of a finished landing-page test, from its own randomized control arm."""
    scale = signups_day * cfg.value_per_conversion
    shipped = conclusion.get("shipped")
    if shipped and conclusion.get("lift") is not None:
        y = float(conclusion["lift"]) * scale
        se = max(float(conclusion.get("lift_se") or 0.0) * scale, 1.0)
        method = f"A/B readout: shipped {shipped}, randomized control arm"
    else:
        y, se = 0.0, 5.0
        method = f"A/B readout: {conclusion.get('status')}, kept control (no change shipped)"
    return Measurement(y, se, method, (), {}, {"signups_day": round(signups_day, 2)})
