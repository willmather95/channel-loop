"""HANDS for landing-page tests, through landing-page-lab's public API.

- `launch` calls `create_experiment()`: guarded variants from the brief, a traffic plan,
  optionally a static build, and the PostHog experiment requests (dry run, never sent).
  It then turns on the split on the flag service (here: `platform/site.json`).
- `read` calls `readout()` on the planned look days only, with the plan's split and
  control, so the loop cannot peek its way into a false winner.
- `conclude` ships a winner or keeps control. Shipping to production is a human approval
  in live use. In the simulator the approval is simulated and logged as such.
"""
from __future__ import annotations

import dataclasses
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from landing_page_lab import Policy, Status, create_experiment, readout
from landing_page_lab.brief import Brief, load_brief

from ..platform import LiveExperiment, SiteState


class LandingHands:
    def __init__(self, brief_path: str | Path, out_dir: Path, site_path: Path, build: bool = False) -> None:
        self.brief: Brief = load_brief(brief_path)
        self.out_dir = out_dir
        self.site_path = site_path
        self.build = build

    def launch(self, decision_id: str, day: date, daily_visitors: int, apply_writes: bool) -> dict[str, Any]:
        key = f"{self.brief.experiment.key}-{day:%Y%m%d}"
        exp = dataclasses.replace(self.brief.experiment, key=key, daily_visitors=max(100, int(daily_visitors)))
        brief = dataclasses.replace(self.brief, experiment=exp)
        out = self.out_dir / key
        handle = create_experiment(brief, out_dir=out if self.build else None, build=self.build)
        arms = [v.key for v in handle.variants]
        info = dict(key=key, arms=arms, angles={v.key: v.angle for v in handle.variants}, control=handle.control,
                    weights=list(handle.weights), look_days=list(handle.plan.look_days), started=day.isoformat(),
                    plan_note=handle.plan.note, posthog_requests=len(handle.posthog.requests),
                    posthog_dry_run=handle.dry_run, built=str(out / "dist") if self.build else None)
        if apply_writes:
            site = SiteState.load(self.site_path)
            if site.experiment is not None:
                raise RuntimeError(f"a test is already live: {site.experiment.key}")
            site.experiment = LiveExperiment(key=key, arms=arms, weights=list(handle.weights), control=handle.control,
                                             started=day.isoformat())
            site.save(self.site_path)
        return info

    @staticmethod
    def next_look(exp: dict[str, Any], looks_done: set[int], day: date) -> int | None:
        """The planned look day that is due (data through yesterday covers it), if any."""
        elapsed = (day - date.fromisoformat(exp["started"])).days
        for ld in exp["look_days"]:
            if ld not in looks_done and elapsed >= ld:
                return ld
        return None

    @staticmethod
    def read(exp: dict[str, Any], counts: dict[str, tuple[int, int]], look_day: int):
        policy = Policy(look_days=tuple(exp["look_days"]))
        arms = {a: counts.get(a, (0, 0)) for a in exp["arms"]}
        return readout(arms, day=look_day, policy=policy, control=exp["control"], weights=exp["weights"])

    @staticmethod
    def window(exp: dict[str, Any], look_day: int) -> tuple[date, date]:
        start = date.fromisoformat(exp["started"])
        return start, start + timedelta(days=look_day - 1)

    def conclude(self, exp: dict[str, Any], r, day: date, apply_writes: bool) -> dict[str, Any]:
        shipped = r.winner if (r.status is Status.WINNER and r.winner and r.winner != exp["control"]) else None
        row = r.arm(shipped) if shipped else None
        lift = row.lift if row else None
        lift_se = ((row.lift_ci[1] - row.lift_ci[0]) / 3.92) if (row and row.lift_ci) else None
        if apply_writes:
            site = SiteState.load(self.site_path)
            site.experiment = None
            if shipped:
                site.production = f"{exp['key']}:{shipped}"
                site.shipped.append({"key": exp["key"], "variant": shipped, "on": day.isoformat()})
            site.save(self.site_path)
        return dict(key=exp["key"], status=r.status.value, winner=r.winner, shipped=shipped, lift=lift,
                    lift_se=lift_se, headline=r.headline(),
                    approval="simulated (production ships need a person in live use)" if shipped else None)
