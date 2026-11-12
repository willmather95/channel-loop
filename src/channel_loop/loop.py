"""The loop: one idempotent tick per day.

    ingest -> readout (landing test, planned looks only) -> measure (due decisions)
           -> observe -> hypothesize -> choose -> act (through the guardrails) -> schedule
           -> mirror to the warehouse -> tick_finished

Every step writes to the journal. A tick whose `tick_finished` is in the journal is a no-op.
A tick that crashed part way resumes at the first step without a `step_done` line. Ad
writes are idempotent on their own (ads-guardrail's run ids and audit log), and landing
and creative requests check for an existing request before writing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from ads_guardrail import AuditLog, Config, InMemoryAdAccount, KillSwitch
from ads_guardrail.executor import replay_audit

from .agent.bandit import choose, posterior, posteriors
from .agent.hypotheses import Candidate, hypothesize, observe
from .agent.measure import measure_entity, measure_landing
from .config import LoopConfig
from .hands.ads import AdsHands
from .hands.creative import CreativeHands
from .hands.landing import LandingHands
from .journal import Journal, JournalState, tick_time
from .platform import PlatformPaths
from .warehouse import Warehouse
from .world import starting_entities


@dataclass(frozen=True)
class LoopPaths:
    root: Path

    @property
    def journal(self) -> Path:
        return self.root / "journal.jsonl"

    @property
    def audit(self) -> Path:
        return self.root / "ads-audit.jsonl"

    @property
    def warehouse(self) -> Path:
        return self.root / "warehouse.duckdb"

    @property
    def landing(self) -> Path:
        return self.root / "landing"

    @property
    def kill_file(self) -> Path:
        return self.root / "KILL_SWITCH"


@dataclass
class TickResult:
    day: date
    status: str  # ran | noop
    decisions: list[dict[str, Any]] = field(default_factory=list)
    measured: list[dict[str, Any]] = field(default_factory=list)
    readout: dict[str, Any] | None = None
    halted: str | None = None


SLOTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ads", ("budget_shift", "bid_up", "bid_down", "creative_refresh")),
    ("site", ("landing_test",)),
)


def decision_seed(seed: int, day: date, slot: str) -> list[int]:
    return [int(seed), day.toordinal(), 1 if slot == "ads" else 2]


class Loop:
    """Wires TRUTH (warehouse), the agent, and HANDS (ads, landing, creative) together."""

    def __init__(self, root: str | Path, cfg: LoopConfig, guard: Config, brief: str | Path, *, seed: int,
                 source, build_sites: bool = False, account=None) -> None:
        self.paths = LoopPaths(Path(root))
        self.paths.root.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.guard = guard
        self.seed = seed
        self.source = source
        self.journal = Journal(self.paths.journal, cfg.tick_hour_utc)
        self.audit = AuditLog(self.paths.audit)
        self.kill_switch = KillSwitch(self.paths.kill_file, guard.kill_switch.env_var)
        self._account = account
        plat = PlatformPaths(self.paths.root)
        self.landing = LandingHands(brief, self.paths.landing, plat.site, build=build_sites)
        self.creative = CreativeHands(plat.creatives, cfg.measure.creative_launch_lag_days)

    # --- the ad account (a fake one, rebuilt from the guardrail audit log) --------------------
    def account(self):
        if self._account is not None:
            return self._account
        acct = InMemoryAdAccount(starting_entities())
        replay_audit(acct, self.audit)
        return acct

    # --- one tick ---------------------------------------------------------------------------
    def tick(self, day: date) -> TickResult:
        js = self.journal.state()
        if day in js.finished_ticks:
            return TickResult(day, "noop")
        if js.last_tick is not None and day <= js.last_tick:
            raise ValueError(f"tick {day} is before the last finished tick {js.last_tick}")
        res = TickResult(day, "ran")
        account = self.account()
        hands = AdsHands(account, self.guard, self.audit, self.kill_switch)
        if not any(r for r in js.steps_done if r[0] == day):
            self.journal.append("tick_started", day, policy=self.cfg.policy, seed=self.seed,
                                source=getattr(self.source, "label", "warehouse"))

        with Warehouse(self.paths.warehouse) as wh:
            self._step(day, "ingest", lambda: self._ingest(day, wh, account))
            self._step(day, "rollbacks", lambda: self._finish_rollbacks(day, hands))
            self._step(day, "readout", lambda: self._readout(day, wh, res))
            self._step(day, "measure", lambda: self._measure(day, wh, res))
            self._step(day, "decide", lambda: self._decide_and_act(day, wh, hands, res))
            wh.upsert_decisions(d.mirror_row() for d in self.journal.state().decisions.values())
        self.journal.append("tick_finished", day)
        return res

    def _step(self, day: date, name: str, fn) -> None:
        if (day, name) in self.journal.state().steps_done:
            return
        fn()
        self.journal.append("step_done", day, step=name)

    def _ingest(self, day: date, wh: Warehouse, account) -> None:
        if self.source is None:
            return  # live: external loaders fill the warehouse
        n = self.source.ingest_through(day - timedelta(days=1), wh, account)
        self.journal.append("ingested", day, days_loaded=n, through=(day - timedelta(days=1)).isoformat(),
                            source=self.source.label)

    def _finish_rollbacks(self, day: date, hands: AdsHands) -> None:
        pending = []
        done = set()
        for r in self.journal.records():
            if r["event"] == "acted" and r.get("details", {}).get("pending_rollback"):
                pending.append(r["details"]["pending_rollback"])
            elif r["event"] == "rolled_back":
                done.add(r["run_id"])
        for run_id in pending:
            if run_id in done or self.kill_switch.reason():
                continue
            restored, conflicts, halted = hands.finish_rollback(run_id, tick_time(day, self.cfg.tick_hour_utc))
            if not halted:
                self.journal.append("rolled_back", day, run_id=run_id, restored=restored, conflicts=conflicts)

    # --- landing test readouts ----------------------------------------------------------------
    def _readout(self, day: date, wh: Warehouse, res: TickResult) -> None:
        js = self.journal.state()
        exp = js.open_experiment
        if exp is None:
            return
        looks_done = {r["look_day"] for r in self.journal.records()
                      if r["event"] == "lp_readout" and r.get("key") == exp["key"]}
        look = LandingHands.next_look(exp, looks_done, day)
        if look is None:
            return
        lo, hi = LandingHands.window(exp, look)
        counts = wh.experiment_counts(exp["key"], lo, hi)
        r = LandingHands.read(exp, counts, look)
        self.journal.append("lp_readout", day, key=exp["key"], decision_id=exp["decision_id"], look_day=look,
                            status=r.status.value, winner=r.winner, leader=r.leader, headline=r.headline(),
                            counts={k: list(v) for k, v in counts.items()})
        res.readout = {"key": exp["key"], "look_day": look, "headline": r.headline()}
        is_last = look == exp["look_days"][-1]
        if not (r.decided or is_last):
            return
        conclusion = self.landing.conclude(exp, r, day, apply_writes=True)
        self.journal.append("lp_concluded", day, decision_id=exp["decision_id"], **conclusion)
        _, signups_day = wh.site_daily(day - timedelta(days=7), day - timedelta(days=1))
        d = js.decisions[exp["decision_id"]]
        m = measure_landing(conclusion, signups_day, self.cfg.agent)
        self._record_measurement(day, d.decision_id, d.arm, d.x, m, res)

    # --- measurement of ad changes --------------------------------------------------------------
    def _measure(self, day: date, wh: Warehouse, res: TickResult) -> None:
        js = self.journal.state()
        keys = sorted(e.key for e in starting_entities())
        for d in sorted(js.due(day), key=lambda d: d.decision_id):
            if d.arm == "landing_test":
                continue  # measured at its readout
            m = measure_entity(wh, d, js, keys, self.cfg.measure)
            self._record_measurement(day, d.decision_id, d.arm, d.x, m, res)

    def _record_measurement(self, day: date, decision_id: str, arm: str, x: float, m, res: TickResult) -> None:
        rec = self.journal.append("measured", day, decision_id=decision_id, arm=arm, x=x, y=round(m.y, 4),
                                  se=round(m.se, 4), method=m.method, controls=list(m.controls), windows=m.windows,
                                  detail=m.detail, counted=True)
        # the record's own seq orders evidence for replay
        st = self.journal.state()
        p = posterior(arm, st.observations, self.cfg.bandit)
        self.journal.append("policy_updated", day, evidence_seq=rec["seq"], **p.to_dict())
        res.measured.append({"decision_id": decision_id, "arm": arm, "y": m.y, "se": m.se, "method": m.method})

    # --- observe, hypothesize, choose, act --------------------------------------------------------
    def _decide_and_act(self, day: date, wh: Warehouse, hands: AdsHands, res: TickResult) -> None:
        """Two independent slots per tick: one ad change, and the landing page.

        They use different resources (campaigns vs site traffic), so they do not compete.
        Each slot is one Thompson draw over its own arms, logged with its rng seed.
        """
        snap = None
        for slot, arms in SLOTS:
            js = self.journal.state()
            did = f"d-{day:%Y%m%d}-{slot}"
            logged = any(r["event"] == "decision" and r.get("decision_id") == did for r in self.journal.records())
            if not logged:
                if snap is None:
                    snap = observe(wh, day, hands.live(), hands.writable(), self.cfg.agent)
                cands = [c for c in hypothesize(snap, wh, js, self.cfg.agent) if c.arm in arms]
                posts = posteriors(js.observations, self.cfg.bandit)
                seed = decision_seed(self.seed, day, slot)
                ch = choose(self.cfg.policy, {c.arm: c.estimate for c in cands}, posts, self.cfg.bandit,
                            np.random.default_rng(seed), locks={c.arm: len(c.treated) for c in cands},
                            hurdle=self.cfg.agent.min_effect_per_locked_campaign)
                picked: Candidate | None = next((c for c in cands if c.arm == ch.arm), None)
                self.journal.append(
                    "decision", day, decision_id=did, slot=slot, policy=self.cfg.policy, rule=ch.rule,
                    rng_seed=seed, locks={c.arm: len(c.treated) for c in cands},
                    hurdle=self.cfg.agent.min_effect_per_locked_campaign, observed=snap.summary() if slot == "ads" else {},
                    candidates=[c.to_dict() for c in cands], scores=ch.scores,
                    beliefs={a: posts[a].to_dict() for a in arms}, evidence_n=len(js.observations),
                    arm=picked.arm if picked else None, target=picked.target if picked else "",
                    hypothesis=picked.hypothesis if picked else "hold: no candidate scored above zero",
                    x=picked.estimate if picked else 0.0, treated=list(picked.treated) if picked else [],
                    params=picked.params if picked else {})
                res.decisions.append({"slot": slot, "arm": ch.arm, "target": picked.target if picked else "",
                                      "scores": ch.scores})
            d = self.journal.state().decisions.get(did)
            if d is None or any(r["event"] == "acted" and r.get("decision_id") == did for r in self.journal.records()):
                continue
            self._act(day, d, hands, res)

    def _act(self, day: date, d, hands: AdsHands, res: TickResult) -> None:
        params = next(r["params"] for r in self.journal.records()
                      if r["event"] == "decision" and r.get("decision_id") == d.decision_id)
        reason = self.kill_switch.reason()
        apply_writes = self.cfg.apply and self.guard.writes_enabled
        due = day + timedelta(days=self.cfg.measure.due_after)
        if reason:
            self.journal.append("halted", day, reason=reason, decision_id=d.decision_id)
            self.journal.append("acted", day, decision_id=d.decision_id, arm=d.arm, outcome="halted", treated=[],
                                changed=[], details={"halt_reason": reason})
            res.halted = reason
            return
        now = tick_time(day, self.cfg.tick_hour_utc)
        if d.arm in ("budget_shift", "bid_up", "bid_down"):
            r = hands.execute(d.arm, params, day, now, d.decision_id, d.hypothesis, apply_writes)
            if r.outcome == "halted":
                self.journal.append("halted", day, reason=r.details.get("halt_reason"), decision_id=d.decision_id)
                res.halted = r.details.get("halt_reason")
            ok = r.outcome in ("applied", "clamped")
            self.journal.append("acted", day, decision_id=d.decision_id, arm=d.arm, outcome=r.outcome,
                                treated=list(d.treated), changed=r.changed, effective=day.isoformat(),
                                measure_due=due.isoformat() if ok else None, details=r.details)
        elif d.arm == "creative_refresh":
            key = params["key"]
            out = self.creative.request(d.decision_id, key, d.target, day, d.hypothesis, apply_writes)
            ok = out["outcome"] == "queued"
            live_on = date.fromisoformat(out.get("live_on", day.isoformat()))
            self.journal.append("acted", day, decision_id=d.decision_id, arm=d.arm, outcome=out["outcome"],
                                treated=[key], changed=[key] if ok else [], effective=live_on.isoformat(),
                                measure_due=(live_on + timedelta(days=self.cfg.measure.due_after)).isoformat()
                                if ok else None, details={**out, "effective": live_on.isoformat()})
        elif d.arm == "landing_test":
            if not apply_writes:
                self.journal.append("acted", day, decision_id=d.decision_id, arm=d.arm, outcome="dry_run",
                                    treated=[], changed=[], details={})
                return
            info = self.landing.launch(d.decision_id, day, params.get("daily_visitors", 1000), apply_writes)
            self.journal.append("lp_launched", day, decision_id=d.decision_id, **{
                k: info[k] for k in ("key", "arms", "control", "weights", "look_days", "started")})
            self.journal.append("acted", day, decision_id=d.decision_id, arm=d.arm, outcome="launched", treated=[],
                                changed=[], details=info)
        else:  # pragma: no cover
            raise ValueError(d.arm)


def load_guard_config(path: str | Path) -> Config:
    from ads_guardrail import load_config
    return load_config(path)


def dump(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str)


def state_summary(js: JournalState) -> dict[str, Any]:
    by_arm: dict[str, int] = {}
    for d in js.decisions.values():
        by_arm[d.arm] = by_arm.get(d.arm, 0) + 1
    return {"ticks": len(js.finished_ticks), "decisions": len(js.decisions), "by_arm": by_arm,
            "evidence": len(js.observations), "open_experiment": js.open_experiment, "halts": len(js.halts)}
