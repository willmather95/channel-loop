"""HANDS for ad changes, through ads-guardrail-agent and nothing else.

Every budget or bid change the loop wants becomes an `external_proposal` in that repo and
goes through its 12 guardrails, its two-key write switch, its kill switch and its
hash-chained audit log. This module never writes to an adapter directly.

A budget shift is two proposals (cut one campaign, raise another). The guardrails can clamp
each side differently, so the loop sizes both sides to the smaller allowed amount first,
then applies. If only one side lands (drift, a write failure, the kill switch), the run is
rolled back with ads-guardrail's own rollback so a half shift never stays live.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from ads_guardrail import AuditLog, ChangeType, Config, KillSwitch, Plan, Verdict, apply, rollback, run_id_for
from ads_guardrail.executor import external_proposal
from ads_guardrail.guardrails import allowed, decide
from ads_guardrail.models import usd


@dataclass
class AdsResult:
    outcome: str  # applied | clamped | blocked | dry_run | halted
    changed: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


class AdsHands:
    def __init__(self, adapter, config: Config, audit: AuditLog, kill_switch: KillSwitch) -> None:
        self.adapter = adapter
        self.config = config
        self.audit = audit
        self.kill_switch = kill_switch

    def live(self) -> dict[str, tuple[float, float | None]]:
        return {e.key: (e.daily_budget_micros / 1e6, None if e.bid_micros is None else e.bid_micros / 1e6)
                for e in self.adapter.list_entities()}

    def writable(self) -> set[str]:
        return {e.key for e in self.adapter.list_entities() if allowed(self.config.guardrails, e.key)}

    def _entity(self, key: str):
        return next(e for e in self.adapter.list_entities() if e.key == key)

    def _proposals(self, changes: list[tuple[str, ChangeType, int]], day: date, source: str, reason: str,
                   confidence: float):
        return [external_proposal(self._entity(k), ct, v, reason=reason, as_of=day, config=self.config,
                                  source=source, confidence=confidence) for k, ct, v in changes]

    def execute(self, arm: str, params: dict[str, Any], day: date, now: datetime, decision_id: str,
                reason: str, apply_writes: bool, confidence: float = 0.8) -> AdsResult:
        source = f"channel-loop:{decision_id}"
        run_id = run_id_for(day, self.config)
        state = self.audit.state()
        already = [r for r in state.applied if r["run_id"] == run_id]
        if run_id in state.finished_runs:  # re-run of a finished tick: report what is already live
            return AdsResult("applied" if already else "blocked", sorted({r["entity_key"] for r in already}),
                             {"run_id": run_id, "note": "run already finished", "writes": len(already)})

        if arm == "budget_shift":
            src, dst = self._entity(params["src"]), self._entity(params["dst"])
            amt = usd(params["amount"])
            ask = [(src.key, ChangeType.BUDGET, src.daily_budget_micros - amt),
                   (dst.key, ChangeType.BUDGET, dst.daily_budget_micros + amt)]
        elif arm in ("bid_up", "bid_down"):
            e = self._entity(params["key"])
            if e.bid_micros is None:
                return AdsResult("blocked", details={"why": "no bid to change"})
            new = int(round(e.bid_micros * (1 + params["pct"])))
            ask = [(e.key, ChangeType.BID, new)]
        else:
            raise ValueError(f"AdsHands cannot do {arm}")

        props = self._proposals(ask, day, source, reason, confidence)
        pre = decide(props, self.adapter.list_entities(), self.config.guardrails, state, now, self.adapter.quantum_micros)
        if any(d.verdict is Verdict.BLOCKED or not d.writable for d in pre):
            return AdsResult("blocked", details={"run_id": run_id, "notes": [list(d.notes) for d in pre],
                                                 "asked": [p.to_dict() for p in props]})
        clamped = any(d.verdict is Verdict.CLAMPED for d in pre)
        if arm == "budget_shift":  # equalize both sides to the smaller allowed move
            moved = min(abs(d.final_delta_micros) for d in pre)
            ask = [(src.key, ChangeType.BUDGET, src.daily_budget_micros - moved),
                   (dst.key, ChangeType.BUDGET, dst.daily_budget_micros + moved)]
            props = self._proposals(ask, day, source, reason, confidence)
            pre = decide(props, self.adapter.list_entities(), self.config.guardrails, state, now,
                         self.adapter.quantum_micros)
            if any(not d.writable for d in pre):
                return AdsResult("blocked", details={"run_id": run_id, "notes": [list(d.notes) for d in pre]})
        plan = Plan(run_id=run_id, as_of=day, proposals=props, decisions=pre, holds=[])
        guard_view = [{"entity_key": d.proposal.entity_key, "change": d.proposal.change_type.value,
                       "before": d.proposal.before_micros / 1e6, "asked": d.proposal.proposed_micros / 1e6,
                       "allowed": d.final_micros / 1e6, "verdict": d.verdict.value, "notes": list(d.notes)}
                      for d in pre]
        res = apply(plan, self.adapter, self.config, self.audit, apply=apply_writes, now=now,
                    kill_switch=self.kill_switch)
        details = {"run_id": run_id, "guardrails": guard_view, "clamped_by_loop_or_caps": clamped}
        if res.mode == "dry_run":
            return AdsResult("dry_run", details=details)
        if res.halted:
            # The kill switch also stops rollbacks. A half shift is undone on the first tick
            # after the switch is released (see Loop._finish_rollbacks).
            pending = run_id if res.applied else None
            return AdsResult("halted", sorted({r["entity_key"] for r in res.applied}),
                             {**details, "halt_reason": res.halt_reason, "pending_rollback": pending})
        if len(res.applied) != len(props):
            if res.applied:
                rollback(run_id, self.adapter, self.config, self.audit, apply=True, now=now,
                         kill_switch=self.kill_switch)
            return AdsResult("blocked", details={**details, "skipped": res.skipped, "rolled_back": bool(res.applied)})
        details["writes"] = [{"entity_key": r["entity_key"], "before": r["before_micros"] / 1e6,
                              "after": r["after_micros"] / 1e6} for r in res.applied]
        return AdsResult("clamped" if clamped else "applied", sorted({r["entity_key"] for r in res.applied}), details)

    def finish_rollback(self, run_id: str, now: datetime) -> tuple[int, int, bool]:
        """Undo a half-applied run. Returns (restored, conflicts, halted)."""
        r = rollback(run_id, self.adapter, self.config, self.audit, apply=True, now=now, kill_switch=self.kill_switch)
        return len(r.restored), len(r.conflicts), r.halted
