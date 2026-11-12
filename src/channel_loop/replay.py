"""Journal replay: re-derive every past decision from the journal alone and check it matches.

For each logged decision, take only the evidence (`measured` lines) that came before it,
rebuild each arm's posterior, re-run the same choice rule with the logged candidates,
rng seed, locks and hurdle, and compare: the chosen arm, every score, and every belief
must match what was logged. Any edit to evidence, beliefs or choices shows up here, and
the hash chain catches edits to anything else.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .agent.bandit import choose, posteriors
from .config import BanditConfig
from .journal import Journal, Observation


@dataclass
class ReplayReport:
    chain_ok: bool
    chain_msg: str
    decisions: int = 0
    matched: int = 0
    mismatches: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.chain_ok and not self.mismatches

    def summary(self) -> str:
        return (f"chain: {self.chain_msg}; decisions re-derived: {self.matched}/{self.decisions} match"
                + ("" if not self.mismatches else f"; {len(self.mismatches)} MISMATCH"))


def replay(journal: Journal, cfg: BanditConfig, tol: float = 1e-6) -> ReplayReport:
    ok, msg = journal.verify()
    rep = ReplayReport(ok, msg)
    obs: list[Observation] = []
    arms_by_decision: dict[str, str] = {}
    for r in journal.records():
        ev = r["event"]
        if ev == "decision":
            arms_by_decision[r["decision_id"]] = r.get("arm") or ""
            rep.decisions += 1
            posts = posteriors(obs, cfg)
            cands = {c["arm"]: c["estimate"] for c in r["candidates"]}
            ch = choose(r["policy"], cands, posts, cfg, np.random.default_rng(r["rng_seed"]),
                        locks=r.get("locks", {}), hurdle=r.get("hurdle", 0.0))
            problems = []
            if ch.arm != r.get("arm"):
                problems.append(f"arm {ch.arm!r} != logged {r.get('arm')!r}")
            for a, s in r["scores"].items():
                if abs(ch.scores.get(a, float("nan")) - s) > tol:
                    problems.append(f"score[{a}] {ch.scores.get(a)} != logged {s}")
            for a, b in r["beliefs"].items():
                mine = posts[a].to_dict()
                if mine["n"] != b["n"] or abs(mine["w0"] - b["w0"]) > tol or abs(mine["w1"] - b["w1"]) > tol:
                    problems.append(f"belief[{a}] {mine} != logged {b}")
            if problems:
                rep.mismatches.append({"decision_id": r["decision_id"], "problems": problems})
            else:
                rep.matched += 1
        elif ev == "measured" and r.get("counted"):
            obs.append(Observation(r["arm"], float(r["x"]), float(r["y"]), float(r["se"]), r["decision_id"], r["seq"]))
    return rep
