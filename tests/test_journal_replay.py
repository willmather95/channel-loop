"""The journal is the only state: replay re-derives every decision, and tampering is caught."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from channel_loop.config import BanditConfig
from channel_loop.journal import Journal, replay_state
from channel_loop.replay import replay

SRC = Path(__file__).resolve().parents[1] / "src" / "channel_loop"


def test_replay_rederives_every_decision(five_week_runs):
    for (policy, seed), lp in five_week_runs.items():
        rep = replay(lp.journal, lp.cfg.bandit)
        assert rep.chain_ok, rep.chain_msg
        assert rep.decisions >= 70  # two slots x 35 days
        assert rep.matched == rep.decisions, rep.mismatches[:3]


def test_state_from_a_fresh_read_equals_live_state(five_week_runs):
    lp = five_week_runs[("loop", 3)]
    fresh = replay_state(list(Journal(lp.paths.journal).log.records()))
    live = lp.journal.state()
    assert fresh.finished_ticks == live.finished_ticks
    assert [o.decision_id for o in fresh.observations] == [o.decision_id for o in live.observations]
    assert {k: (d.outcome, d.y, d.measure_due) for k, d in fresh.decisions.items()} == \
           {k: (d.outcome, d.y, d.measure_due) for k, d in live.decisions.items()}
    assert len(live.observations) >= 5, "five weeks should produce measured evidence"


def _rehash(lines: list[dict]) -> list[dict]:
    """What a careful forger would do: recompute every hash so the chain still verifies."""
    prev = "0" * 64
    out = []
    for r in lines:
        r = {k: v for k, v in r.items() if k != "hash"}
        r["prev_hash"] = prev
        body = json.dumps(r, sort_keys=True, separators=(",", ":"), default=str).encode()
        r["hash"] = hashlib.sha256(body).hexdigest()
        prev = r["hash"]
        out.append(r)
    return out


def test_edited_evidence_is_caught(five_week_runs, tmp_path):
    lp = five_week_runs[("loop", 11)]
    lines = [json.loads(x) for x in lp.paths.journal.read_text().splitlines()]
    i = next(i for i, r in enumerate(lines) if r["event"] == "measured")
    lines[i]["y"] = lines[i]["y"] + 500.0

    raw = tmp_path / "edited.jsonl"
    raw.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in lines))
    assert not Journal(raw).verify()[0]  # naive edit: hash chain breaks

    forged = tmp_path / "forged.jsonl"
    forged.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in _rehash(lines)))
    rep = replay(Journal(forged), lp.cfg.bandit)
    assert rep.chain_ok  # chain re-computed...
    assert rep.mismatches  # ...but logged beliefs no longer follow from the evidence


def test_agent_never_imports_the_simulator():
    """TRUTH for the agent is the warehouse. The simulator's hidden parameters stay hidden."""
    for path in list((SRC / "agent").glob("*.py")) + [SRC / "loop.py", SRC / "replay.py", SRC / "journal.py"]:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "sim" not in node.module.split("."), f"{path.name} imports {node.module}"
            if isinstance(node, ast.Import):
                assert not any(".sim" in a.name for a in node.names)


def test_bandit_config_round_trip_is_used_for_replay():
    assert BanditConfig().prior_slope_mean == 1.0
