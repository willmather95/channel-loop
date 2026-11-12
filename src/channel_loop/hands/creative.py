"""HANDS for creative refreshes: a queue, not a generator.

The loop only asks for a refresh. The request names the campaign, the reason and a short
brief that ugc-content-machine (or a person) can work from. New creative goes live after a
production lag. In live use a person approves it before upload. In the simulator the
approval is simulated and the creative goes live on its `live_on` day.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

from ..platform import CreativeRequest, CreativeState


class CreativeHands:
    def __init__(self, path: Path, lag_days: int, max_open: int = 2) -> None:
        self.path = path
        self.lag_days = lag_days
        self.max_open = max_open

    def request(self, decision_id: str, entity_key: str, name: str, day: date, reason: str,
                apply_writes: bool) -> dict[str, Any]:
        st = CreativeState.load(self.path)
        if any(r.request_id == decision_id for r in st.requests):  # idempotent re-run
            r = next(r for r in st.requests if r.request_id == decision_id)
            return {"outcome": "queued", "request_id": r.request_id, "live_on": r.live_on, "note": "already queued"}
        open_n = sum(1 for r in st.requests if r.status == "queued")
        if open_n >= self.max_open:
            return {"outcome": "blocked", "why": f"{open_n} refreshes already waiting (max {self.max_open})"}
        live_on = day + timedelta(days=self.lag_days)
        brief = (f"Refresh for '{name}': same offer and landing page, new hook and first frame. "
                 f"Why: {reason} Draft with ugc-content-machine; a person approves before upload.")
        if not apply_writes:
            return {"outcome": "dry_run", "live_on": live_on.isoformat(), "brief": brief}
        st.requests.append(CreativeRequest(decision_id, entity_key, day.isoformat(), live_on.isoformat(), brief))
        st.save(self.path)
        return {"outcome": "queued", "request_id": decision_id, "live_on": live_on.isoformat(), "brief": brief,
                "approval": "simulated (a person approves creative in live use)"}
