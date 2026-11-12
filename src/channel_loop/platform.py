"""State of the outside systems the loop's HANDS touch, other than the ad account.

In live use these are real systems: the feature-flag service that routes landing-page
traffic, the site host, and the creative queue a person approves. In this repo they are
small JSON files under `<state>/platform/`. The HANDS write them; the SIMULATED market
reads them to decide what visitors see. The agent never reads them directly. It only
sees their effects in the warehouse.

The ad account itself is ads-guardrail-agent's in-memory account, rebuilt each tick by
replaying that repo's audit log.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


@dataclass
class LiveExperiment:
    key: str
    arms: list[str]
    weights: list[float]  # percent, as the flag splits traffic (e.g. 34/33/33)
    control: str
    started: str  # ISO date of the first day with split traffic


@dataclass
class SiteState:
    """What the landing page serves: the production page, plus a split test if one is live."""

    production: str = "original"  # label of the page in production
    experiment: LiveExperiment | None = None
    shipped: list[dict[str, Any]] = field(default_factory=list)  # [{key, variant, on}]

    @classmethod
    def load(cls, path: Path) -> "SiteState":
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text())
        exp = raw.get("experiment")
        return cls(production=raw.get("production", "original"),
                   experiment=LiveExperiment(**exp) if exp else None,
                   shipped=list(raw.get("shipped", [])))

    def save(self, path: Path) -> None:
        _atomic_write(path, asdict(self))


@dataclass
class CreativeRequest:
    request_id: str
    entity_key: str
    requested_on: str
    live_on: str  # first day the new creative serves
    brief: str
    status: str = "queued"  # queued -> live


@dataclass
class CreativeState:
    """Live creative per campaign, plus refresh requests waiting to go live."""

    live: dict[str, dict[str, str]] = field(default_factory=dict)  # entity_key -> {creative_id, launched_on}
    requests: list[CreativeRequest] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "CreativeState":
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text())
        return cls(live=dict(raw.get("live", {})), requests=[CreativeRequest(**r) for r in raw.get("requests", [])])

    def save(self, path: Path) -> None:
        _atomic_write(path, {"live": self.live, "requests": [asdict(r) for r in self.requests]})

    def promote_due(self, day: date) -> list[CreativeRequest]:
        """Requests whose go-live day has come become the live creative. Returns what launched."""
        launched = []
        for r in self.requests:
            if r.status == "queued" and date.fromisoformat(r.live_on) <= day:
                r.status = "live"
                self.live[r.entity_key] = {"creative_id": r.request_id, "launched_on": r.live_on}
                launched.append(r)
        return launched


@dataclass(frozen=True)
class PlatformPaths:
    root: Path

    @property
    def site(self) -> Path:
        return self.root / "platform" / "site.json"

    @property
    def creatives(self) -> Path:
        return self.root / "platform" / "creatives.json"
