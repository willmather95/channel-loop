"""SIMULATED data source: plays the part of the ad platforms, the site and the nightly loaders.

Each tick for day D, the warehouse needs day D-1. This source simulates any missing days
with whatever settings are live at that moment (budgets and bids from the ad account,
the landing-page split from site.json, creatives from creatives.json), then loads them into
DuckDB the same way a nightly loader would. The first call also builds a 28-day warm-up
history so the agent has data on day one.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from ..platform import CreativeState, PlatformPaths, SiteState
from ..warehouse import Warehouse
from .market import Market


class SimSource:
    label = "SIMULATED"

    def __init__(self, root: Path, seed: int, start: date, horizon_days: int, value_per_conversion: float) -> None:
        self.root = root
        self.path = root / "sim" / "market.json"
        self.paths = PlatformPaths(root)
        self.seed, self.start, self.horizon, self.value = seed, start, horizon_days, value_per_conversion
        self._market: Market | None = None

    @property
    def market(self) -> Market:
        if self._market is None:
            if self.path.exists():
                self._market = Market.load(self.path)
            else:
                self._market = Market(self.seed, self.start, self.horizon, value_per_conversion=self.value)
                if not self.paths.creatives.exists():
                    self._market.initial_creatives().save(self.paths.creatives)
                if not self.paths.site.exists():
                    SiteState().save(self.paths.site)
        return self._market

    def ingest_through(self, last: date, wh: Warehouse, account) -> int:
        """Simulate and load every day up to and including `last`. Returns days loaded."""
        m = self.market
        day = m.origin if m.last_day is None else m.last_day + timedelta(days=1)
        n = 0
        while day <= last:
            site = SiteState.load(self.paths.site)
            creatives = CreativeState.load(self.paths.creatives)
            if creatives.promote_due(day):
                creatives.save(self.paths.creatives)
            data = m.simulate_day(day, account, site, creatives)
            wh.load_day(day, data["ads"], data["settings"], data["web"])
            m.save(self.path)  # after the load: a crash in between re-simulates the same day identically
            day += timedelta(days=1)
            n += 1
        return n
