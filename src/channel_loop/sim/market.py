"""SIMULATED market. Every number it produces is made up. The agent never imports this module.

What it models, per campaign and day:

- Diminishing returns: clicks grow like spend ** beta, with a hidden beta per campaign
  (0.30 to 0.90). The agent assumes one elasticity for all, so its estimates are biased.
- Bids: a hidden "clearing bid" per campaign. Bid below it and the campaign under-spends
  (logistic pacing curve). Bid far above it and each click costs more.
- Creative fatigue: CTR and clicks per dollar decay with creative age, with a hidden time
  constant (short on Meta, long on Search). A refresh gets a new hidden quality draw and
  goes live after a production lag.
- Landing page: each test arm has a hidden true lift vs the current page. Shipping a
  winner changes every later visitor's conversion rate by that hidden amount.
- Seasonality and noise: weekday pattern, a slow trend, a Thanksgiving-week dip, a shared
  daily demand shock (AR(1)), a campaign-level shock (AR(1)), binomial and Poisson counts.
- Lag: each signup is reported by the ad platform 0, 1 or 2 days later (65/25/10%).
- Learning reset: a budget or bid move over 25% costs 15% efficiency for 5 days.

Noise for a given (seed, day, campaign) is the same no matter what the agent does
(common random numbers), so policies can be compared seed by seed.
"""
from __future__ import annotations

import json
import math
import zlib
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from ..platform import CreativeState, SiteState
from ..world import CAMPAIGNS, LANDING_PAGE, key_of

WEEKDAY = (1.06, 1.08, 1.06, 1.02, 0.95, 0.86, 0.90)  # Mon..Sun demand
SPEND_WEEKDAY = (1.0, 1.0, 1.0, 1.0, 0.97, 0.92, 0.93)
HOLIDAYS = {(11, 25): 0.85, (11, 26): 0.60, (11, 27): 0.72, (11, 28): 0.82}
LAG_KERNEL = (0.65, 0.25, 0.10)
SESSION_RATE = 0.92  # share of paid clicks that become a landing-page session
ORGANIC_SESSIONS = 900.0
ORGANIC_CR = 0.035

RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "brand": dict(cpc=(1.2, 1.8), beta=(0.30, 0.45), cr=(0.10, 0.14), ctr=(0.10, 0.14), tau=(200, 300),
                  age=(30, 90), clear=(1.40, 1.60)),
    "search": dict(cpc=(1.8, 3.0), beta=(0.55, 0.90), cr=(0.045, 0.095), ctr=(0.04, 0.07), tau=(80, 160),
                   age=(20, 80), clear=(1.38, 1.60)),
    "social": dict(cpc=(0.8, 1.3), beta=(0.60, 0.90), cr=(0.018, 0.042), ctr=(0.008, 0.014), tau=(18, 40),
                   age=(15, 45), clear=(1.38, 1.60)),
    "retarget": dict(cpc=(0.9, 1.4), beta=(0.35, 0.60), cr=(0.05, 0.10), ctr=(0.010, 0.020), tau=(15, 30),
                     age=(10, 35), clear=(1.38, 1.60)),
}


@dataclass(frozen=True)
class Hidden:
    """The true response of one campaign. SIMULATED and hidden from the agent."""

    key: str
    cpc0: float
    beta: float
    cr: float
    ctr0: float
    clear_bid: float  # bid where pacing is 50%-ish; starting bid / clearing ratio
    tau: float  # creative fatigue time constant, days
    start_age: int  # age of the live creative on the first warm-up day
    s0: float  # reference spend (starting budget)


def _u(rng: np.random.Generator, lo_hi: tuple[float, float]) -> float:
    return float(rng.uniform(*lo_hi))


def utilization(bid: float, clear_bid: float) -> float:
    """Share of budget the platform can spend at this bid (logistic pacing curve)."""
    x = bid / clear_bid
    return float(min(1.0, max(0.05, 1.0 / (1.0 + math.exp(-(x - 1.1) / 0.1)))))


def cpc_multiplier(bid: float, clear_bid: float) -> float:
    """Bidding well above the clearing bid buys the same clicks at a higher price."""
    return float((max(bid / clear_bid, 0.5) / 1.45) ** 0.35)


def fatigue(age: int, tau: float, quality: float) -> float:
    return quality * (0.6 + 0.4 * math.exp(-max(age, 0) / tau))


class Market:
    def __init__(self, seed: int, start: date, horizon_days: int = 84, warmup_days: int = 28,
                 value_per_conversion: float = 60.0) -> None:
        self.seed = int(seed)
        self.start = start
        self.horizon_days = horizon_days
        self.warmup_days = warmup_days
        self.origin = start - timedelta(days=warmup_days)
        self.value = value_per_conversion
        self.n_days = warmup_days + horizon_days + 30
        rng = np.random.default_rng([self.seed, 0])
        self.hidden: dict[str, Hidden] = {}
        under = set(rng.choice([i for i, c in enumerate(CAMPAIGNS) if c.kind != "brand"], size=2, replace=False).tolist())
        for i, c in enumerate(CAMPAIGNS):
            r = RANGES[c.kind]
            ratio = _u(rng, (1.06, 1.18)) if i in under else _u(rng, r["clear"])
            self.hidden[key_of(c)] = Hidden(
                key=key_of(c), cpc0=_u(rng, r["cpc"]), beta=_u(rng, r["beta"]), cr=_u(rng, r["cr"]),
                ctr0=_u(rng, r["ctr"]), clear_bid=c.bid / ratio, tau=_u(rng, r["tau"]),
                start_age=int(_u(rng, r["age"])), s0=float(c.budget))
        self.common = self._ar1(np.random.default_rng([self.seed, 1]), self.n_days, 0.06, 0.4)
        self.idio = {key_of(c): self._ar1(np.random.default_rng([self.seed, 2, i]), self.n_days, 0.12, 0.5)
                     for i, c in enumerate(CAMPAIGNS)}
        self.organic_shock = self._ar1(np.random.default_rng([self.seed, 3]), self.n_days, 0.08, 0.5)
        # mutable state
        self.last_day: date | None = None
        self.page_mult = 1.0
        self.exp_mults: dict[str, dict[str, float]] = {}
        self.applied_ships: list[str] = []
        self.ship_days: dict[str, str] = {}
        self.pending: dict[str, dict[str, int]] = {k: {} for k in self.hidden}
        self.history: dict[str, dict[str, Any]] = {}
        self.learning_until: dict[str, str] = {}

    # ---------------------------------------------------------------------------------------
    @staticmethod
    def _ar1(rng: np.random.Generator, n: int, sd: float, phi: float) -> np.ndarray:
        """Mean-one multiplicative AR(1) shocks."""
        eps = rng.normal(0.0, sd * math.sqrt(1 - phi * phi), size=n)
        z = np.zeros(n)
        z[0] = rng.normal(0.0, sd)
        for t in range(1, n):
            z[t] = phi * z[t - 1] + eps[t]
        return np.exp(z - 0.5 * sd * sd)

    def idx(self, day: date) -> int:
        i = (day - self.origin).days
        if not 0 <= i < self.n_days:
            raise ValueError(f"{day} is outside the simulated range")
        return i

    def demand(self, day: date) -> float:
        week = max(0, (day - self.start).days) / 7.0
        return WEEKDAY[day.weekday()] * (1 + 0.004 * week) * HOLIDAYS.get((day.month, day.day), 1.0)

    def initial_creatives(self) -> CreativeState:
        st = CreativeState()
        for k, h in self.hidden.items():
            launched = self.origin - timedelta(days=h.start_age)
            st.live[k] = {"creative_id": f"cr0-{k.rsplit(':', 1)[1]}", "launched_on": launched.isoformat()}
        return st

    def creative_quality(self, creative_id: str) -> float:
        if creative_id.startswith("cr0-"):
            return 1.0
        rng = np.random.default_rng([self.seed, 4, zlib.crc32(creative_id.encode())])
        return float(np.exp(rng.normal(0.0, 0.10)))

    def arm_mults(self, site: SiteState, day: date) -> tuple[str, list[str], list[float], list[float]]:
        """(experiment key or '', arm labels, traffic weights, true conversion multipliers)."""
        exp = site.experiment
        if exp is None or date.fromisoformat(exp.started) > day:
            return "", [site.production], [1.0], [self.page_mult]
        if exp.key not in self.exp_mults:
            rng = np.random.default_rng([self.seed, 5, zlib.crc32(exp.key.encode())])
            mu = 0.03 - 0.02 * len(self.applied_ships)
            self.exp_mults[exp.key] = {a: (1.0 if a == exp.control else
                                           float(1.0 + np.clip(rng.normal(mu, 0.10), -0.25, 0.35))) for a in exp.arms}
        w = np.array(exp.weights, dtype=float)
        w = w / w.sum()
        return exp.key, list(exp.arms), w.tolist(), [self.page_mult * self.exp_mults[exp.key][a] for a in exp.arms]

    def _apply_ships(self, site: SiteState, day: date) -> None:
        for s in site.shipped:
            tag = f"{s['key']}:{s['variant']}"
            if tag in self.applied_ships or date.fromisoformat(s["on"]) > day:
                continue
            mult = self.exp_mults.get(s["key"], {}).get(s["variant"], 1.0)
            self.page_mult *= mult
            self.applied_ships.append(tag)
            self.ship_days[tag] = day.isoformat()

    # --- the expected (noise-free given realized shocks) response, used for hidden truth ----
    def expected(self, key: str, day: date, budget: float, bid: float, age: int, quality: float,
                 page_mult: float, learning: float = 1.0) -> tuple[float, float, float]:
        """(spend, clicks, signups) expected for one campaign-day at these settings."""
        h = self.hidden[key]
        i = self.idx(day)
        spend = budget * utilization(bid, h.clear_bid) * SPEND_WEEKDAY[day.weekday()]
        if spend <= 0:
            return 0.0, 0.0, 0.0
        cpc = h.cpc0 * cpc_multiplier(bid, h.clear_bid)
        clicks = (spend / cpc) * (spend / h.s0) ** (h.beta - 1.0) * self.demand(day) * self.common[i] \
            * self.idio[key][i] * fatigue(age, h.tau, quality) * learning
        signups = clicks * SESSION_RATE * min(0.9, h.cr * page_mult)
        return spend, clicks, signups

    def expected_organic(self, day: date, page_mult: float) -> float:
        i = self.idx(day)
        return ORGANIC_SESSIONS * self.demand(day) * self.common[i] * self.organic_shock[i] * ORGANIC_CR * page_mult

    # --- one simulated day ------------------------------------------------------------------
    def simulate_day(self, day: date, account, site: SiteState, creatives: CreativeState) -> dict[str, list[dict]]:
        if self.last_day is not None and day != self.last_day + timedelta(days=1):
            raise ValueError(f"market is at {self.last_day}; next day must be {self.last_day + timedelta(days=1)}")
        if self.last_day is None and day != self.origin:
            raise ValueError(f"first simulated day must be {self.origin}")
        i = self.idx(day)
        self._apply_ships(site, day)
        exp_key, arms, weights, mults = self.arm_mults(site, day)
        eff_mult = float(np.dot(weights, mults))
        ads, settings, web = [], [], []
        web_acc = {a: [0, 0, 0, 0] for a in arms}  # paid sessions, paid signups, organic sessions, organic signups
        hist: dict[str, Any] = {"page_mult": self.page_mult, "eff_mult": eff_mult, "experiment": exp_key, "entities": {}}
        prev = self.history.get((day - timedelta(days=1)).isoformat())
        for j, c in enumerate(CAMPAIGNS):
            key = key_of(c)
            h = self.hidden[key]
            ent = account.entity(key)
            budget, bid = ent.daily_budget_micros / 1e6, (ent.bid_micros or 0) / 1e6
            cr_live = creatives.live[key]
            age = (day - date.fromisoformat(cr_live["launched_on"])).days
            q = self.creative_quality(cr_live["creative_id"])
            if prev is not None:
                pe = prev["entities"][key]
                if (abs(budget / pe["budget"] - 1) > 0.25) or (pe["bid"] and abs(bid / pe["bid"] - 1) > 0.25):
                    self.learning_until[key] = (day + timedelta(days=5)).isoformat()
            learning = 0.85 if self.learning_until.get(key, "") > day.isoformat() else 1.0
            rng = np.random.default_rng([self.seed, 10, i, j])
            spend, clicks_exp, _ = self.expected(key, day, budget, bid, age, q, 1.0, learning)
            spend = min(budget, spend * float(np.exp(rng.normal(0, 0.03))))
            clicks = int(rng.poisson(clicks_exp))
            ctr = h.ctr0 * fatigue(age, h.tau, q)
            impressions = int(round(clicks / max(ctr, 1e-6))) if clicks else int(round(spend / h.cpc0 / ctr))
            sessions = int(rng.binomial(clicks, SESSION_RATE))
            split = rng.multinomial(sessions, weights)
            true_signups = 0
            for a, n_s, m in zip(arms, split, mults):
                s = int(rng.binomial(int(n_s), min(0.9, h.cr * m)))
                web_acc[a][0] += int(n_s)
                web_acc[a][1] += s
                true_signups += s
            # platform-reported conversions arrive with a lag
            alloc = rng.multinomial(true_signups, LAG_KERNEL)
            for k, n in enumerate(alloc):
                if n:
                    d = (day + timedelta(days=k)).isoformat()
                    self.pending[key][d] = self.pending[key].get(d, 0) + int(n)
            reported = self.pending[key].pop(day.isoformat(), 0)
            ads.append(dict(platform=c.platform.value, account_id=c.account_id, entity_id=c.entity_id,
                            entity_name=c.name, date=day, spend=round(spend, 2), impressions=impressions,
                            clicks=clicks, conversions=float(reported),
                            conversion_value=round(reported * self.value, 2)))
            settings.append(dict(entity_key=key, date=day, daily_budget=budget, bid=bid,
                                 creative_id=cr_live["creative_id"], creative_launched_on=cr_live["launched_on"]))
            hist["entities"][key] = dict(budget=budget, bid=bid, creative_id=cr_live["creative_id"],
                                         launched_on=cr_live["launched_on"], learning=learning)
        orng = np.random.default_rng([self.seed, 11, i])
        org_sessions = int(orng.poisson(ORGANIC_SESSIONS * self.demand(day) * self.common[i] * self.organic_shock[i]))
        for a, n_s, m in zip(arms, orng.multinomial(org_sessions, weights), mults):
            web_acc[a][2] += int(n_s)
            web_acc[a][3] += int(orng.binomial(int(n_s), min(0.9, ORGANIC_CR * m)))
        for a, (ps, pc, os_, oc) in web_acc.items():
            web.append(dict(date=day, page=LANDING_PAGE, experiment_key=exp_key, variant=a, source="paid",
                            sessions=ps, signups=pc))
            web.append(dict(date=day, page=LANDING_PAGE, experiment_key=exp_key, variant=a, source="organic",
                            sessions=os_, signups=oc))
        self.history[day.isoformat()] = hist
        self.last_day = day
        return {"ads": ads, "settings": settings, "web": web}

    # --- persistence --------------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return dict(seed=self.seed, start=self.start.isoformat(), horizon_days=self.horizon_days,
                    warmup_days=self.warmup_days, value=self.value,
                    last_day=self.last_day.isoformat() if self.last_day else None, page_mult=self.page_mult,
                    exp_mults=self.exp_mults, applied_ships=self.applied_ships, ship_days=self.ship_days, pending=self.pending,
                    history=self.history, learning_until=self.learning_until,
                    hidden={k: asdict(h) for k, h in self.hidden.items()})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state_dict(), sort_keys=True))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "Market":
        d = json.loads(path.read_text())
        m = cls(d["seed"], date.fromisoformat(d["start"]), d["horizon_days"], d["warmup_days"], d["value"])
        m.last_day = date.fromisoformat(d["last_day"]) if d["last_day"] else None
        m.page_mult = d["page_mult"]
        m.exp_mults = d["exp_mults"]
        m.applied_ships = d["applied_ships"]
        m.ship_days = d.get("ship_days", {})
        m.pending = d["pending"]
        m.history = d["history"]
        m.learning_until = d["learning_until"]
        return m
