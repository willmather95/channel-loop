"""Loop configuration, loaded from TOML (config/loop.toml).

The loop has its own rules on top of the guardrails in ads-guardrail-agent and the stats
gates in landing-page-lab. Those two repos keep their own caps; this file only sets what
the loop itself decides: how it measures, how it scores, and how much it may have open.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

POLICIES: tuple[str, ...] = ("loop", "loop_ts", "greedy", "random", "static")
ARMS: tuple[str, ...] = ("budget_shift", "bid_up", "bid_down", "creative_refresh", "landing_test")


@dataclass(frozen=True)
class MeasureConfig:
    pre_days: int = 7  # baseline window before the change
    lag_days: int = 1  # days skipped after the change (conversion lag, learning)
    post_days: int = 7  # outcome window
    creative_launch_lag_days: int = 3  # a refresh goes live this many days after it is requested
    bootstrap_reps: int = 400
    min_controls: int = 1  # fewer clean controls than this = pre/post only, flagged, wider error

    @property
    def due_after(self) -> int:
        """Days after the action day when the post window is complete and measurable."""
        return self.lag_days + self.post_days + 1


@dataclass(frozen=True)
class BanditConfig:
    # Bayesian linear model per arm, in units of $100 of profit per day:
    #   measured_effect = w0 + w1 * hypothesis_estimate + noise
    prior_intercept_mean: float = 0.0
    prior_intercept_sd: float = 0.3
    prior_slope_mean: float = 1.0  # start by trusting my own estimates; evidence bends it
    prior_slope_sd: float = 0.4
    model_noise_sd: float = 0.3  # extra noise on top of each measurement's own standard error
    scale: float = 100.0  # dollars per model unit


@dataclass(frozen=True)
class AgentConfig:
    value_per_conversion: float = 60.0  # assumed value of one trial signup, in dollars
    lookback_days: int = 14
    elasticity_prior: float = 0.7  # the agent's guess: marginal return = 0.7 x average return
    shift_amount: float = 60.0  # budget shift the loop asks for; guardrails may cut it
    shift_min_gap: float = 0.25  # min marginal-ROI gap for a shift
    bid_up_pct: float = 0.15
    bid_down_pct: float = 0.10
    underpacing_utilization: float = 0.80
    saturated_utilization: float = 0.95
    refresh_ctr_ratio: float = 0.85  # refresh when CTR fell below 85% of its reference
    refresh_cooldown_days: int = 21
    landing_cooldown_days: int = 3
    landing_prior_lift: float = 0.05  # expected value of a test, as a fraction of site signups
    max_open_entity_measurements: int = 3
    # An ad change locks its campaigns until it is measured (about 9 days). It must be worth
    # at least this much per locked campaign per day, or the loop keeps the capacity free.
    min_effect_per_locked_campaign: float = 15.0


@dataclass(frozen=True)
class LoopConfig:
    agent: AgentConfig = field(default_factory=AgentConfig)
    measure: MeasureConfig = field(default_factory=MeasureConfig)
    bandit: BanditConfig = field(default_factory=BanditConfig)
    policy: str = "loop"  # loop | loop_ts | greedy | random | static
    apply: bool = False  # caller half of the write switch; guardrail config holds the other half
    tick_hour_utc: int = 6

    def with_policy(self, policy: str) -> "LoopConfig":
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy!r}")
        return replace(self, policy=policy)


def _section(cls: type, raw: dict[str, Any]) -> Any:
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown keys in [{cls.__name__}]: {sorted(unknown)}")
    return cls(**raw)


def load_loop_config(path: str | Path | None) -> LoopConfig:
    if path is None:
        return LoopConfig()
    data = tomllib.loads(Path(path).read_text())
    cfg = LoopConfig(
        agent=_section(AgentConfig, data.get("agent", {})),
        measure=_section(MeasureConfig, data.get("measure", {})),
        bandit=_section(BanditConfig, data.get("bandit", {})),
        policy=str(data.get("loop", {}).get("policy", "loop")),
        apply=bool(data.get("loop", {}).get("apply", False)),
    )
    return cfg.with_policy(cfg.policy)
