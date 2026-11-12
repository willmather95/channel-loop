"""Bayesian decision over action types: one small Bayesian linear model per arm, Thompson sampling.

For each arm the loop believes

    measured_effect / scale = w0 + w1 * (hypothesis_estimate / scale) + noise

w1 answers "how far can I trust my own estimates for this kind of change?" and w0 catches
a constant bias (for example, every bid cut costing a bit more than the estimate says).
The prior is Normal with w1 centered on 0.5 ("half-trust"). Each measurement arrives with
its own standard error, so a noisy diff-in-diff counts for less than a clean A/B readout.
The posterior is exact (conjugate Gaussian), so replaying the journal gives the same
beliefs, to the last digit, every time.

Two ways to act on the beliefs:

- `loop` (default): Bayesian decision. Score each candidate by its posterior-mean effect
  and act on the best one if it clears the bar. The prior starts at "trust my estimates"
  (w1 = 1), so day one looks like greedy; evidence then bends each arm's calibration.
- `loop_ts`: Thompson sampling. Draw one (w0, w1) per arm and act on the best draw. It
  explores more. In the 12-week simulation that exploration cost more than it earned (see
  the README), so it is kept as an ablation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..config import ARMS, BanditConfig
from ..journal import Observation


@dataclass(frozen=True)
class Posterior:
    arm: str
    n: int
    mean: tuple[float, float]  # (w0, w1) in model units
    cov: tuple[tuple[float, float], tuple[float, float]]

    def predict(self, x_dollars: float, scale: float) -> tuple[float, float]:
        """Posterior mean and sd of the effect, in $/day, for a hypothesis estimate x."""
        phi = np.array([1.0, x_dollars / scale])
        m = float(phi @ np.array(self.mean))
        v = float(phi @ np.array(self.cov) @ phi)
        return m * scale, math.sqrt(max(v, 0.0)) * scale

    def sample(self, rng: np.random.Generator) -> tuple[float, float]:
        cov = np.array(self.cov)
        chol = np.linalg.cholesky(cov + 1e-12 * np.eye(2))
        z = rng.standard_normal(2)
        w = np.array(self.mean) + chol @ z
        return float(w[0]), float(w[1])

    def to_dict(self) -> dict:
        return {"arm": self.arm, "n": self.n, "w0": round(self.mean[0], 6), "w1": round(self.mean[1], 6),
                "sd_w0": round(math.sqrt(self.cov[0][0]), 6), "sd_w1": round(math.sqrt(self.cov[1][1]), 6)}


def posterior(arm: str, obs: list[Observation], cfg: BanditConfig) -> Posterior:
    m0 = np.array([cfg.prior_intercept_mean, cfg.prior_slope_mean])
    p0 = np.diag([1.0 / cfg.prior_intercept_sd ** 2, 1.0 / cfg.prior_slope_sd ** 2])
    prec = p0.copy()
    b = p0 @ m0
    n = 0
    for o in obs:
        if o.arm != arm:
            continue
        phi = np.array([1.0, o.x / cfg.scale])
        var = (o.se / cfg.scale) ** 2 + cfg.model_noise_sd ** 2
        prec += np.outer(phi, phi) / var
        b += phi * (o.y / cfg.scale) / var
        n += 1
    cov = np.linalg.inv(prec)
    mean = cov @ b
    return Posterior(arm, n, (float(mean[0]), float(mean[1])),
                     ((float(cov[0, 0]), float(cov[0, 1])), (float(cov[1, 0]), float(cov[1, 1]))))


def posteriors(obs: list[Observation], cfg: BanditConfig) -> dict[str, Posterior]:
    return {a: posterior(a, obs, cfg) for a in ARMS}


@dataclass(frozen=True)
class Choice:
    arm: str | None  # None = hold
    scores: dict[str, float]  # $/day score per candidate arm (sampled, estimated or random)
    rule: str


def choose(policy: str, candidates: dict[str, float], posts: dict[str, Posterior], cfg: BanditConfig,
           rng: np.random.Generator, locks: dict[str, int] | None = None, hurdle: float = 0.0) -> Choice:
    """Pick one arm (or hold). `candidates` maps arm -> the hypothesis' own estimate in $/day.

    Every ad change ties up the campaigns it touches until it is measured, so a change has
    to clear `hurdle` dollars per day for each campaign it locks (`locks[arm]`). The rule is
    the same for the loop and for greedy; only the effect they believe differs.
    Arms are visited in a fixed order so a given rng seed always gives the same choice.
    """
    locks = locks or {}
    arms = [a for a in ARMS if a in candidates]
    if policy == "static" or not arms:
        return Choice(None, {}, "static: never changes anything" if policy == "static" else "no candidates")
    if policy == "random":
        options = arms + [None]
        pick = options[int(rng.integers(len(options)))]
        return Choice(pick, {a: 0.0 for a in arms}, "random: uniform over candidates and hold")
    if policy == "greedy":
        scores = {a: float(candidates[a]) for a in arms}
        rule = "greedy: act on the largest own estimate, no learning"
    elif policy == "loop_ts":
        scores = {}
        for a in arms:
            w0, w1 = posts[a].sample(rng)
            scores[a] = (w0 + w1 * candidates[a] / cfg.scale) * cfg.scale
        rule = "loop_ts: Thompson sample from each arm's posterior"
    elif policy == "loop":
        scores = {a: posts[a].predict(candidates[a], cfg.scale)[0] for a in arms}
        rule = "loop: posterior-mean effect per arm (learned calibration, no sampling noise)"
    else:
        raise ValueError(policy)

    def net(a: str) -> float:  # $/day per locked campaign, above the hurdle
        n = locks.get(a, 0)
        return scores[a] / n - hurdle if n else scores[a]

    best = max(arms, key=net)
    return Choice(best if net(best) > 0 else None, {a: round(s, 4) for a, s in scores.items()},
                  rule + (f"; must clear ${hurdle:.0f}/day per locked campaign" if hurdle else ""))
