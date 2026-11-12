from __future__ import annotations

import math

import numpy as np
import pytest

from channel_loop.agent.bandit import choose, posterior, posteriors
from channel_loop.config import ARMS, BanditConfig
from channel_loop.journal import Observation
from channel_loop.stats import bootstrap_ci, compare, sign_flip_p


def obs(arm, x, y, se, i=0):
    return Observation(arm, x, y, se, f"d{i}", i)


def test_prior_only_posterior_is_the_prior():
    cfg = BanditConfig()
    p = posterior("bid_up", [], cfg)
    assert p.n == 0
    assert p.mean == pytest.approx((cfg.prior_intercept_mean, cfg.prior_slope_mean))
    assert p.cov[0][0] == pytest.approx(cfg.prior_intercept_sd ** 2)
    assert p.cov[1][1] == pytest.approx(cfg.prior_slope_sd ** 2)


def test_posterior_matches_closed_form_conjugate_update():
    cfg = BanditConfig()
    o = [obs("budget_shift", 40.0, 10.0, 80.0, 0), obs("budget_shift", 25.0, -5.0, 60.0, 1)]
    p = posterior("budget_shift", o, cfg)
    s0 = np.diag([cfg.prior_intercept_sd ** 2, cfg.prior_slope_sd ** 2])
    m0 = np.array([cfg.prior_intercept_mean, cfg.prior_slope_mean])
    X = np.array([[1, 0.40], [1, 0.25]])
    y = np.array([0.10, -0.05])
    R = np.diag([(0.8 ** 2 + cfg.model_noise_sd ** 2), (0.6 ** 2 + cfg.model_noise_sd ** 2)])
    cov = np.linalg.inv(np.linalg.inv(s0) + X.T @ np.linalg.inv(R) @ X)
    mean = cov @ (np.linalg.inv(s0) @ m0 + X.T @ np.linalg.inv(R) @ y)
    assert p.mean == pytest.approx(tuple(mean))
    assert np.array(p.cov) == pytest.approx(cov)


def test_evidence_moves_belief_toward_truth_and_shrinks_uncertainty():
    """If an arm's real effect is about 3x the estimate, the slope learns it."""
    cfg = BanditConfig()
    rng = np.random.default_rng(0)
    data = []
    for i in range(40):
        x = float(rng.uniform(20, 80))
        data.append(obs("creative_refresh", x, 3 * x + float(rng.normal(0, 60)), 60.0, i))
    p0 = posterior("creative_refresh", [], cfg)
    p = posterior("creative_refresh", data, cfg)
    assert p.mean[1] > 1.5  # well past the prior's 1.0
    mean50, sd50 = p.predict(50.0, cfg.scale)
    assert abs(mean50 - 150.0) < 25.0  # predicted effect for a $50 estimate is close to the true $150
    assert p0.predict(50.0, cfg.scale)[0] == pytest.approx(50.0)
    assert sd50 < p0.predict(50.0, cfg.scale)[1] / 2  # and it is much more certain
    other = posterior("bid_up", data, cfg)  # evidence only counts for its own arm
    assert other.n == 0


def test_choice_is_deterministic_for_a_seed():
    cfg = BanditConfig()
    posts = posteriors([], cfg)
    cands = {"budget_shift": 40.0, "bid_up": 25.0, "creative_refresh": 30.0}
    a = choose("loop_ts", cands, posts, cfg, np.random.default_rng([1, 2, 3]))
    b = choose("loop_ts", cands, posts, cfg, np.random.default_rng([1, 2, 3]))
    assert a == b


def test_static_always_holds_and_random_can_hold():
    cfg = BanditConfig()
    posts = posteriors([], cfg)
    cands = {"budget_shift": 40.0}
    assert choose("static", cands, posts, cfg, np.random.default_rng(0)).arm is None
    picks = {choose("random", cands, posts, cfg, np.random.default_rng(i)).arm for i in range(50)}
    assert picks == {None, "budget_shift"}


def test_hurdle_is_per_locked_campaign():
    cfg = BanditConfig()
    posts = posteriors([], cfg)
    # a $40 shift locks two campaigns: $20 each, above a $15 hurdle; a $25 bid change locks one
    c = choose("greedy", {"budget_shift": 40.0, "bid_up": 25.0}, posts, cfg, np.random.default_rng(0),
               locks={"budget_shift": 2, "bid_up": 1}, hurdle=15.0)
    assert c.arm == "bid_up"  # $10 over the bar beats $5 over
    c = choose("greedy", {"budget_shift": 28.0}, posts, cfg, np.random.default_rng(0),
               locks={"budget_shift": 2}, hurdle=15.0)
    assert c.arm is None  # $14 per campaign is under the bar


def test_learned_negative_arm_is_dropped_by_the_mean_policy():
    cfg = BanditConfig()
    data = [obs("bid_up", 30.0, -60.0, 30.0, i) for i in range(12)]
    posts = posteriors(data, cfg)
    c = choose("loop", {"bid_up": 30.0, "bid_down": 10.0}, posts, cfg, np.random.default_rng(0))
    assert c.arm == "bid_down"
    assert c.scores["bid_up"] < 0
    greedy = choose("greedy", {"bid_up": 30.0, "bid_down": 10.0}, posts, cfg, np.random.default_rng(0))
    assert greedy.arm == "bid_up"  # greedy cannot learn


def test_all_arms_have_posteriors():
    assert set(posteriors([], BanditConfig())) == set(ARMS)


def test_sign_flip_exact_and_bootstrap():
    assert sign_flip_p(np.array([1.0, 2.0, 3.0, 4.0, 5.0])) == pytest.approx(1 / 32)
    assert sign_flip_p(np.array([1.0, -1.0])) == pytest.approx(0.75)
    lo, hi = bootstrap_ci(np.array([1.0, 2.0, 3.0, 4.0]))
    assert lo <= 2.5 <= hi
    c = compare(np.array([5.0, 6.0, -1.0, 4.0]))
    assert c.wins == 3 and c.n == 4
