import numpy as np

from ur7e_line_follower.reward import (
    tracking_reward, gated_progress_reward,
    distance_outside_rectangle, offwall_penalty, recent_rmse,
)


def test_tracking_reward_is_informative_beyond_8cm():
    values = [tracking_reward(d) for d in (0.08, 0.20, 0.30, 0.48)]
    assert values[0] > values[1] > values[2] > values[3]
    assert len(set(round(v, 6) for v in values)) == 4


def test_tracking_reward_zero_on_line_and_minus_one_at_workspace_limit():
    assert tracking_reward(0.0) == 0.0
    assert tracking_reward(0.50) == -1.0


def test_positive_progress_is_gated_when_far_from_line():
    near = gated_progress_reward(0.01, 0.03, nominal_step_m=0.01)
    far = gated_progress_reward(0.01, 0.20, nominal_step_m=0.01)
    backward_far = gated_progress_reward(-0.01, 0.20, nominal_step_m=0.01)
    assert near > 0.0
    assert far == 0.0
    assert backward_far < 0.0


def test_offwall_penalty_is_graduated():
    inside = distance_outside_rectangle(
        np.array([0.0, 0.5]), y_min=-0.65, y_max=0.65, z_min=0.2, z_max=1.3)
    close = distance_outside_rectangle(
        np.array([0.70, 0.5]), y_min=-0.65, y_max=0.65, z_min=0.2, z_max=1.3)
    far = distance_outside_rectangle(
        np.array([0.90, 0.5]), y_min=-0.65, y_max=0.65, z_min=0.2, z_max=1.3)
    assert inside == 0.0
    assert offwall_penalty(inside) > offwall_penalty(close) > offwall_penalty(far)


def test_recent_rmse_recovers_after_bad_episode_start():
    values = [0.50] * 50 + [0.02] * 30
    assert recent_rmse(values, window=30) < 0.03
    assert recent_rmse(values, window=80) > 0.20


def test_all_tracking_profiles_are_monotonic():
    from ur7e_line_follower.reward import REWARD_PROFILES
    for profile in REWARD_PROFILES:
        values = [tracking_reward(d, profile=profile) for d in (0.0, 0.04, 0.15, 0.45)]
        assert values[0] > values[1] > values[2] > values[3], (profile, values)


def test_normalized_huber_is_bounded_and_huber_has_stronger_far_field_pull():
    normalized = tracking_reward(0.50, profile='normalized_huber')
    raw = tracking_reward(0.50, profile='huber')
    log = tracking_reward(0.50, profile='log')
    assert normalized == -1.0
    assert log == -1.0
    assert raw < normalized


def test_invalid_reward_profile_is_rejected():
    import pytest
    with pytest.raises(ValueError):
        tracking_reward(0.1, profile='not-a-profile')
