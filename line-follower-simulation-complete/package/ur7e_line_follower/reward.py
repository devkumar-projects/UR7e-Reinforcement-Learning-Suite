"""Reward helpers for UR7e line-following.

The helpers are ROS/Gazebo independent and therefore unit-testable.  Three
tracking profiles are intentionally available for controlled A/B experiments:

``log``
    Bounded logarithmic shaping with high resolution close to the line.
``huber``
    Strong, unbounded far-field pull.  Useful as a diagnostic, but it changes
    the reward magnitude substantially.
``normalized_huber``
    Huber shape divided by its value at the workspace limit.  This keeps the
    strong far-field direction of Huber while preserving a stable ``[-1, 0]``
    range.  It is the default profile.
"""
from __future__ import annotations

import math
import numpy as np

REWARD_PROFILES = ("log", "huber", "normalized_huber")
DEFAULT_REWARD_PROFILE = "normalized_huber"


def _huber_loss(normalized_distance: float) -> float:
    """Scalar Huber loss with unit transition point."""
    x = max(float(normalized_distance), 0.0)
    if x <= 1.0:
        return 0.5 * x * x
    return x - 0.5


def tracking_reward(
    distance_m: float,
    *,
    near_scale_m: float = 0.04,
    max_distance_m: float = 0.50,
    profile: str = DEFAULT_REWARD_PROFILE,
    huber_delta_m: float = 0.10,
) -> float:
    """Return an informative tracking reward over the complete workspace.

    Args:
        distance_m: Ordered lateral distance from the laser spot to the line.
        near_scale_m: Curvature scale for the logarithmic profile.
        max_distance_m: Distance mapped to the worst bounded reward.
        profile: One of :data:`REWARD_PROFILES`.
        huber_delta_m: Huber transition distance.

    Returns:
        ``log`` and ``normalized_huber`` return values in ``[-1, 0]``.
        ``huber`` is non-positive and intentionally unbounded until the caller's
        workspace clipping at ``max_distance_m``.
    """
    if profile not in REWARD_PROFILES:
        raise ValueError(
            f"reward profile invalide: {profile!r}; attendu {REWARD_PROFILES}")

    max_d = max(float(max_distance_m), 1e-9)
    d = float(np.clip(distance_m, 0.0, max_d))

    if profile == "log":
        scale = max(float(near_scale_m), 1e-9)
        denom = math.log1p(max_d / scale)
        return -float(math.log1p(d / scale) / max(denom, 1e-12))

    delta = max(float(huber_delta_m), 1e-9)
    value = _huber_loss(d / delta)
    if profile == "huber":
        return -float(value)

    maximum = _huber_loss(max_d / delta)
    return -float(value / max(maximum, 1e-12))


def gated_progress_reward(
    delta_s_m: float,
    lateral_error_m: float,
    *,
    nominal_step_m: float,
    gain: float = 2.0,
    tracking_gate_m: float = 0.10,
) -> float:
    """Reward ordered forward progress only near the path.

    Backward motion remains penalised everywhere.  Positive progress is removed
    when the spot is too far from the line, preventing shortcut/wandering reward.
    """
    raw = float(gain) * float(np.clip(
        float(delta_s_m) / max(float(nominal_step_m), 1e-9), -1.0, 1.0
    ))
    if float(lateral_error_m) > float(tracking_gate_m):
        return min(raw, 0.0)
    return raw


def distance_outside_rectangle(
    point_yz: np.ndarray | None,
    *,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> float:
    """Euclidean distance from a point to the rectangular wall boundary.

    Returns zero for points on or inside the wall.  ``None`` is treated as an
    unknown/far point and returns ``inf``.
    """
    if point_yz is None:
        return float("inf")
    p = np.asarray(point_yz, dtype=np.float64).reshape(2)
    dy = max(float(y_min) - p[0], 0.0, p[0] - float(y_max))
    dz = max(float(z_min) - p[1], 0.0, p[1] - float(z_max))
    return float(math.hypot(dy, dz))


def offwall_penalty(
    outside_distance_m: float,
    *,
    base: float = -1.0,
    extra: float = -2.0,
    scale_m: float = 0.10,
) -> float:
    """Graduated penalty for leaving the wall instead of a flat constant."""
    if not np.isfinite(outside_distance_m):
        return float(base + extra)
    ratio = float(np.clip(outside_distance_m / max(scale_m, 1e-9), 0.0, 1.0))
    return float(base + extra * ratio)


def recent_rmse(
    values: list[float] | np.ndarray,
    window: int = 30,
    default: float = 0.50,
) -> float:
    """RMSE over the most recent samples, used for recoverable success logic."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    arr = arr[-max(1, int(window)):]
    return float(np.sqrt(np.mean(np.square(arr))))
