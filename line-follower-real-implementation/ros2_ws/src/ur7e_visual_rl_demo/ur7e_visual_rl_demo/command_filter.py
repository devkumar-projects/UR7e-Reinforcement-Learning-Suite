"""MGI post-filter: singular-direction LQR attenuation + null-space objective.

This keeps the requested wall velocity as the primary task.  The secondary
manipulability term is projected in the null space of the calibrated laser-wall
Jacobian and uniformly rescaled when a joint-speed limit is reached.
"""
from __future__ import annotations

import numpy as np


def damped_pinv(J: np.ndarray, damping: float = 0.015) -> np.ndarray:
    J = np.asarray(J, dtype=np.float64)
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    inv = s / (s * s + float(damping) ** 2)
    return Vt.T @ np.diag(inv) @ U.T


def manipulability(J: np.ndarray) -> float:
    J = np.asarray(J, dtype=np.float64)
    d = float(np.linalg.det(J @ J.T)) if J.size else 0.0
    return float(np.sqrt(max(d, 0.0)))


def lqr_attenuate(J: np.ndarray, qdot: np.ndarray, sigma_threshold: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Diagonal LQR-style damping based on participation in weak singular axes."""
    J = np.asarray(J, dtype=np.float64)
    qdot = np.asarray(qdot, dtype=np.float64).reshape(6)
    _, s, Vt = np.linalg.svd(J, full_matrices=False)
    if len(s) == 0:
        return qdot * 0.5, np.ones(6)
    sigma_min = float(s[-1])
    weakness = float(np.clip(1.0 - sigma_min / max(sigma_threshold, 1e-9), 0.0, 1.0))
    weak_axis = Vt[-1]
    participation = weak_axis * weak_axis
    participation /= float(np.max(participation) + 1e-12)
    # Equivalent diagonal state-feedback attenuation.  At good conditioning the
    # gain is near zero; near a singularity the participating joints are damped.
    K = weakness * (0.15 + 1.85 * participation)
    return qdot / (1.0 + K), K


def nullspace_manipulability_correction(
    q: np.ndarray,
    jacobian_fn,
    damping: float = 0.015,
    gain: float = 0.02,
    activation_sigma: float = 0.04,
    eps: float = 2e-4,
) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(6)
    J = np.asarray(jacobian_fn(q), dtype=np.float64)
    s = np.linalg.svd(J, compute_uv=False)
    if len(s) < 2 or float(s[-1]) >= activation_sigma:
        return np.zeros(6, dtype=np.float64)
    grad = np.zeros(6, dtype=np.float64)
    for i in range(6):
        qp, qm = q.copy(), q.copy()
        qp[i] += eps
        qm[i] -= eps
        grad[i] = (manipulability(jacobian_fn(qp)) - manipulability(jacobian_fn(qm))) / (2.0 * eps)
    n = float(np.linalg.norm(grad))
    if n > 1e-12:
        grad /= n
    N = np.eye(6) - damped_pinv(J, damping) @ J
    activation = float(np.clip((activation_sigma - s[-1]) / activation_sigma, 0.0, 1.0))
    return float(gain) * activation * (N @ grad)


def filter_joint_command(
    q: np.ndarray,
    qdot_raw: np.ndarray,
    jacobian_fn,
    max_joint_speed: float,
    damping: float = 0.015,
    enable_nullspace: bool = True,
) -> tuple[np.ndarray, dict]:
    q = np.asarray(q, dtype=np.float64).reshape(6)
    raw = np.asarray(qdot_raw, dtype=np.float64).reshape(6)
    J = np.asarray(jacobian_fn(q), dtype=np.float64)
    lqr, gains = lqr_attenuate(J, raw)
    null = (
        nullspace_manipulability_correction(q, jacobian_fn, damping=damping)
        if enable_nullspace else np.zeros(6, dtype=np.float64)
    )
    out = lqr + null
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > float(max_joint_speed) > 0.0:
        out *= float(max_joint_speed) / peak
    s = np.linalg.svd(J, compute_uv=False)
    cond = float('inf') if len(s) < 2 or s[-1] < 1e-10 else float(s[0] / s[-1])
    return np.nan_to_num(out), {
        'condition': cond,
        'sigma_min': float(s[-1]) if len(s) else 0.0,
        'sigma_max': float(s[0]) if len(s) else 0.0,
        'manipulability': manipulability(J),
        'lqr_gain_mean': float(np.mean(gains)),
        'lqr_gain_max': float(np.max(gains)),
        'raw_norm': float(np.linalg.norm(raw)),
        'lqr_norm': float(np.linalg.norm(lqr)),
        'null_norm': float(np.linalg.norm(null)),
        'out_norm': float(np.linalg.norm(out)),
    }
