"""
Singularités, manipulabilité et filtrage de commande pour la tâche laser/mur.

La correction secondaire est projetée dans le noyau de J_wall, pas seulement de
la Jacobienne TCP. Ainsi elle perturbe beaucoup moins la position du spot laser.
"""
from __future__ import annotations

import numpy as np

from .kinematics import jacobian, wall_jacobian, damped_pseudoinverse, jacobian_condition

W_MIN = 0.015
W_REF = 0.115
LAMBDA_BASE = 1e-4
LAMBDA_MAX = 0.08
NULL_ALPHA = 0.10
NULL_ACTIVATION_W = W_REF
SIGMA_NORM = 0.8


def _safe_svd(J: np.ndarray):
    return np.linalg.svd(np.asarray(J, dtype=np.float64), full_matrices=False)


def yoshikawa(q: np.ndarray, task: str = 'tcp') -> float:
    """Indice √det(JJᵀ). task='tcp' -> J 3×6, task='wall' -> J_wall 2×6."""
    J = wall_jacobian(q) if task == 'wall' else jacobian(q)
    JJT = J @ J.T
    d = float(np.linalg.det(JJT)) if JJT.size else 0.0
    return float(np.sqrt(max(d, 0.0)))


def singular_values(q: np.ndarray, task: str = 'tcp') -> np.ndarray:
    J = wall_jacobian(q) if task == 'wall' else jacobian(q)
    return np.linalg.svd(J, compute_uv=False)


def manipulability_obs(q: np.ndarray, w_ref: float = W_REF,
                       sv_ref: float = SIGMA_NORM) -> np.ndarray:
    sv_tcp = singular_values(q, 'tcp')
    w_wall = yoshikawa(q, 'wall')
    # Observation plus pertinente pour la tâche : w_wall + sigma min/max TCP.
    return np.array([
        np.clip(w_wall / max(w_ref, 1e-9), 0.0, 1.0),
        np.clip(sv_tcp[-1] / max(sv_ref, 1e-9), 0.0, 1.0),
        np.clip(sv_tcp[0] / max(sv_ref, 1e-9), 0.0, 1.0),
    ], dtype=np.float32)


def damped_pinv(J: np.ndarray, lambda_base: float = LAMBDA_BASE,
                lambda_max: float = LAMBDA_MAX,
                sigma_thresh: float = 0.05):
    U, s, Vt = _safe_svd(J)
    if len(s) == 0:
        return np.zeros((np.asarray(J).shape[1], np.asarray(J).shape[0])), lambda_max
    sigma_min = float(s[-1])
    if sigma_min < sigma_thresh:
        ratio = np.clip(1.0 - sigma_min / max(sigma_thresh, 1e-12), 0.0, 1.0)
        lam = lambda_base + ratio * (lambda_max - lambda_base)
    else:
        lam = lambda_base
    s_inv = s / (s * s + lam * lam)
    return Vt.T @ np.diag(s_inv) @ U.T, float(lam)


def manipulability_gradient(q: np.ndarray, task: str = 'wall', eps: float = 1e-4) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(6)
    w0 = yoshikawa(q, task=task)
    grad = np.zeros(6, dtype=np.float64)
    for i in range(6):
        qp = q.copy(); qp[i] += eps
        qm = q.copy(); qm[i] -= eps
        grad[i] = (yoshikawa(qp, task=task) - yoshikawa(qm, task=task)) / (2.0 * eps)
    if not np.all(np.isfinite(grad)):
        grad[:] = 0.0
    # Normalisation pour éviter des corrections énormes près d'une zone mal conditionnée.
    n = float(np.linalg.norm(grad))
    if n > 1.0:
        grad /= n
    return grad


def null_space_manip_correction(q: np.ndarray, alpha: float = NULL_ALPHA,
                                lambda_base: float = LAMBDA_BASE,
                                max_norm: float = 0.15) -> np.ndarray:
    """Correction secondaire projetée dans Ker(J_wall)."""
    q = np.asarray(q, dtype=np.float64).reshape(6)
    Jw = wall_jacobian(q)
    if np.linalg.norm(Jw) < 1e-9:
        return np.zeros(6, dtype=np.float64)

    # No secondary motion is useful in a well-conditioned configuration.
    # Applying it permanently made the null-space term larger than the primary
    # task command.  A later per-joint saturation then destroyed J_wall*N = 0
    # and introduced a systematic downward laser drift.
    w_wall = yoshikawa(q, task='wall')
    activation = float(np.clip(
        (NULL_ACTIVATION_W - w_wall)
        / max(NULL_ACTIVATION_W - W_MIN, 1e-9), 0.0, 1.0))
    if activation <= 0.0:
        return np.zeros(6, dtype=np.float64)

    Jp, _ = damped_pinv(Jw, lambda_base=lambda_base)
    N = np.eye(6) - Jp @ Jw
    grad_w = manipulability_gradient(q, task='wall')
    qdot = (alpha * activation) * (N @ grad_w)
    n = float(np.linalg.norm(qdot))
    if n > max_norm:
        qdot *= max_norm / n
    return qdot


def lqr_gains(q: np.ndarray, w_yoshikawa: float | None = None,
              q_lqr: float = 1.0, r_base: float = 1.0,
              r_min: float = 0.05) -> np.ndarray:
    """
    Gains d'atténuation par axe. Plus l'axe participe à la direction singulière
    de J_wall, plus K est grand, donc plus la commande est réduite.
    """
    Jw = wall_jacobian(q)
    if np.linalg.norm(Jw) < 1e-12:
        return np.ones(6, dtype=np.float64) * 0.5
    _, s, Vt = _safe_svd(Jw)
    v_singular = Vt[-1]
    participation = v_singular * v_singular
    participation /= participation.max() + 1e-12
    sigma_min = float(s[-1]) if len(s) else 0.0
    ratio = np.clip(1.0 - sigma_min / 0.05, 0.0, 1.0)
    # Axes dangereux : R diminue => K augmente.
    r_per_axis = r_base - ratio * (r_base - r_min) * participation
    r_per_axis = np.clip(r_per_axis, r_min, r_base)
    K = np.sqrt(1.0 + q_lqr / r_per_axis) - 1.0
    return K.astype(np.float64)


def lqr_velocity_correction(q: np.ndarray, q_dot_desired: np.ndarray,
                            w_yoshikawa: float | None = None,
                            q_dot_max: float = 1.0):
    """Filtre type LQR-diagonal : réduit les composantes dangereuses près singularité."""
    q_dot_desired = np.asarray(q_dot_desired, dtype=np.float64).reshape(6)
    K = lqr_gains(q, w_yoshikawa)
    corrected = q_dot_desired / (1.0 + K)
    return np.clip(corrected, -q_dot_max, q_dot_max)


def command_filter_diagnostics(q: np.ndarray, q_dot_desired: np.ndarray,
                               q_dot_max: float = 1.0) -> dict:
    w_tcp = yoshikawa(q, 'tcp')
    w_wall = yoshikawa(q, 'wall')
    K = lqr_gains(q, w_wall)
    lqr_cmd = lqr_velocity_correction(q, q_dot_desired, w_wall, q_dot_max)
    null_cmd = null_space_manip_correction(q)
    out_cmd = lqr_cmd + null_cmd
    # Uniform scaling preserves the task direction and the null-space
    # property.  Independent per-joint clipping can turn a true null-space
    # vector into a large Cartesian drift.
    peak = float(np.max(np.abs(out_cmd))) if out_cmd.size else 0.0
    if q_dot_max > 0.0 and peak > q_dot_max:
        out_cmd *= float(q_dot_max) / peak
    Jw = wall_jacobian(q)
    sv_wall = np.linalg.svd(Jw, compute_uv=False) if np.linalg.norm(Jw) > 0 else np.zeros(2)
    return {
        'w_tcp': float(w_tcp),
        'w_wall': float(w_wall),
        'sigma_wall_min': float(sv_wall[-1]) if len(sv_wall) else 0.0,
        'sigma_wall_max': float(sv_wall[0]) if len(sv_wall) else 0.0,
        'cond_wall': jacobian_condition(Jw),
        'lqr_gain_mean': float(np.mean(K)),
        'lqr_gain_max': float(np.max(K)),
        'raw_cmd_norm': float(np.linalg.norm(q_dot_desired)),
        'lqr_cmd_norm': float(np.linalg.norm(lqr_cmd)),
        'null_cmd_norm': float(np.linalg.norm(null_cmd)),
        'out_cmd_norm': float(np.linalg.norm(out_cmd)),
        'wall_speed_raw': float(np.linalg.norm(Jw @ q_dot_desired)),
        'wall_speed_out': float(np.linalg.norm(Jw @ out_cmd)),
        'lqr_cmd': lqr_cmd,
        'null_cmd': null_cmd,
        'out_cmd': out_cmd,
        'K': K,
    }


def singularity_penalty(q: np.ndarray, w_ref: float = W_MIN,
                        p_max: float = 0.3) -> float:
    w = yoshikawa(q, task='wall')
    if w >= w_ref:
        return 0.0
    return -float(p_max * (1.0 - np.clip(w / max(w_ref, 1e-12), 0.0, 1.0)))


def check_known_singularities(q: np.ndarray,
                              thresh_elbow: float = 0.15,
                              thresh_wrist: float = 0.15) -> dict:
    q = np.asarray(q, dtype=np.float64).reshape(6)
    q3 = float(q[2])
    q5 = float(q[4])
    J = jacobian(q)
    waist_measure = float(np.linalg.norm(J[:, 0]))
    return {
        'elbow': {
            'is_near': abs(np.sin(q3)) < thresh_elbow,
            'distance': float(abs(np.sin(q3))),
        },
        'wrist': {
            'is_near': abs(np.sin(q5)) < thresh_wrist,
            'distance': float(abs(np.sin(q5))),
        },
        'shoulder': {
            'is_near': waist_measure < thresh_elbow,
            'distance': waist_measure,
        },
    }
