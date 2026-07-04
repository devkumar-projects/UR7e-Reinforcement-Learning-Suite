"""
Cinématique UR7e nominale + géométrie laser/mur.

Le modèle utilise les constantes nominales UR7e. Pour un vrai robot, remplacer les
paramètres nominaux par la calibration usine extraite du bras.
"""
from __future__ import annotations

import numpy as np

# Limites du mur cohérentes avec target_line.py. Elles sont volontairement
# répétées ici pour éviter un import circulaire.
WALL_X_DEFAULT = 1.0
WALL_Y_MIN, WALL_Y_MAX = -0.65, 0.65
WALL_Z_MIN, WALL_Z_MAX = 0.20, 1.30
MIN_RAY_X = 1e-3
MAX_RAY_LENGTH = 3.0

# Transformations nominales des joints UR7e dans la convention utilisée par le
# package UR ROS 2. Chaque tuple = (x, y, z, roll, pitch, yaw) puis rotation q_i
# autour de l'axe z local.
UR_NOMINAL_JOINT_TRANSFORMS = [
    (0.0,     0.0,     0.1625, 0.0,       0.0,     0.0),
    (0.0,     0.0,     0.0,    np.pi / 2, 0.0,     0.0),
    (-0.425,  0.0,     0.0,    0.0,       0.0,     0.0),
    (-0.3922, 0.0,     0.1333, 0.0,       0.0,     0.0),
    (0.0,    -0.0997,  0.0,    np.pi / 2, 0.0,     0.0),
    (0.0,     0.0996,  0.0,    np.pi / 2, np.pi,   np.pi),
]


def _as_q(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.shape[0] != 6:
        raise ValueError(f"Expected 6 joint values, got shape {q.shape}")
    return q


def _rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def _raw_fk(q: np.ndarray) -> np.ndarray:
    q = _as_q(q)
    T = np.eye(4, dtype=np.float64)
    for i, (x, y, z, ro, p, yw) in enumerate(UR_NOMINAL_JOINT_TRANSFORMS):
        To = np.eye(4, dtype=np.float64)
        To[:3, :3] = _rpy(ro, p, yw)
        To[:3, 3] = [x, y, z]
        cq, sq = np.cos(q[i]), np.sin(q[i])
        Tj = np.eye(4, dtype=np.float64)
        Tj[:3, :3] = [[cq, -sq, 0.0], [sq, cq, 0.0], [0.0, 0.0, 1.0]]
        T = T @ To @ Tj
    return T


def fk_transform(q: np.ndarray) -> np.ndarray:
    """Transformation homogène nominale base -> tool0 dans le repère monde du package."""
    T = _raw_fk(q).copy()
    # Convention historique du package : axes x/y inversés par rapport au raw UR.
    T[0, 3] *= -1.0
    T[1, 3] *= -1.0
    T[0, :3] *= -1.0
    T[1, :3] *= -1.0
    return T


def fk_ur(q: np.ndarray) -> np.ndarray:
    """Position TCP [x, y, z] dans le repère monde du package."""
    return fk_transform(q)[:3, 3].copy()


def fk_ur_toolz(q: np.ndarray) -> np.ndarray:
    """Direction normalisée de l'axe Z outil, utilisée comme direction du laser."""
    v = fk_transform(q)[:3, 2].copy()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def laser_wall_intersection_unbounded(q: np.ndarray, wall_x: float = WALL_X_DEFAULT):
    """Intersection du rayon laser avec le plan x=wall_x, sans test de taille du mur."""
    pos = fk_ur(q)
    direction = fk_ur_toolz(q)
    if direction[0] <= MIN_RAY_X:
        return None
    t = (wall_x - pos[0]) / direction[0]
    if t < 0.0 or t > MAX_RAY_LENGTH:
        return None
    yz = np.array([pos[1] + t * direction[1], pos[2] + t * direction[2]], dtype=np.float64)
    if not np.all(np.isfinite(yz)):
        return None
    return yz


def laser_wall_dot(q: np.ndarray, wall_x: float = WALL_X_DEFAULT,
                   check_bounds: bool = True, margin: float = 0.0):
    """
    Position [y,z] du point laser sur le mur, ou None si le faisceau ne touche
    pas la face utile du mur.
    """
    yz = laser_wall_intersection_unbounded(q, wall_x)
    if yz is None or not check_bounds:
        return yz
    y, z = float(yz[0]), float(yz[1])
    if not (WALL_Y_MIN + margin <= y <= WALL_Y_MAX - margin):
        return None
    if not (WALL_Z_MIN + margin <= z <= WALL_Z_MAX - margin):
        return None
    return yz


def jacobian(q: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Jacobienne numérique 3×6 de position TCP."""
    q = _as_q(q)
    J = np.zeros((3, 6), dtype=np.float64)
    for i in range(6):
        qp = q.copy(); qp[i] += eps
        qm = q.copy(); qm[i] -= eps
        J[:, i] = (fk_ur(qp) - fk_ur(qm)) / (2.0 * eps)
    return J


def wall_jacobian(q: np.ndarray, wall_x: float = WALL_X_DEFAULT,
                  eps: float = 1e-5, bounded: bool = False) -> np.ndarray:
    """Jacobienne numérique 2×6 : q_dot -> vitesse [y_dot,z_dot] du laser sur le mur."""
    q = _as_q(q)
    J = np.zeros((2, 6), dtype=np.float64)
    f = laser_wall_dot if bounded else laser_wall_intersection_unbounded
    dot0 = f(q, wall_x)
    if dot0 is None:
        return J
    for i in range(6):
        qp = q.copy(); qp[i] += eps
        qm = q.copy(); qm[i] -= eps
        dp = f(qp, wall_x)
        dm = f(qm, wall_x)
        if dp is not None and dm is not None:
            J[:, i] = (dp - dm) / (2.0 * eps)
        elif dp is not None:
            J[:, i] = (dp - dot0) / eps
        elif dm is not None:
            J[:, i] = (dot0 - dm) / eps
    J[~np.isfinite(J)] = 0.0
    return J


def damped_pseudoinverse(J: np.ndarray, damping: float = 1e-4) -> np.ndarray:
    """Pseudo-inverse amortie stable pour matrices rectangulaires."""
    J = np.asarray(J, dtype=np.float64)
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    s_inv = s / (s * s + damping * damping)
    return Vt.T @ np.diag(s_inv) @ U.T


def cartesian_to_joint_vel(q: np.ndarray, tcp_vel: np.ndarray,
                           max_jvel: float = 1.5, damping: float = 1e-4) -> np.ndarray:
    """MGI différentielle pour une vitesse TCP translationnelle [vx,vy,vz]."""
    tcp_vel = np.asarray(tcp_vel, dtype=np.float64).reshape(3)
    dq = damped_pseudoinverse(jacobian(q), damping) @ tcp_vel
    norm = float(np.linalg.norm(dq))
    if norm > max_jvel:
        dq *= max_jvel / norm
    return dq


def wall_velocity_to_joint_vel(q: np.ndarray, wall_vel: np.ndarray,
                               max_jvel: float = 1.5, damping: float = 1e-4) -> np.ndarray:
    """MGI différentielle de tâche : vitesse désirée du point laser sur le mur -> q_dot."""
    wall_vel = np.asarray(wall_vel, dtype=np.float64).reshape(2)
    dq = damped_pseudoinverse(wall_jacobian(q), damping) @ wall_vel
    norm = float(np.linalg.norm(dq))
    if norm > max_jvel:
        dq *= max_jvel / norm
    return dq


def jacobian_condition(J: np.ndarray, eps: float = 1e-9) -> float:
    s = np.linalg.svd(np.asarray(J, dtype=np.float64), compute_uv=False)
    if len(s) == 0 or s[-1] < eps:
        return float('inf')
    return float(s[0] / s[-1])
