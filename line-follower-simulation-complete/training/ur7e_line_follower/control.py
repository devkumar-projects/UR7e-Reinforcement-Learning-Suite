"""Commande cartésienne V2.2 du point laser sur le mur.

L'action RL ne commande plus directement les six articulations. Elle définit une
vitesse normalisée [v_y, v_z] du spot laser dans le plan du mur. La MGI
différentielle transforme ensuite cette vitesse en q_dot, puis le bridge applique
le filtre LQR et la correction de manipulabilité dans le noyau de J_wall.
"""
from __future__ import annotations

import numpy as np

from .kinematics import wall_velocity_to_joint_vel

ACTION_SPACE_DIM = 2
CONTROL_SCHEMA_VERSION = 2
MAX_WALL_SPEED_M_S = 0.12
MAX_JOINT_SPEED_RAD_S = 0.35
ACTION_EMA_ALPHA = 0.50
MGI_DAMPING = 0.015


def normalized_action_to_wall_velocity(action: np.ndarray,
                                       previous_wall_velocity: np.ndarray | None = None,
                                       alpha: float = ACTION_EMA_ALPHA,
                                       max_speed: float = MAX_WALL_SPEED_M_S) -> np.ndarray:
    """Convertit action [-1,1]^2 en vitesse mur [y_dot,z_dot] lissée.

    Le clamp par norme garantit que la vitesse diagonale ne dépasse jamais
    ``max_speed``. Le filtre EMA réduit les à-coups au démarrage de SAC.
    """
    a = np.clip(np.asarray(action, dtype=np.float64).reshape(ACTION_SPACE_DIM), -1.0, 1.0)
    raw = a * float(max_speed)
    n = float(np.linalg.norm(raw))
    if n > max_speed > 0.0:
        raw *= max_speed / n
    if previous_wall_velocity is None:
        return raw
    prev = np.asarray(previous_wall_velocity, dtype=np.float64).reshape(2)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return (1.0 - alpha) * prev + alpha * raw


def wall_action_to_joint_velocity(q: np.ndarray, action: np.ndarray,
                                  previous_wall_velocity: np.ndarray | None = None,
                                  max_speed: float = MAX_WALL_SPEED_M_S) -> tuple[np.ndarray, np.ndarray]:
    """Retourne ``(q_dot, wall_velocity)`` pour une action RL 2D.

    ``max_speed`` permet au curriculum de réduire l'amplitude des actions aux
    premiers niveaux sans changer l'espace d'action normalisé de SAC.
    """
    wall_velocity = normalized_action_to_wall_velocity(
        action, previous_wall_velocity, max_speed=max_speed)
    q_dot = wall_velocity_to_joint_vel(
        q, wall_velocity, max_jvel=MAX_JOINT_SPEED_RAD_S, damping=MGI_DAMPING)
    q_dot = np.nan_to_num(q_dot, nan=0.0, posinf=0.0, neginf=0.0)
    return q_dot, wall_velocity
