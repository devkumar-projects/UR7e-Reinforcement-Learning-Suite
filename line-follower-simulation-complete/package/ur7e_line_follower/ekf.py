"""
Filtre de Kalman pour le point laser sur le mur.

État : x = [y, z, vy, vz].
La prédiction utilise la vitesse cinématique du spot J_wall(q) @ q_dot comme
entrée externe. Les mesures FK et caméra absolue sont séparées. On ne transforme
plus un offset ligne-laser KLT en mesure absolue du laser, car ce serait un
mélange incohérent entre position du laser et position de la ligne cible.
"""
from __future__ import annotations

import numpy as np

from .kinematics import wall_jacobian


class LaserDotEKF:
    """KF linéaire avec entrée cinématique non linéaire externe."""

    def __init__(self, dt: float = 0.004,
                 q_pos_std: float = 0.002,
                 q_vel_std: float = 0.10,
                 r_fk_std: float = 0.006,
                 r_cam_std: float = 0.018,
                 wall_x: float = 1.0):
        self._dt = float(dt)
        self._wall_x = float(wall_x)
        self._x = np.zeros(4, dtype=np.float64)
        self._P = np.eye(4, dtype=np.float64) * 0.05
        self._Q = np.diag([q_pos_std**2, q_pos_std**2, q_vel_std**2, q_vel_std**2])
        self._R_fk = np.eye(2, dtype=np.float64) * r_fk_std**2
        self._R_cam = np.eye(2, dtype=np.float64) * r_cam_std**2
        self._H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        self._innovation = np.zeros(2, dtype=np.float64)
        self._innovation_cov = np.eye(2, dtype=np.float64)
        self._last_update_type = 'none'
        self._initialized = False
        self._last_nees = np.nan
        self._last_nis = np.nan

    def reset(self, y: float = 0.0, z: float = 0.0):
        self._x[:] = [float(y), float(z), 0.0, 0.0]
        self._P[:] = np.diag([1e-4, 1e-4, 1e-2, 1e-2])
        self._innovation[:] = 0.0
        self._innovation_cov[:] = np.eye(2)
        self._last_update_type = 'reset'
        self._last_nees = np.nan
        self._last_nis = np.nan
        self._initialized = True

    def predict(self, q: np.ndarray | None = None, q_dot: np.ndarray | None = None,
                dt: float | None = None):
        if not self._initialized:
            return
        dt = self._dt if dt is None else float(dt)
        F = np.array([[1.0, 0.0, dt, 0.0],
                      [0.0, 1.0, 0.0, dt],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
        x_pred = F @ self._x
        if q is not None and q_dot is not None:
            Jw = wall_jacobian(np.asarray(q), self._wall_x)
            v_wall = Jw @ np.asarray(q_dot, dtype=np.float64).reshape(6)
            if np.all(np.isfinite(v_wall)):
                x_pred[2:] = v_wall
        self._x = x_pred
        self._P = F @ self._P @ F.T + self._Q
        self._P = 0.5 * (self._P + self._P.T)

    def _update(self, z: np.ndarray, R: np.ndarray, source: str) -> bool:
        z = np.asarray(z, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(z)):
            return False
        if not self._initialized:
            self.reset(float(z[0]), float(z[1]))
            self._last_update_type = source
            return True
        y = z - self._H @ self._x
        S = self._H @ self._P @ self._H.T + R
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            Sinv = np.linalg.pinv(S)
        K = self._P @ self._H.T @ Sinv
        self._x = self._x + K @ y
        I_KH = np.eye(4) - K @ self._H
        # Joseph form : covariance positive semi-définie plus robuste.
        self._P = I_KH @ self._P @ I_KH.T + K @ R @ K.T
        self._P = 0.5 * (self._P + self._P.T)
        self._innovation = y
        self._innovation_cov = S
        self._last_update_type = source
        self._last_nis = float(y.T @ Sinv @ y)
        return True

    def update_fk(self, y: float, z: float) -> bool:
        return self._update(np.array([y, z]), self._R_fk, 'fk')

    def update_camera_absolute(self, y: float, z: float) -> bool:
        """Mise à jour caméra seulement si on dispose d'une mesure métrique absolue [y,z]."""
        return self._update(np.array([y, z]), self._R_cam, 'camera_absolute')

    def update_camera(self, *args, **kwargs) -> bool:
        """Compatibilité : les offsets KLT ne sont plus acceptés comme mesure absolue."""
        return False

    def nees_against(self, y_true: float, z_true: float) -> float:
        if not self._initialized:
            return float('nan')
        e = np.asarray([y_true, z_true], dtype=np.float64) - self._x[:2]
        P2 = self._P[:2, :2]
        try:
            val = float(e.T @ np.linalg.inv(P2) @ e)
        except np.linalg.LinAlgError:
            val = float(e.T @ np.linalg.pinv(P2) @ e)
        self._last_nees = val
        return val

    @property
    def position(self) -> np.ndarray:
        return self._x[:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self._x[2:].copy()

    @property
    def covariance(self) -> np.ndarray:
        return self._P.copy()

    @property
    def uncertainty(self) -> np.ndarray:
        return np.sqrt(np.maximum(np.diag(self._P)[:2], 0.0))

    @property
    def innovation(self) -> np.ndarray:
        return self._innovation.copy()

    @property
    def innovation_cov(self) -> np.ndarray:
        return self._innovation_cov.copy()

    @property
    def nis(self) -> float:
        return self._last_nis

    @property
    def nees(self) -> float:
        return self._last_nees

    @property
    def last_update_type(self) -> str:
        return self._last_update_type

    @property
    def initialized(self) -> bool:
        return self._initialized
