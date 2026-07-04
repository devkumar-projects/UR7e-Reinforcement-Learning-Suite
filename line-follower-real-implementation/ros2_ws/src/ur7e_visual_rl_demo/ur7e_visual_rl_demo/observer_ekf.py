"""Fusion of calibrated MGD and absolute camera laser measurements.

State: ``[y, z, vy, vz]`` in the physical drawing-plane coordinates.  The
prediction velocity comes from the calibrated wall Jacobian and measured joint
velocity.  MGD and camera measurements use separate noise models.
"""
from __future__ import annotations

import numpy as np


class LaserWallEKF:
    def __init__(
        self,
        dt: float = 0.15,
        process_pos_std: float = 0.0015,
        process_vel_std: float = 0.04,
        mgd_std: float = 0.006,
        camera_std: float = 0.010,
    ) -> None:
        self.dt = float(dt)
        self.x = np.zeros(4, dtype=np.float64)
        self.P = np.diag([0.02, 0.02, 0.05, 0.05]) ** 2
        self.Q_base = np.diag([
            process_pos_std**2, process_pos_std**2,
            process_vel_std**2, process_vel_std**2,
        ])
        self.R_mgd = np.eye(2) * float(mgd_std)**2
        self.camera_std = float(camera_std)
        self.H = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]])
        self.initialized = False
        self.last_nis = float('nan')
        self.last_source = 'none'
        self.last_innovation = np.zeros(2)

    def reset(self, position_yz: np.ndarray) -> None:
        p = np.asarray(position_yz, dtype=np.float64).reshape(2)
        self.x[:] = [p[0], p[1], 0.0, 0.0]
        self.P[:] = np.diag([1e-4, 1e-4, 5e-3, 5e-3])
        self.initialized = True
        self.last_source = 'reset'
        self.last_nis = float('nan')

    def predict(self, wall_velocity: np.ndarray | None = None, dt: float | None = None) -> None:
        if not self.initialized:
            return
        dt = self.dt if dt is None else float(dt)
        F = np.array([
            [1., 0., dt, 0.],
            [0., 1., 0., dt],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ])
        self.x = F @ self.x
        if wall_velocity is not None:
            v = np.asarray(wall_velocity, dtype=np.float64).reshape(2)
            if np.all(np.isfinite(v)):
                self.x[2:] = v
        scale = max(dt / max(self.dt, 1e-6), 0.25)
        self.P = F @ self.P @ F.T + self.Q_base * scale
        self.P = 0.5 * (self.P + self.P.T)

    def _update(self, measurement: np.ndarray, R: np.ndarray, source: str, nis_gate: float) -> bool:
        z = np.asarray(measurement, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(z)):
            return False
        if not self.initialized:
            self.reset(z)
            self.last_source = source
            return True
        innovation = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        Sinv = np.linalg.pinv(S)
        nis = float(innovation.T @ Sinv @ innovation)
        self.last_nis = nis
        self.last_innovation = innovation.copy()
        if not np.isfinite(nis) or nis > float(nis_gate):
            return False
        K = self.P @ self.H.T @ Sinv
        self.x = self.x + K @ innovation
        I_KH = np.eye(4) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.last_source = source
        return True

    def update_mgd(self, position_yz: np.ndarray, nis_gate: float = 25.0) -> bool:
        return self._update(position_yz, self.R_mgd, 'mgd', nis_gate)

    def update_camera(
        self,
        position_yz: np.ndarray,
        confidence: float,
        nis_gate: float = 25.0,
    ) -> bool:
        conf = float(np.clip(confidence, 0.05, 1.0))
        std = self.camera_std / np.sqrt(conf)
        R = np.eye(2) * std**2
        return self._update(position_yz, R, 'camera', nis_gate)

    @property
    def position(self) -> np.ndarray:
        return self.x[:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2:].copy()

    @property
    def uncertainty(self) -> np.ndarray:
        return np.sqrt(np.maximum(np.diag(self.P)[:2], 0.0))
