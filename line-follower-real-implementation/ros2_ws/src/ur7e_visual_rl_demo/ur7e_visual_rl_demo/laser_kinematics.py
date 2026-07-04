"""Calibrated MGD/MGI for a laser rigidly attached to the active UR7e TCP.

The official UR calibration YAML is used for the six fixed transforms.  At
startup the active TCP offset is estimated from ``tcp_pose_broadcaster``.  The
laser ray is then intersected with the physical drawing plane ``x=wall_x``.
"""
from __future__ import annotations

import math
import numpy as np

from .calibrated_kinematics import CalibratedURKinematics


def _axis_vector(axis_name: str) -> np.ndarray:
    name = str(axis_name).strip().lower().replace(' ', '')
    axes = {
        'tool_x': np.array([1.0, 0.0, 0.0]),
        '+tool_x': np.array([1.0, 0.0, 0.0]),
        '-tool_x': np.array([-1.0, 0.0, 0.0]),
        'tool_y': np.array([0.0, 1.0, 0.0]),
        '+tool_y': np.array([0.0, 1.0, 0.0]),
        '-tool_y': np.array([0.0, -1.0, 0.0]),
        'tool_z': np.array([0.0, 0.0, 1.0]),
        '+tool_z': np.array([0.0, 0.0, 1.0]),
        '-tool_z': np.array([0.0, 0.0, -1.0]),
    }
    if name not in axes:
        raise ValueError(f'Unsupported LASER_AXIS={axis_name!r}')
    return axes[name].astype(np.float64)


class CalibratedLaserWallModel:
    def __init__(
        self,
        kinematics: CalibratedURKinematics,
        wall_x: float = 1.0,
        laser_axis: str = 'tool_z',
        origin_offset_m: float = 0.0,
        max_ray_length_m: float = 3.0,
    ) -> None:
        self.kin = kinematics
        self.wall_x = float(wall_x)
        self.axis_local = _axis_vector(laser_axis)
        self.origin_offset_m = float(origin_offset_m)
        self.max_ray_length_m = float(max_ray_length_m)

    def ray(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        T = self.kin.tcp_transform(np.asarray(q, dtype=np.float64).reshape(6))
        direction = T[:3, :3] @ self.axis_local
        n = float(np.linalg.norm(direction))
        if n < 1e-12:
            raise RuntimeError('Laser direction is degenerate')
        direction /= n
        origin = T[:3, 3] + self.origin_offset_m * direction
        return origin, direction

    def wall_dot(self, q: np.ndarray) -> np.ndarray | None:
        origin, direction = self.ray(q)
        if abs(float(direction[0])) < 1e-5:
            return None
        t = (self.wall_x - float(origin[0])) / float(direction[0])
        if t <= 0.0 or t > self.max_ray_length_m:
            return None
        point = origin + t * direction
        yz = point[1:3]
        if not np.all(np.isfinite(yz)):
            return None
        return yz.astype(np.float64)

    def wall_jacobian(self, q: np.ndarray, eps: float = 2e-5) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).reshape(6)
        J = np.zeros((2, 6), dtype=np.float64)
        d0 = self.wall_dot(q)
        if d0 is None:
            return J
        for i in range(6):
            qp, qm = q.copy(), q.copy()
            qp[i] += eps
            qm[i] -= eps
            dp, dm = self.wall_dot(qp), self.wall_dot(qm)
            if dp is not None and dm is not None:
                J[:, i] = (dp - dm) / (2.0 * eps)
            elif dp is not None:
                J[:, i] = (dp - d0) / eps
            elif dm is not None:
                J[:, i] = (d0 - dm) / eps
        J[~np.isfinite(J)] = 0.0
        return J

    def condition(self, q: np.ndarray) -> float:
        J = self.wall_jacobian(q)
        s = np.linalg.svd(J, compute_uv=False)
        if len(s) < 2 or s[-1] < 1e-10:
            return math.inf
        return float(s[0] / s[-1])

    def solve_wall_velocity(
        self,
        q: np.ndarray,
        wall_velocity_yz: np.ndarray,
        damping: float = 0.015,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        J = self.wall_jacobian(q)
        v = np.asarray(wall_velocity_yz, dtype=np.float64).reshape(2)
        A = J @ J.T + float(damping) ** 2 * np.eye(2)
        try:
            qdot = J.T @ np.linalg.solve(A, v)
        except np.linalg.LinAlgError:
            qdot = np.linalg.pinv(J) @ v
        return np.nan_to_num(qdot), self.condition(q), J
