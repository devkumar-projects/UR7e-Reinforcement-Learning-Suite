"""No-motion helper to verify the physical board plane and laser tool axis."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState

from .calibrated_kinematics import CalibratedURKinematics, pose_matrix
from .common import JOINT_NAMES
from .laser_kinematics import CalibratedLaserWallModel


class LaserGeometryProbe(Node):
    def __init__(self) -> None:
        super().__init__('ur7e_laser_geometry_probe')
        self.declare_parameter('calibration_file', '')
        self.declare_parameter('wall_x', 1.0)
        self.declare_parameter('laser_origin_offset_m', 0.0)
        self.declare_parameter('wait_s', 4.0)
        self.q = np.zeros(6)
        self.qmap = {}
        self.have_q = False
        self.tcp_pos = np.zeros(3)
        self.tcp_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.have_tcp = False
        self.create_subscription(JointState, '/joint_states', self._joint, 20)
        self.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose', self._tcp, 20)

    def _joint(self, msg: JointState) -> None:
        if not self.qmap:
            self.qmap = {n: i for i, n in enumerate(msg.name)}
        if all(n in self.qmap for n in JOINT_NAMES):
            for j, n in enumerate(JOINT_NAMES):
                self.q[j] = float(msg.position[self.qmap[n]])
            self.have_q = True

    def _tcp(self, msg: PoseStamped) -> None:
        p, o = msg.pose.position, msg.pose.orientation
        self.tcp_pos[:] = [p.x, p.y, p.z]
        self.tcp_quat[:] = [o.x, o.y, o.z, o.w]
        self.have_tcp = True

    def run(self) -> bool:
        gp = self.get_parameter
        deadline = time.monotonic() + float(gp('wait_s').value)
        while rclpy.ok() and time.monotonic() < deadline and not (self.have_q and self.have_tcp):
            rclpy.spin_once(self, timeout_sec=0.05)
        if not (self.have_q and self.have_tcp):
            print('[FAIL] /joint_states or /tcp_pose_broadcaster/pose unavailable')
            return False
        path = Path(str(gp('calibration_file').value)).expanduser().resolve()
        kin = CalibratedURKinematics.from_yaml(path)
        kin.estimate_tcp_offset(self.q, pose_matrix(self.tcp_pos, self.tcp_quat))
        wall_x = float(gp('wall_x').value)
        offset = float(gp('laser_origin_offset_m').value)
        print('=== LASER GEOMETRY PROBE — NO MOTION ===')
        print('TCP base [m]:', np.round(self.tcp_pos, 6).tolist())
        print('Configured board plane: base x =', wall_x, 'm')
        print('Choose the row matching the physical beam direction and a positive ray distance.')
        valid = []
        for axis in ('tool_x', '-tool_x', 'tool_y', '-tool_y', 'tool_z', '-tool_z'):
            model = CalibratedLaserWallModel(kin, wall_x=wall_x, laser_axis=axis,
                                             origin_offset_m=offset)
            origin, direction = model.ray(self.q)
            dot = model.wall_dot(self.q)
            if abs(float(direction[0])) < 1e-9:
                distance = float('inf')
            else:
                distance = (wall_x - float(origin[0])) / float(direction[0])
            state = 'VALID' if dot is not None else 'invalid'
            print(f'{axis:>7}: dir={np.round(direction, 5).tolist()} '
                  f'ray={distance:+.3f} m  yz={None if dot is None else np.round(dot, 5).tolist()}  {state}')
            if dot is not None:
                valid.append(axis)
        print('Valid candidate axes:', ', '.join(valid) if valid else 'none')
        print('Set LASER_AXIS in config/real.env only after visually confirming the beam mount.')
        return bool(valid)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LaserGeometryProbe()
    ok = node.run()
    node.destroy_node()
    rclpy.shutdown()
    if not ok:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
