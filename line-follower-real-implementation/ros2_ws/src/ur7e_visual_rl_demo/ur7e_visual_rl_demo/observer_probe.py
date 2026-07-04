"""No-motion preflight for the complete observer chain."""
from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from stable_baselines3 import SAC
from std_msgs.msg import Float32MultiArray

from .calibrated_kinematics import CalibratedURKinematics, pose_matrix
from .common import JOINT_NAMES, min_joint_margin
from .laser_kinematics import CalibratedLaserWallModel


class ObserverProbe(Node):
    def __init__(self) -> None:
        super().__init__('ur7e_observer_probe')
        for name, default in [
            ('model_path', ''), ('model_sha256', ''), ('calibration_file', ''),
            ('homography_file', ''), ('laser_axis', 'tool_z'),
        ]:
            self.declare_parameter(name, default)
        self.declare_parameter('wall_x', 1.0)
        self.declare_parameter('laser_origin_offset_m', 0.0)
        self.declare_parameter('probe_s', 5.0)
        self.declare_parameter('min_klt_confidence', 0.20)
        self.declare_parameter('max_mgd_camera_disagreement_m', 0.060)
        self.declare_parameter('max_condition', 100.0)

        self.q = np.zeros(6); self.qd = np.zeros(6); self.qmap = {}; self.have_q = False
        self.tcp_pos = np.zeros(3); self.tcp_quat = np.array([0., 0., 0., 1.]); self.have_tcp = False
        self.det = np.zeros(7); self.wall = np.zeros(11)
        self.last = {'joint': 0., 'tcp': 0., 'image': 0., 'det': 0., 'wall': 0.}
        self.count = {k: 0 for k in self.last}
        self.create_subscription(JointState, '/joint_states', self._joint, 30)
        self.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose', self._tcp, 30)
        self.create_subscription(Image, '/line_camera', self._image, qos_profile_sensor_data)
        self.create_subscription(Float32MultiArray, '/line_detection', self._det, 20)
        self.create_subscription(Float32MultiArray, '/camera_wall_measurement', self._wall, 20)
        self.client = ActionClient(self, FollowJointTrajectory,
                                   '/scaled_joint_trajectory_controller/follow_joint_trajectory')

    def _touch(self, name: str) -> None:
        self.last[name] = time.monotonic(); self.count[name] += 1

    def _joint(self, msg: JointState) -> None:
        if not self.qmap: self.qmap = {n: i for i, n in enumerate(msg.name)}
        if all(n in self.qmap for n in JOINT_NAMES):
            for j, n in enumerate(JOINT_NAMES):
                idx = self.qmap[n]; self.q[j] = msg.position[idx]
                if idx < len(msg.velocity): self.qd[j] = msg.velocity[idx]
            self.have_q = True; self._touch('joint')

    def _tcp(self, msg: PoseStamped) -> None:
        p, o = msg.pose.position, msg.pose.orientation
        self.tcp_pos[:] = [p.x, p.y, p.z]; self.tcp_quat[:] = [o.x, o.y, o.z, o.w]
        self.have_tcp = True; self._touch('tcp')

    def _image(self, _msg: Image) -> None: self._touch('image')
    def _det(self, msg: Float32MultiArray) -> None:
        if len(msg.data) >= 7: self.det[:] = msg.data[:7]; self._touch('det')
    def _wall(self, msg: Float32MultiArray) -> None:
        if len(msg.data) >= 11: self.wall[:] = msg.data[:11]; self._touch('wall')

    @staticmethod
    def _check_homography(path: Path) -> tuple[bool, str]:
        if not path.is_file(): return False, f'missing homography: {path}'
        try:
            data = yaml.safe_load(path.read_text()) or {}
            H = np.asarray(data.get('homography', []), dtype=float).reshape(3, 3)
            rmse = float(data.get('rmse_m', math.inf))
            if not np.all(np.isfinite(H)): return False, 'homography non-finite'
            if rmse > 0.020: return False, f'homography RMSE {rmse*1000:.1f} mm > 20 mm'
            return True, f'RMSE={rmse*1000:.2f} mm'
        except Exception as exc:
            return False, str(exc)

    def run(self) -> bool:
        gp = self.get_parameter
        model_path = Path(str(gp('model_path').value)).expanduser().resolve()
        expected = str(gp('model_sha256').value).strip().lower()
        calibration = Path(str(gp('calibration_file').value)).expanduser().resolve()
        homography = Path(str(gp('homography_file').value)).expanduser().resolve()
        print('=== UR7e CAMERA + LASER OBSERVER PREFLIGHT (NO MOTION) ===')

        if not model_path.is_file(): print('[FAIL] model missing:', model_path); return False
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if expected and digest != expected: print('[FAIL] model SHA mismatch'); return False
        model = SAC.load(str(model_path), device='cpu')
        if tuple(model.observation_space.shape) != (33,) or tuple(model.action_space.shape) != (2,):
            print('[FAIL] model contract mismatch'); return False
        print('[PASS] model', model_path.name, digest[:12])

        if not calibration.is_file(): print('[FAIL] calibration missing:', calibration); return False
        kin = CalibratedURKinematics.from_yaml(calibration)
        print('[PASS] factory calibration', calibration)
        ok_h, detail_h = self._check_homography(homography)
        print(('[PASS] ' if ok_h else '[FAIL] ') + 'camera homography ' + detail_h)
        if not ok_h: return False

        duration = float(gp('probe_s').value)
        start_count = self.count.copy(); t0 = time.monotonic()
        while rclpy.ok() and time.monotonic() - t0 < duration:
            rclpy.spin_once(self, timeout_sec=0.05)
        elapsed = max(time.monotonic() - t0, 1e-6)
        now = time.monotonic()
        for key in self.last:
            hz = (self.count[key] - start_count[key]) / elapsed
            age = 999. if self.last[key] <= 0 else now - self.last[key]
            print(f'{key:>6}: {hz:6.1f} Hz age={age:.3f}s')
            if hz < 5.0 or age > 0.5:
                print('[FAIL] stale/slow topic:', key); return False
        if not (self.have_q and self.have_tcp): print('[FAIL] missing q/tcp'); return False

        kin.estimate_tcp_offset(self.q, pose_matrix(self.tcp_pos, self.tcp_quat))
        laser = CalibratedLaserWallModel(
            kin, wall_x=float(gp('wall_x').value),
            laser_axis=str(gp('laser_axis').value),
            origin_offset_m=float(gp('laser_origin_offset_m').value),
        )
        mgd = laser.wall_dot(self.q)
        if mgd is None: print('[FAIL] MGD laser ray does not hit wall'); return False
        cam_valid = self.wall[0] > 0.5
        if not cam_valid: print('[FAIL] camera metric measurement invalid'); return False
        cam = self.wall[1:3]
        disagreement = float(np.linalg.norm(mgd - cam))
        cond = laser.condition(self.q)
        print('MGD laser [y,z] m:', np.round(mgd, 5).tolist())
        print('CAM laser [y,z] m:', np.round(cam, 5).tolist())
        print('MGD↔CAM disagreement:', round(disagreement*1000, 2), 'mm')
        print('KLT confidence:', round(float(self.det[2]), 3))
        print('Task condition:', round(cond, 2))
        print('Min joint margin:', round(min_joint_margin(self.q), 3), 'rad')
        if self.det[0] < 0.5 or self.det[6] < 0.5: print('[FAIL] line/laser not visible'); return False
        if self.det[2] < float(gp('min_klt_confidence').value): print('[FAIL] KLT confidence low'); return False
        if disagreement > float(gp('max_mgd_camera_disagreement_m').value): print('[FAIL] MGD/camera disagreement'); return False
        if not math.isfinite(cond) or cond > float(gp('max_condition').value): print('[FAIL] task condition'); return False
        if min_joint_margin(self.q) < 0.20: print('[FAIL] joint margin'); return False
        if not self.client.wait_for_server(timeout_sec=3.0): print('[FAIL] scaled trajectory action unavailable'); return False
        print('[PREFLIGHT PASS] Full chain: camera + KLT + homography + MGD + MGI availability + controller')
        return True


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ObserverProbe()
    ok = node.run()
    node.destroy_node(); rclpy.shutdown()
    if not ok: raise SystemExit(2)
