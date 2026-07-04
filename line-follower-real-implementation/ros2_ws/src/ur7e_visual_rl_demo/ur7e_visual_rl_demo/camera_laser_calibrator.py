"""One-time eye-to-hand calibration using the physical laser as the probe.

The operator manually jogs the laser to well-spread locations on the board and
presses ENTER.  Each sample pairs the detected laser pixel with the calibrated
MGD intersection of the same laser ray with the board plane.  A planar
homography ``pixel -> [wall_y, wall_z]`` is saved for camera/EKF fusion.
"""
from __future__ import annotations

import select
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray

from .calibrated_kinematics import CalibratedURKinematics, pose_matrix
from .common import JOINT_NAMES
from .laser_kinematics import CalibratedLaserWallModel


class CameraLaserCalibrator(Node):
    def __init__(self) -> None:
        super().__init__('camera_laser_calibrator')
        self.declare_parameter('calibration_file', '')
        self.declare_parameter('output_file', '')
        self.declare_parameter('wall_x', 1.0)
        self.declare_parameter('laser_axis', 'tool_z')
        self.declare_parameter('laser_origin_offset_m', 0.0)
        self.declare_parameter('required_samples', 12)
        gp = self.get_parameter
        calibration = Path(str(gp('calibration_file').value)).expanduser().resolve()
        self.output = Path(str(gp('output_file').value)).expanduser().resolve()
        self.required = max(6, int(gp('required_samples').value))
        kin = CalibratedURKinematics.from_yaml(calibration)
        self.model = CalibratedLaserWallModel(
            kin, wall_x=float(gp('wall_x').value),
            laser_axis=str(gp('laser_axis').value),
            origin_offset_m=float(gp('laser_origin_offset_m').value),
        )
        self.q = np.zeros(6)
        self.q_map = {}
        self.have_q = False
        self.tcp_pos = np.zeros(3)
        self.tcp_quat = np.array([0., 0., 0., 1.])
        self.have_tcp = False
        self.measurement = np.zeros(12)
        self.measurement_time = 0.0
        self.offset_ready = False
        self.pixel_samples: list[list[float]] = []
        self.wall_samples: list[list[float]] = []
        self.create_subscription(JointState, '/joint_states', self._joint_cb, 20)
        self.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose', self._tcp_cb, 20)
        self.create_subscription(Float32MultiArray, '/line_measurement', self._measurement_cb, 20)

    def _joint_cb(self, msg: JointState) -> None:
        if not self.q_map:
            self.q_map = {name: i for i, name in enumerate(msg.name)}
        if all(name in self.q_map for name in JOINT_NAMES):
            for i, name in enumerate(JOINT_NAMES):
                self.q[i] = float(msg.position[self.q_map[name]])
            self.have_q = True

    def _tcp_cb(self, msg: PoseStamped) -> None:
        p, o = msg.pose.position, msg.pose.orientation
        self.tcp_pos[:] = [p.x, p.y, p.z]
        self.tcp_quat[:] = [o.x, o.y, o.z, o.w]
        self.have_tcp = True

    def _measurement_cb(self, msg: Float32MultiArray) -> None:
        if len(msg.data) >= 12:
            self.measurement[:] = np.asarray(msg.data[:12], dtype=np.float64)
            self.measurement_time = time.monotonic()

    def initialize_offset(self) -> bool:
        if not (self.have_q and self.have_tcp):
            return False
        measured = pose_matrix(self.tcp_pos, self.tcp_quat)
        self.model.kin.estimate_tcp_offset(self.q, measured)
        self.offset_ready = True
        return True

    def sample(self) -> tuple[bool, str]:
        if not self.offset_ready and not self.initialize_offset():
            return False, 'joint state / TCP pose unavailable'
        if time.monotonic() - self.measurement_time > 0.5:
            return False, 'line detector measurement stale'
        if self.measurement[0] < 0.5 or self.measurement[10] < 0.5:
            return False, 'red laser spot not detected'
        dot = self.model.wall_dot(self.q)
        if dot is None:
            return False, 'MGD laser ray does not intersect configured wall plane'
        uv = self.measurement[1:3].copy()
        if self.pixel_samples:
            pixel_dist = min(float(np.linalg.norm(uv - np.asarray(p))) for p in self.pixel_samples)
            wall_dist = min(float(np.linalg.norm(dot - np.asarray(p))) for p in self.wall_samples)
            if pixel_dist < 18.0 or wall_dist < 0.025:
                return False, 'sample too close to an existing point; spread the laser farther'
        self.pixel_samples.append(uv.tolist())
        self.wall_samples.append(dot.tolist())
        return True, f'uv=({uv[0]:.1f},{uv[1]:.1f}) -> wall=({dot[0]:+.4f},{dot[1]:+.4f}) m'

    def fit_and_save(self) -> tuple[bool, str]:
        if len(self.pixel_samples) < 6:
            return False, 'at least six samples are required'
        src = np.asarray(self.pixel_samples, dtype=np.float64)
        dst = np.asarray(self.wall_samples, dtype=np.float64)
        H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=0.015)
        if H is None:
            return False, 'cv2.findHomography failed'
        src_h = np.column_stack([src, np.ones(len(src))])
        pred_h = (H @ src_h.T).T
        pred = pred_h[:, :2] / pred_h[:, 2:3]
        err = np.linalg.norm(pred - dst, axis=1)
        inliers = np.ones(len(src), dtype=bool) if mask is None else mask.reshape(-1).astype(bool)
        rmse = float(np.sqrt(np.mean(err[inliers] ** 2))) if np.any(inliers) else float('inf')
        inlier_count = int(np.count_nonzero(inliers))
        if inlier_count < max(9, int(np.ceil(0.65 * len(src)))):
            return False, f'not enough consistent samples: {inlier_count}/{len(src)} inliers'
        if not np.isfinite(rmse) or rmse > 0.010:
            return False, f'calibration RMSE too high: {rmse*1000:.1f} mm'
        data = {
            'schema_version': 1,
            'description': 'Pixel to physical drawing-plane [y,z] homography',
            'homography': H.reshape(-1).tolist(),
            'rmse_m': rmse,
            'inlier_count': int(np.count_nonzero(inliers)),
            'sample_count': int(len(src)),
            'pixel_samples': src.tolist(),
            'wall_yz_samples_m': dst.tolist(),
            'created_unix': time.time(),
        }
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
        return True, f'saved {self.output} | RMSE={rmse*1000:.2f} mm | inliers={np.count_nonzero(inliers)}/{len(src)}'

    def run(self) -> bool:
        print('\n=== CAMERA / LASER PLANE CALIBRATION ===', flush=True)
        print('Robot en manuel/réduit. External Control non lancé.', flush=True)
        print(f'ENTER=point | status | undo | drop N | save | quit ({self.required} points)', flush=True)
        last_status = 0.0
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            now = time.monotonic()
            if now - last_status > 3.0:
                age = now - self.measurement_time
                if self.have_q and self.have_tcp and age < 0.8:
                    print('[PRET] Appuie sur ENTREE.', flush=True)
                else:
                    print(f'[ATTENTE] joint={self.have_q} tcp={self.have_tcp} mesure_age={age:.2f}s', flush=True)
                last_status = now
            readable, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not readable:
                continue
            raw = sys.stdin.readline()
            if raw == '':
                return False
            command = raw.strip().lower()
            if command in ('q', 'quit', 'exit'):
                print('Calibration annulée.', flush=True)
                return False
            if command == 'status':
                print(f'[STATUS] joint={self.have_q} tcp={self.have_tcp} age={time.monotonic()-self.measurement_time:.3f}s laser={self.measurement[10]:.0f}', flush=True)
                continue
            if command == 'undo':
                if self.pixel_samples:
                    self.pixel_samples.pop(); self.wall_samples.pop()
                print(f'Samples: {len(self.pixel_samples)}/{self.required}', flush=True)
                continue
            if command.startswith('drop '):
                try:
                    idx = int(command.split()[1]) - 1
                    if idx < 0 or idx >= len(self.pixel_samples):
                        raise ValueError
                    self.pixel_samples.pop(idx); self.wall_samples.pop(idx)
                    print(f'[DROP] Point {idx+1} supprimé.', flush=True)
                except Exception:
                    print('[DROP] Usage: drop N', flush=True)
                continue
            if command == 'save':
                ok, detail = self.fit_and_save()
                print(('[PASS] ' if ok else '[FAIL] ') + detail, flush=True)
                return ok
            if not self.have_q:
                print('[SAMPLE REJECTED] /joint_states absent', flush=True); continue
            if not self.have_tcp:
                print('[SAMPLE REJECTED] TCP absent', flush=True); continue
            if time.monotonic() - self.measurement_time > 0.8:
                print('[SAMPLE REJECTED] mesure caméra périmée', flush=True); continue
            ok, detail = self.sample()
            print(('[SAMPLE PASS] ' if ok else '[SAMPLE REJECTED] ') + detail, flush=True)
            print(f'Samples: {len(self.pixel_samples)}/{self.required}', flush=True)
            if len(self.pixel_samples) >= self.required:
                ok, detail = self.fit_and_save()
                print(('[PASS] ' if ok else '[FAIL] ') + detail, flush=True)
                return ok
        return False


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CameraLaserCalibrator()
    ok = node.run()
    node.destroy_node()
    rclpy.shutdown()
    if not ok:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
