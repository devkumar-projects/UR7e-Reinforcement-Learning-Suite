"""Guarded real UR7e visual line-following runner.

Complete chain:
  camera -> HSV + KLT -> metric homography
  calibrated MGD + measured q/qdot -> EKF fusion with camera
  SAC action in wall [y,z]
  optional hybrid visual stabilizer (SAC remains active)
  calibrated differential MGI -> LQR singular-direction filter -> null-space term
  finite micro-trajectories through scaled_joint_trajectory_controller
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from stable_baselines3 import SAC
from std_msgs.msg import Float32MultiArray
from trajectory_msgs.msg import JointTrajectoryPoint

from .calibrated_kinematics import CalibratedURKinematics, pose_matrix
from .command_filter import filter_joint_command, manipulability
from .common import (
    JOINT_NAMES, TRAINING_HOME, TRAINING_HOME_DOT, TRAINING_HOME_TCP,
    TRAINING_MAX_WALL_SPEED_M_S, OBS_DIM, append_csv, clip_norm,
    min_joint_margin, q_normalized, vector_age, wrapped_joint_delta,
)
from .laser_kinematics import CalibratedLaserWallModel
from .observer_ekf import LaserWallEKF

MOVE_CONFIRMATION = 'MOVE_UR7E_CAMERA_LASER_RL'


def _unit(v: np.ndarray, fallback=(0.0, 1.0)) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(2)
    n = float(np.linalg.norm(v))
    return np.asarray(fallback, dtype=np.float64) if n < 1e-9 else v / n


def _clip_vec(v: np.ndarray, max_norm: float) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).copy()
    n = float(np.linalg.norm(v))
    if n > max_norm > 0.0:
        v *= max_norm / n
    return v


class VisualPolicyRunner(Node):
    def __init__(self) -> None:
        super().__init__('ur7e_visual_policy_runner')
        # Files and geometry
        self.declare_parameter('model_path', '')
        self.declare_parameter('calibration_file', '')
        self.declare_parameter('wall_x', 1.0)
        self.declare_parameter('laser_axis', 'tool_z')
        self.declare_parameter('laser_origin_offset_m', 0.0)
        # Execution
        self.declare_parameter('mode', 'shadow')
        self.declare_parameter('confirmation', '')
        self.declare_parameter('control_mode', 'hybrid_rl')
        self.declare_parameter('duration_s', 30.0)
        self.declare_parameter('segment_s', 0.15)
        self.declare_parameter('max_wall_speed_m_s', 0.010)
        self.declare_parameter('max_joint_speed_rad_s', 0.080)
        self.declare_parameter('rl_weight', 0.65)
        self.declare_parameter('visual_cross_track_gain', 1.2)
        self.declare_parameter('visual_forward_fraction', 0.85)
        self.declare_parameter('mgi_damping', 0.015)
        self.declare_parameter('enable_nullspace', True)
        # Observer/safety gates
        self.declare_parameter('min_klt_confidence', 0.20)
        self.declare_parameter('max_sensor_age_s', 0.50)
        self.declare_parameter('max_cross_track_m', 0.050)
        self.declare_parameter('max_mgd_camera_disagreement_m', 0.060)
        self.declare_parameter('max_task_condition', 100.0)
        self.declare_parameter('max_fk_validation_error_m', 0.015)
        self.declare_parameter('max_local_joint_delta_rad', 0.60)
        self.declare_parameter('max_ekf_nis', 25.0)
        self.declare_parameter('log_root', '~/.ros/ur7e_line_follower/camera_laser_runs')
        gp = self.get_parameter
        self.mode = str(gp('mode').value).strip().lower()
        self.control_mode = str(gp('control_mode').value).strip().lower()
        self.duration_s = float(gp('duration_s').value)
        self.segment_s = float(gp('segment_s').value)
        self.max_wall_speed = float(gp('max_wall_speed_m_s').value)
        self.max_joint_speed = float(gp('max_joint_speed_rad_s').value)
        self.rl_weight = float(np.clip(gp('rl_weight').value, 0.0, 1.0))
        self.cross_gain = float(gp('visual_cross_track_gain').value)
        self.forward_fraction = float(np.clip(gp('visual_forward_fraction').value, 0.0, 1.0))
        self.damping = float(gp('mgi_damping').value)
        self.enable_nullspace = bool(gp('enable_nullspace').value)
        self.min_klt = float(gp('min_klt_confidence').value)
        self.max_age = float(gp('max_sensor_age_s').value)
        self.max_cross = float(gp('max_cross_track_m').value)
        self.max_disagreement = float(gp('max_mgd_camera_disagreement_m').value)
        self.max_condition = float(gp('max_task_condition').value)
        self.max_fk_error = float(gp('max_fk_validation_error_m').value)
        self.max_local_delta = float(gp('max_local_joint_delta_rad').value)
        self.max_nis = float(gp('max_ekf_nis').value)

        if self.mode not in ('shadow', 'move'):
            raise ValueError('mode must be shadow or move')
        if self.control_mode not in ('rl_only', 'hybrid_rl', 'visual_only'):
            raise ValueError('control_mode must be rl_only, hybrid_rl or visual_only')
        if self.mode == 'move':
            if str(gp('confirmation').value) != MOVE_CONFIRMATION:
                raise RuntimeError(f'MOVE blocked: confirmation must be {MOVE_CONFIRMATION!r}')
            # Hard ceiling for this demonstration bundle.  The config defaults are lower.
            if self.duration_s > 60.0:
                raise RuntimeError('Real visual MOVE is hard-limited to 60 s')
            if self.max_wall_speed > 0.015:
                raise RuntimeError('Real visual MOVE is hard-limited to 15 mm/s')
            if self.max_joint_speed > 0.10:
                raise RuntimeError('Real visual MOVE is hard-limited to 0.10 rad/s')
        else:
            self.duration_s = min(max(self.duration_s, 5.0), 180.0)

        model_path = Path(str(gp('model_path').value)).expanduser().resolve()
        self.model = SAC.load(str(model_path), device='cpu')
        if tuple(self.model.observation_space.shape) != (OBS_DIM,) or tuple(self.model.action_space.shape) != (2,):
            raise RuntimeError('SAC model contract must be obs=33D, action=2D')
        calibration = Path(str(gp('calibration_file').value)).expanduser().resolve()
        kin = CalibratedURKinematics.from_yaml(calibration)
        self.laser_model = CalibratedLaserWallModel(
            kin, wall_x=float(gp('wall_x').value),
            laser_axis=str(gp('laser_axis').value),
            origin_offset_m=float(gp('laser_origin_offset_m').value),
        )
        self.ekf = LaserWallEKF(dt=self.segment_s)

        stamp = time.strftime('%Y%m%d_%H%M%S')
        root = Path(str(gp('log_root').value)).expanduser()
        self.run_dir = root / f'{self.mode}_{self.control_mode}_{stamp}'
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / 'policy_trace.csv'

        # ROS state
        self.q = np.zeros(6); self.qd = np.zeros(6); self.qmap = {}; self.have_q = False
        self.tcp_pos = np.zeros(3); self.tcp_quat = np.array([0., 0., 0., 1.]); self.have_tcp = False
        self.cam = np.zeros(7); self.guidance = np.zeros(3); self.wall = np.zeros(11)
        self.last = {'joint': 0., 'tcp': 0., 'image': 0., 'det': 0., 'guidance': 0., 'wall': 0.}
        self.q_ref = None; self.tcp_ref = None; self.dot_ref = None
        self.previous_action = np.zeros(2); self.previous_wall_velocity = np.zeros(2)
        self.last_loop = time.monotonic()

        self.create_subscription(JointState, '/joint_states', self._joint, 30)
        self.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose', self._tcp, 30)
        self.create_subscription(Image, '/line_camera', self._image, qos_profile_sensor_data)
        self.create_subscription(Float32MultiArray, '/line_detection', self._det, 20)
        self.create_subscription(Float32MultiArray, '/line_guidance', self._guidance, 20)
        self.create_subscription(Float32MultiArray, '/camera_wall_measurement', self._wall, 20)
        self.client = ActionClient(self, FollowJointTrajectory,
                                   '/scaled_joint_trajectory_controller/follow_joint_trajectory')

    def _joint(self, msg: JointState) -> None:
        if not self.qmap: self.qmap = {name: i for i, name in enumerate(msg.name)}
        if all(name in self.qmap for name in JOINT_NAMES):
            for j, name in enumerate(JOINT_NAMES):
                idx = self.qmap[name]; self.q[j] = msg.position[idx]
                if idx < len(msg.velocity): self.qd[j] = msg.velocity[idx]
            self.have_q = True; self.last['joint'] = time.monotonic()

    def _tcp(self, msg: PoseStamped) -> None:
        p, o = msg.pose.position, msg.pose.orientation
        self.tcp_pos[:] = [p.x, p.y, p.z]; self.tcp_quat[:] = [o.x, o.y, o.z, o.w]
        self.have_tcp = True; self.last['tcp'] = time.monotonic()

    def _image(self, _msg: Image) -> None: self.last['image'] = time.monotonic()
    def _det(self, msg: Float32MultiArray) -> None:
        if len(msg.data) >= 7:
            self.cam[:] = np.asarray(msg.data[:7], dtype=float); self.last['det'] = time.monotonic()
    def _guidance(self, msg: Float32MultiArray) -> None:
        if len(msg.data) >= 3:
            self.guidance[:] = np.asarray(msg.data[:3], dtype=float); self.last['guidance'] = time.monotonic()
    def _wall(self, msg: Float32MultiArray) -> None:
        if len(msg.data) >= 11:
            self.wall[:] = np.asarray(msg.data[:11], dtype=float); self.last['wall'] = time.monotonic()

    def spin_for(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.03)

    def initialize(self) -> tuple[bool, str]:
        self.spin_for(3.0)
        if not (self.have_q and self.have_tcp): return False, 'joint/TCP unavailable'
        if self.wall[0] < 0.5: return False, 'metric camera measurement unavailable'
        self.laser_model.kin.estimate_tcp_offset(self.q, pose_matrix(self.tcp_pos, self.tcp_quat))
        mgd = self.laser_model.wall_dot(self.q)
        if mgd is None: return False, 'MGD laser ray does not intersect board'
        camera = self.wall[1:3]
        self.ekf.reset(0.5 * (mgd + camera))
        self.ekf.update_mgd(mgd)
        self.ekf.update_camera(camera, max(float(self.wall[7]), 0.1))
        self.q_ref = self.q.copy(); self.tcp_ref = self.tcp_pos.copy(); self.dot_ref = self.ekf.position.copy()
        return True, 'ok'

    def update_ekf(self) -> tuple[np.ndarray | None, np.ndarray | None, bool]:
        now = time.monotonic(); dt = float(np.clip(now - self.last_loop, 0.01, 0.5)); self.last_loop = now
        mgd = self.laser_model.wall_dot(self.q)
        if mgd is None: return None, None, False
        J = self.laser_model.wall_jacobian(self.q)
        self.ekf.predict(J @ self.qd, dt=dt)
        self.ekf.update_mgd(mgd, nis_gate=self.max_nis)
        camera = None; camera_ok = False
        if self.wall[0] > 0.5:
            camera = self.wall[1:3].copy()
            camera_ok = self.ekf.update_camera(camera, float(self.wall[7]), nis_gate=self.max_nis)
        return mgd, camera, camera_ok

    def safety(self, mgd: np.ndarray | None, camera: np.ndarray | None) -> tuple[bool, str, float, float, float]:
        stale = [k for k, t in self.last.items() if vector_age(t) > self.max_age]
        if stale: return False, 'stale_' + '_'.join(stale), math.inf, math.inf, math.inf
        if self.cam[0] < 0.5: return False, 'line_lost', math.inf, math.inf, math.inf
        if self.cam[6] < 0.5: return False, 'laser_lost', math.inf, math.inf, math.inf
        if self.cam[2] < self.min_klt: return False, 'klt_low', math.inf, math.inf, math.inf
        if self.wall[0] < 0.5 or camera is None: return False, 'camera_metric_invalid', math.inf, math.inf, math.inf
        if self.q_ref is None: return False, 'adapter_not_initialized', math.inf, math.inf, math.inf
        if np.max(np.abs(wrapped_joint_delta(self.q, self.q_ref))) > self.max_local_delta:
            return False, 'outside_local_adapter_region', math.inf, math.inf, math.inf
        if min_joint_margin(self.q) < 0.20: return False, 'joint_limit_margin', math.inf, math.inf, math.inf
        if mgd is None: return False, 'mgd_invalid', math.inf, math.inf, math.inf
        disagreement = float(np.linalg.norm(mgd - camera))
        cross = float(np.linalg.norm(self.wall[8:10]))
        cond = self.laser_model.condition(self.q)
        fk_error = float(np.linalg.norm(self.laser_model.kin.position(self.q) - self.tcp_pos))
        if disagreement > self.max_disagreement: return False, 'mgd_camera_disagreement', cond, disagreement, cross
        if cross > self.max_cross: return False, 'cross_track_too_large', cond, disagreement, cross
        if not math.isfinite(cond) or cond > self.max_condition: return False, 'task_singularity', cond, disagreement, cross
        if fk_error > self.max_fk_error: return False, 'calibrated_fk_mismatch', cond, disagreement, cross
        return True, 'ok', cond, disagreement, cross

    def manip_obs(self, q_model: np.ndarray) -> np.ndarray:
        # Keep the observation scale close to training while command safety uses
        # the calibrated physical Jacobian separately.
        Jw = self.laser_model.wall_jacobian(self.q)
        Jtcp = self.laser_model.kin.geometric_jacobian(self.q)[:3]
        sw = np.linalg.svd(Jw, compute_uv=False)
        st = np.linalg.svd(Jtcp, compute_uv=False)
        w = manipulability(Jw)
        return np.array([
            np.clip(w / 0.115, 0., 1.),
            np.clip((st[-1] if len(st) else 0.) / 0.8, 0., 1.),
            np.clip((st[0] if len(st) else 0.) / 0.8, 0., 1.),
        ], dtype=np.float32)

    def observation(self) -> np.ndarray:
        assert self.q_ref is not None and self.tcp_ref is not None and self.dot_ref is not None
        q_model = TRAINING_HOME + wrapped_joint_delta(self.q, self.q_ref)
        tcp_model = TRAINING_HOME_TCP + (self.tcp_pos - self.tcp_ref)
        dot_model = TRAINING_HOME_DOT + (self.ekf.position - self.dot_ref)
        guidance = np.clip(np.nan_to_num(self.guidance), -1., 1.)
        cam = np.clip(np.nan_to_num(self.cam), -1., 1.)
        sigma = np.clip(self.ekf.uncertainty / 0.05, 0., 1.)
        prev_wall = np.clip(self.previous_wall_velocity / TRAINING_MAX_WALL_SPEED_M_S, -1., 1.)
        obs = np.concatenate([
            q_normalized(q_model), tcp_model, dot_model, [1.0],
            guidance[:2], [float(np.clip(guidance[2], 0., 1.))], cam,
            TRAINING_HOME_DOT + (self.ekf.position - self.dot_ref), sigma,
            self.manip_obs(q_model), self.previous_action, prev_wall,
        ]).astype(np.float32)
        if obs.shape != (33,) or not np.all(np.isfinite(obs)):
            raise RuntimeError(f'invalid observation {obs.shape}')
        return np.clip(obs, -2., 2.)

    def command(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        action, _ = self.model.predict(obs, deterministic=True)
        action = np.clip(np.asarray(action, dtype=float).reshape(2), -1., 1.)
        raw_rl = action * self.max_wall_speed
        raw_rl = _clip_vec(raw_rl, self.max_wall_speed)
        # EMA identical in spirit to training control.
        v_rl = 0.5 * self.previous_wall_velocity + 0.5 * raw_rl

        tangent = _unit(self.wall[5:7])
        cross = self.wall[8:10]
        v_visual = tangent * (self.forward_fraction * self.max_wall_speed) + self.cross_gain * cross
        v_visual = _clip_vec(v_visual, self.max_wall_speed)
        if self.control_mode == 'rl_only':
            v_cmd = v_rl
        elif self.control_mode == 'visual_only':
            v_cmd = v_visual
        else:
            v_cmd = self.rl_weight * v_rl + (1.0 - self.rl_weight) * v_visual
            min_forward = 0.10 * self.max_wall_speed
            forward = float(np.dot(v_cmd, tangent))
            if forward < min_forward:
                v_cmd += (min_forward - forward) * tangent
        v_cmd = _clip_vec(v_cmd, self.max_wall_speed)

        qdot_raw, condition, J = self.laser_model.solve_wall_velocity(self.q, v_cmd, damping=self.damping)
        qdot, diag = filter_joint_command(
            self.q, qdot_raw, self.laser_model.wall_jacobian,
            max_joint_speed=self.max_joint_speed, damping=self.damping,
            enable_nullspace=self.enable_nullspace,
        )
        qdot = clip_norm(qdot, self.max_joint_speed * math.sqrt(2.0))
        diag['condition'] = max(float(diag.get('condition', 0.0)), condition)
        diag['v_rl_y'] = float(v_rl[0]); diag['v_rl_z'] = float(v_rl[1])
        diag['v_visual_y'] = float(v_visual[0]); diag['v_visual_z'] = float(v_visual[1])
        return action, v_cmd, qdot, diag

    def send_microtrajectory(self, qdot: np.ndarray) -> bool:
        target = self.q + np.asarray(qdot) * self.segment_s
        max_delta = self.max_joint_speed * self.segment_s
        target = self.q + np.clip(target - self.q, -max_delta, max_delta)
        goal = FollowJointTrajectory.Goal(); goal.trajectory.joint_names = list(JOINT_NAMES)
        point = JointTrajectoryPoint(); point.positions = target.tolist(); point.velocities = [0.] * 6
        point.time_from_start = Duration(seconds=self.segment_s).to_msg()
        goal.trajectory.points = [point]
        send = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send, timeout_sec=2.0)
        if not send.done() or send.result() is None or not send.result().accepted: return False
        handle = send.result(); result = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result, timeout_sec=self.segment_s + 2.0)
        if not result.done():
            handle.cancel_goal_async(); return False
        wrapped = result.result()
        return bool(wrapped is not None and wrapped.status == GoalStatus.STATUS_SUCCEEDED)

    def log(self, step: int, safe: bool, reason: str, mgd, camera, cond, disagreement, cross,
            action=None, v_cmd=None, qdot=None, diag=None) -> None:
        action = np.zeros(2) if action is None else action
        v_cmd = np.zeros(2) if v_cmd is None else v_cmd
        qdot = np.zeros(6) if qdot is None else qdot
        diag = {} if diag is None else diag
        append_csv(self.csv_path, {
            'time': time.time(), 'step': step, 'mode': self.mode, 'control_mode': self.control_mode,
            'safe': int(safe), 'reason': reason,
            **{f'q{i}': self.q[i] for i in range(6)},
            'tcp_x': self.tcp_pos[0], 'tcp_y': self.tcp_pos[1], 'tcp_z': self.tcp_pos[2],
            'mgd_y': np.nan if mgd is None else mgd[0], 'mgd_z': np.nan if mgd is None else mgd[1],
            'cam_y': np.nan if camera is None else camera[0], 'cam_z': np.nan if camera is None else camera[1],
            'ekf_y': self.ekf.position[0], 'ekf_z': self.ekf.position[1], 'ekf_nis': self.ekf.last_nis,
            'klt': self.cam[2], 'progress': self.wall[10], 'cross_track_m': cross,
            'mgd_camera_m': disagreement, 'condition': cond,
            'action_y': action[0], 'action_z': action[1], 'cmd_y': v_cmd[0], 'cmd_z': v_cmd[1],
            **{f'qd{i}': qdot[i] for i in range(6)},
            'lqr_gain_max': diag.get('lqr_gain_max', 0.0),
            'null_norm': diag.get('null_norm', 0.0),
        })

    def run(self) -> bool:
        print('=== UR7e CAMERA + LASER SAC LINE FOLLOWER ===')
        print('MODE:', self.mode.upper(), '| CONTROL:', self.control_mode)
        print('Observers: KLT + calibrated MGD + camera homography + EKF')
        print('Control: SAC -> differential MGI -> LQR + null-space -> scaled trajectory controller')
        print('Limits:', f'{self.max_wall_speed*1000:.1f} mm/s, {self.max_joint_speed:.3f} rad/s, {self.duration_s:.1f}s')
        print('Log:', self.run_dir)
        ok, reason = self.initialize()
        if not ok: print('[FAIL]', reason); return False
        if not self.client.wait_for_server(timeout_sec=3.0): print('[FAIL] trajectory controller unavailable'); return False

        start = time.monotonic(); step = 0; safe_streak = 0; goal_streak = 0
        safe_count = 0; blocked_count = 0
        while rclpy.ok() and time.monotonic() - start < self.duration_s:
            self.spin_for(0.04)
            mgd, camera, camera_update = self.update_ekf()
            safe, reason, cond, disagreement, cross = self.safety(mgd, camera)
            if not safe:
                blocked_count += 1
                safe_streak = 0; self.log(step, False, reason, mgd, camera, cond, disagreement, cross)
                print(f'[BLOCKED step={step}] {reason}')
                if self.mode == 'move':
                    print('[MOVE ABORT] no new trajectory; controller holds the last finite target')
                    return False
                self.spin_for(0.10); step += 1; continue
            safe_count += 1
            safe_streak += 1
            obs = self.observation()
            action, v_cmd, qdot, diag = self.command(obs)
            cond = max(cond, float(diag['condition']))
            self.log(step, True, 'ok', mgd, camera, cond, disagreement, cross, action, v_cmd, qdot, diag)
            print(
                f'[{self.mode} {step:03d}] a=({action[0]:+.2f},{action[1]:+.2f}) '
                f'v=({v_cmd[0]*1000:+.1f},{v_cmd[1]*1000:+.1f})mm/s '
                f'cross={cross*1000:.1f}mm KLT={self.cam[2]:.2f} '
                f'MGD-CAM={disagreement*1000:.1f}mm EKFnis={self.ekf.last_nis:.1f} '
                f'cond={cond:.1f} qd={np.max(np.abs(qdot)):.3f} progress={100*self.wall[10]:.0f}%'
            )
            if self.wall[10] >= 0.985:
                goal_streak += 1
                if goal_streak >= 5:
                    print('[DRAWING COMPLETE] visual progress >= 98.5%')
                    break
            else:
                goal_streak = 0
            if self.mode == 'move':
                if safe_streak < 8:
                    self.spin_for(0.08); step += 1; continue
                if not self.send_microtrajectory(qdot):
                    print('[MOVE ABORT] trajectory action failed'); return False
            else:
                self.spin_for(self.segment_s)
            self.previous_action = action; self.previous_wall_velocity = v_cmd
            step += 1
        total = safe_count + blocked_count
        blocked_ratio = blocked_count / max(total, 1)
        print(f'[SUMMARY] safe={safe_count} blocked={blocked_count} blocked_ratio={100.0*blocked_ratio:.1f}%')
        if self.mode == 'shadow' and (safe_count < 5 or blocked_ratio > 0.25):
            print('[SHADOW FAIL] observer chain was not continuously reliable enough')
            return False
        print(f'[{self.mode.upper()} COMPLETE] trace={self.csv_path}')
        return True


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisualPolicyRunner()
    ok = node.run()
    node.destroy_node(); rclpy.shutdown()
    if not ok: raise SystemExit(2)


if __name__ == '__main__':
    main()
