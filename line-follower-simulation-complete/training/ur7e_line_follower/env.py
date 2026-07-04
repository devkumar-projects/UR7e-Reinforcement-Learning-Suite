"""Gymnasium environment for UR7e laser line following — schema V5 (33D).

Observation V5:
[0:6]   q_norm                 normalised joint positions
[6:9]   tcp                    TCP position (x, y, z)
[9:11]  dot_yz                 laser/wall-plane intersection from encoders
[11]    on_wall                1 when the intersection is inside the wall
[12:15] visual_guidance        real image lookahead or privileged debug target
[15:22] camera V4              line/KLT/tangent/coverage/laser detection
[22:24] ekf_pos                EKF wall position
[24:26] ekf_sigma              normalised EKF uncertainty
[26:29] manipulability         wall-task manipulability features
[29:31] previous_action        previous normalised SAC action
[31:33] previous_wall_velocity previous filtered wall velocity / max speed

Action: normalised wall-plane velocity [v_y, v_z].

V5 fixes the saturated tracking reward, historical-max progression bug, stale
continuous commands during SAC updates, hidden EMA state, unrecoverable global
RMSE success condition, and flat off-wall transitions.
"""
from __future__ import annotations

import json
import math
import time
import pathlib
from collections import deque
import numpy as np
import gymnasium
from gymnasium import spaces
import rclpy

from .bridge import (LineFollowerBridge, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH,
                     ELBOW_COMMAND_LIMIT_HIGH, WALL_X)
from .kinematics import fk_ur, wall_jacobian, jacobian_condition
from .control import (ACTION_SPACE_DIM, CONTROL_SCHEMA_VERSION,
                      MAX_WALL_SPEED_M_S, MAX_JOINT_SPEED_RAD_S,
                      wall_action_to_joint_velocity)
from .target_line import (load_line, random_line, random_line_from_start, curriculum_line_from_start,
                          straight_line_from_start, anchor_line_to_start, waypoint_abscissae,
                          DEFAULT_HOME_DOT, RMSE_SUCCESS_THRESH,
                          WALL_Y_MIN, WALL_Y_MAX, WALL_Z_MIN, WALL_Z_MAX,
                          closest_point_on_polyline, arc_length)
from .singularity import manipulability_obs, singularity_penalty, yoshikawa
from .trajectory_store import load_current_trajectory, save_current_trajectory
from .reward import (DEFAULT_REWARD_PROFILE, REWARD_PROFILES, tracking_reward,
                     gated_progress_reward, distance_outside_rectangle,
                     offwall_penalty as compute_offwall_penalty, recent_rmse)

# ── Schéma V5 ────────────────────────────────────────────────────────────────
OBSERVATION_SCHEMA_VERSION: int = 5
OBSERVATION_SPACE_DIM:      int = 33

# Caméra statique définie dans worlds/line_follower.sdf : 75° horizontal.
# Le détecteur redimensionne le flux 640x480 en 320x240.
_CAM_W, _CAM_H = 320.0, 240.0
_FOV_H    = 1.3090
_FX       = _CAM_W / (2.0 * math.tan(_FOV_H / 2.0))      # ≈ 208.5 px/m @ 1m
_SCALE_PX = math.sqrt(_CAM_W**2 + _CAM_H**2) / 2.0       # = 200.0 px

SIM_CONTROL_RATE_HZ = 250.0
# Une action RL doit être réellement appliquée assez longtemps pour produire
# un déplacement mesurable. La V2.2 ne la maintenait que 5 cycles (≈20 ms),
# puis envoyait zéro : le bras paraissait immobile.
PHYSICS_STEPS = 25      # 0,10 s à 250 Hz
SETTLE_STEPS  = 5
PHYSICS_TIMEOUT_S = 4.0
SETTLE_TIMEOUT_S  = 2.0
RL_DT         = PHYSICS_STEPS / SIM_CONTROL_RATE_HZ
MAX_STEPS     = 300
MAX_JVEL      = MAX_JOINT_SPEED_RAD_S
TRIES_PER_TRAJ   = 5
WAYPOINT_THRESH  = 0.035
WAYPOINT_BONUS   = 0.25
COMPLETION_BONUS = 10.0
MAX_DOT_DIST  = 0.50
TRACKING_NEAR_SCALE_M = 0.04
TRACKING_GATE_M       = 0.10
PROGRESS_STEP_SCALE_M = MAX_WALL_SPEED_M_S * RL_DT
PROGRESS_REWARD_GAIN  = 2.0
VISION_LOSS_PENALTY    = 0.20
ACTION_DELTA_PENALTY_GAIN = 0.02
STAGNATION_GRACE_STEPS = 12    # was 20 — pénaliser l'immobilité plus vite
STAGNATION_ABORT_STEPS = 80
STAGNATION_PROGRESS_EPS_M = 0.0010
STAGNATION_PENALTY_MAX = 0.75
REWARD_SCALE = 0.1             # normalise Q-values vers [-50,+5] au lieu de [-500,+50]
CURRICULUM_SUCCESS_THRESHOLD = 0.85
CURRICULUM_MIN_EPISODES = 20
CURRICULUM_WINDOW = 50
ORDERED_WINDOW = 8
WAYPOINT_ARC_TOL_M = 0.006
WAYPOINT_ACCEPTANCE_M = 0.08
SUCCESS_WINDOW_STEPS = 30
OFFWALL_ABORT_STEPS = 5
OFFWALL_TERMINAL_PENALTY = -3.0

_VISUAL_GRACE_S  = 2.0
_LINE_TIMEOUT_S  = 3.0
_LASER_TIMEOUT_S = 3.0


class UR7eLineFollowerEnv(gymnasium.Env):
    metadata = {'render_modes': []}

    def __init__(self, line_path=None, line_shape: str = 's_curve',
                 random_trajectories: bool = True,
                 rmse_thresh: float = RMSE_SUCCESS_THRESH,
                 sensor_noise: bool = True,
                 update_dot_visual: bool = False,
                 observation_mode: str = 'real',
                 trials_per_trajectory: int = TRIES_PER_TRAJ,
                 curriculum: bool = True,
                 training_profile: str = 'realistic',
                 max_steps: int | None = None,
                 guided_reset: bool = True,
                 deterministic_pulse: bool = True,
                 fixed_line_length: float = 0.25,
                 reward_profile: str = DEFAULT_REWARD_PROFILE):
        """Create the corrected line-following environment.

        ``training_profile='minimal_straight_line_debug'`` forces a short, fixed,
        noise-free line with privileged guidance and a 120-step horizon.  It is
        the mandatory GO/NO-GO configuration before realistic visual training.
        """
        super().__init__()
        if training_profile not in ('realistic', 'minimal_straight_line_debug'):
            raise ValueError(f"training_profile invalide : {training_profile!r}")
        if reward_profile not in REWARD_PROFILES:
            raise ValueError(
                f"reward_profile invalide : {reward_profile!r}; attendu {REWARD_PROFILES}")
        self._training_profile = training_profile
        self._reward_profile = reward_profile
        self._minimal_profile = training_profile == 'minimal_straight_line_debug'
        if self._minimal_profile:
            observation_mode = 'privileged_debug'
            sensor_noise = False
            random_trajectories = False
            curriculum = False
            trials_per_trajectory = 10_000
            max_steps = 120 if max_steps is None else max_steps
        if observation_mode not in ('real', 'privileged_debug', 'zero'):
            raise ValueError(f"observation_mode invalide : {observation_mode!r}")

        self._observation_mode  = observation_mode
        self._sensor_noise      = bool(sensor_noise)
        self._joint_noise_sigma_rad = 0.002
        self._fk_noise_sigma_m  = 0.0
        self._cam_noise_sigma_m = 0.0
        self._update_visual     = bool(update_dot_visual)
        self._random_traj       = bool(random_trajectories)
        self._rmse_thresh       = float(rmse_thresh)
        self._trials_per_traj   = max(1, int(trials_per_trajectory))
        self._curriculum_enabled = bool(curriculum)
        self._guided_reset_enabled = bool(guided_reset)
        self._deterministic_pulse = bool(deterministic_pulse)
        self._fixed_line_length = float(max(fixed_line_length, 0.05))
        self._max_steps = int(MAX_STEPS if max_steps is None else max_steps)
        self._rng = np.random.default_rng()

        self.waypoints = (
            random_line_from_start(self._rng, DEFAULT_HOME_DOT)
            if random_trajectories else load_line(line_path, line_shape)
        )
        self._n_wp = len(self.waypoints)
        self._path_length = arc_length(self.waypoints)
        self._waypoint_s = waypoint_abscissae(self.waypoints)

        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(OBSERVATION_SPACE_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACTION_SPACE_DIM,), dtype=np.float32)

        if not rclpy.ok():
            rclpy.init()
        visual_required = (observation_mode == 'real') or self._update_visual
        self.node = LineFollowerBridge(visual_enabled=visual_required)

        self._step = 0
        self._total_steps = 0
        self._wp_idx = 0
        self._episode_num = 0
        self._tries_on_traj = 0
        self._ep_rewards: list[float] = []
        self._ep_distances: list[float] = []
        self._ep_ordered_distances: list[float] = []
        self._ep_offwall_count = 0
        self._ep_dot_path: list[np.ndarray] = []
        self._ep_reward_term_sums: dict[str, float] = {}
        self._last_action = np.zeros(ACTION_SPACE_DIM, dtype=np.float64)
        self._last_wall_velocity = np.zeros(2, dtype=np.float64)
        self._last_jvel_pre_filter = np.zeros(6, dtype=np.float64)
        self._last_reward_terms: dict[str, float] = {}
        self._last_closest = {
            'distance': np.nan, 'closest': np.array([np.nan, np.nan]),
            'segment_index': 0, 'abscissa': 0.0,
        }
        self._last_abscissa = 0.0
        self._max_abscissa = 0.0
        self._stagnation_steps = 0
        self._consecutive_offwall_steps = 0
        self._traj_count = 0
        self._curriculum_level_value = 0 if self._curriculum_enabled else 2
        self._recent_successes = deque(maxlen=CURRICULUM_WINDOW)
        self._episode_outcome_recorded = False

        self._last_cmd_raw = np.zeros(6, dtype=np.float64)
        self._last_cmd_lqr = np.zeros(6, dtype=np.float64)
        self._last_cmd_null = np.zeros(6, dtype=np.float64)
        self._last_cmd_out = np.zeros(6, dtype=np.float64)
        self._last_pulse_duration_s = 0.0
        self._fresh_detection = True

        self._save_dir = pathlib.Path.home() / '.ros' / 'ur7e_line_follower' / 'episodes'
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._cam_health_checked = False
        self._reset_mono = 0.0
        time.sleep(0.5)

    def set_sensor_noise(self, dot_std_m: float = 0.0,
                         joint_std_rad: float | None = None,
                         cam_std_m: float | None = None):
        self._sensor_noise = True
        self._fk_noise_sigma_m  = float(max(dot_std_m, 0.0))
        self._cam_noise_sigma_m = float(max(dot_std_m if cam_std_m is None else cam_std_m, 0.0))
        if joint_std_rad is not None:
            self._joint_noise_sigma_rad = float(max(joint_std_rad, 0.0))

    def _pulse(self, jvel: np.ndarray):
        """Apply one action for a deterministic physical interval, then stop.

        Gazebo runs in real time.  Stopping before the SAC gradient update avoids
        an action duration that depends on CPU/logging load.  Command diagnostics
        are snapshotted before ``stop()`` resets them to zero.
        """
        if hasattr(self.node, 'drain_callbacks'):
            self.node.drain_callbacks(max_cycles=20)
        detection_before = int(getattr(self.node, '_detection_frame_count', 0))
        t0 = time.monotonic()
        self.node.publish_velocity(jvel)
        moved = self.node.wait_for_n_steps(n_steps=PHYSICS_STEPS, timeout=PHYSICS_TIMEOUT_S)
        self._last_cmd_raw = np.asarray(self.node.last_cmd_raw, dtype=np.float64).copy()
        self._last_cmd_lqr = np.asarray(self.node.last_cmd_lqr, dtype=np.float64).copy()
        self._last_cmd_null = np.asarray(self.node.last_cmd_null, dtype=np.float64).copy()
        self._last_cmd_out = np.asarray(self.node.last_cmd_out, dtype=np.float64).copy()
        if self._deterministic_pulse:
            self.node.stop()
            self.node.wait_for_n_steps(n_steps=SETTLE_STEPS, timeout=SETTLE_TIMEOUT_S)
        self._fresh_detection = True
        if self._observation_mode == 'real':
            self._fresh_detection = bool(
                self.node.wait_for_detection_after(detection_before, timeout=0.20))
        self._last_pulse_duration_s = float(time.monotonic() - t0)
        return bool(moved)

    def _maybe_noisy_dot(self, dot):
        """Return the wall-plane intersection, even when it is outside the wall.

        The previous implementation replaced an off-wall FK point by ``None``,
        hiding the direction required to recover.  Joint encoders and the known
        laser geometry make this measurement available on both simulation and the
        real robot, so keeping it does not leak target information.
        """
        if dot is None:
            return None
        value = np.asarray(dot, dtype=np.float64).copy()
        if self._sensor_noise and self._fk_noise_sigma_m > 0:
            value += self.np_random.normal(0.0, self._fk_noise_sigma_m, size=2)
        return value

    @staticmethod
    def _is_on_wall(dot: np.ndarray | None) -> bool:
        if dot is None:
            return False
        y, z = np.asarray(dot, dtype=np.float64).reshape(2)
        return bool(WALL_Y_MIN <= y <= WALL_Y_MAX and WALL_Z_MIN <= z <= WALL_Z_MAX)

    def _action_speed_scale(self) -> float:
        """Competence-based speed curriculum while preserving action space."""
        if self._minimal_profile:
            return 0.45
        return (0.45, 0.70, 1.0)[min(self._curriculum_level(), 2)]

    def _record_episode_outcome(self, success: bool) -> None:
        if self._episode_outcome_recorded:
            return
        self._episode_outcome_recorded = True
        if not self._curriculum_enabled:
            return
        self._recent_successes.append(bool(success))
        if (self._curriculum_level_value < 2
                and len(self._recent_successes) >= CURRICULUM_MIN_EPISODES
                and float(np.mean(self._recent_successes)) >= CURRICULUM_SUCCESS_THRESHOLD):
            self._curriculum_level_value += 1
            self._recent_successes.clear()
            self.node.get_logger().info(
                f'[curriculum] passage au niveau {self._curriculum_level_value} '
                f'après réussite >= {CURRICULUM_SUCCESS_THRESHOLD:.0%}')

    def _accumulate_reward_terms(self, terms: dict[str, float]) -> None:
        self._last_reward_terms = {k: float(v) for k, v in terms.items()}
        for key, value in self._last_reward_terms.items():
            self._ep_reward_term_sums[key] = (
                self._ep_reward_term_sums.get(key, 0.0) + REWARD_SCALE * float(value))

    def _advance_waypoints(self, current_abscissa: float) -> int:
        """Advance ordered waypoints by arc length, robust to small overshoots."""
        advanced = 0
        while (self._wp_idx < self._n_wp
               and current_abscissa + WAYPOINT_ARC_TOL_M >= self._waypoint_s[self._wp_idx]):
            self._wp_idx += 1
            advanced += 1
        return advanced

    def _compute_analytical_cam(self, dot, on_wall: float) -> np.ndarray:
        """Vecteur caméra 7 valeurs depuis FK+trajectoire (ablation privileged_debug)."""
        if on_wall < 0.5 or dot is None:
            return np.zeros(7, dtype=np.float32)
        closest = closest_point_on_polyline(dot, self.waypoints,
                                            start_idx=max(self._wp_idx - 1, 0),
                                            window=ORDERED_WINDOW)
        cp  = closest['closest']
        seg = min(int(closest['segment_index']), len(self.waypoints) - 2)
        tangent = self.waypoints[seg + 1] - self.waypoints[seg]
        tn = float(np.linalg.norm(tangent))
        tangent = np.array([1.0, 0.0]) if tn < 1e-9 else tangent / tn
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        delta = np.asarray(dot, dtype=np.float64) - cp
        if self._sensor_noise and self._cam_noise_sigma_m > 0:
            delta = delta + self.np_random.normal(0.0, self._cam_noise_sigma_m, size=2)
        signed_normal = float(np.dot(delta, normal))
        offset_n = float(np.clip((signed_normal * _FX) / _SCALE_PX, -1.0, 1.0))
        theta    = float(np.arctan2(float(tangent[1]), float(tangent[0])))
        cos_t    = float(np.cos(theta))
        sin_t    = float(np.sin(theta))
        coverage = float(np.clip(1.0 - closest['distance'] / MAX_DOT_DIST, 0.0, 1.0))
        return np.array([1.0, offset_n, 1.0, cos_t, sin_t, coverage, 1.0], dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        q = self.node.joint_pos.copy()
        if self._sensor_noise and self._joint_noise_sigma_rad > 0:
            q = q + self.np_random.normal(0.0, self._joint_noise_sigma_rad, size=6)
        tcp    = fk_ur(q)
        dot = self._maybe_noisy_dot(self.node.get_laser_dot())
        on_wall = 1.0 if self._is_on_wall(dot) else 0.0
        dot_yz = np.asarray(dot if dot is not None else np.zeros(2), dtype=np.float64)
        wp_idx  = min(self._wp_idx, self._n_wp - 1)
        q_norm  = np.clip(2.0 * (q - JOINT_LIMITS_LOW) / (JOINT_LIMITS_HIGH - JOINT_LIMITS_LOW) - 1.0,
                          -1.0, 1.0)

        # Slots 12-14 : guidance réelle issue uniquement de l'image en mode real.
        # Aucun waypoint Gazebo exact n'est exposé à la politique réelle.
        if self._observation_mode == 'privileged_debug':
            guidance_xy = (
                np.clip((self.waypoints[wp_idx] - dot_yz) / MAX_DOT_DIST, -1.0, 1.0)
                if dot is not None else np.zeros(2)
            )
            guidance_progress = float(min(1.0, self._wp_idx / max(self._n_wp, 1)))
        elif self._observation_mode == 'real':
            guidance = np.asarray(getattr(self.node, 'cam_guidance', np.zeros(3)), dtype=np.float32).reshape(3)
            guidance = np.clip(np.nan_to_num(guidance, nan=0.0), -1.0, 1.0)
            guidance_xy = guidance[:2]
            guidance_progress = float(np.clip(guidance[2], 0.0, 1.0))
        else:
            guidance_xy = np.zeros(2)
            guidance_progress = 0.0

        # Camera V4 (7 valeurs). Les trois modes doivent être strictement distincts :
        #   real             -> flux ROS réel, même lorsque ligne ou laser sont absents ;
        #   privileged_debug -> caméra analytique uniquement ;
        #   zero             -> sept zéros, indépendamment des topics ROS actifs.
        real_cam = np.asarray(self.node.cam_detection, dtype=np.float32).reshape(7).copy()
        real_cam = np.nan_to_num(real_cam, nan=0.0, posinf=1.0, neginf=-1.0)
        real_cam = np.clip(real_cam, -1.0, 1.0)
        if self._observation_mode == 'real':
            cam = real_cam
        elif self._observation_mode == 'privileged_debug':
            cam = self._compute_analytical_cam(dot_yz, on_wall)
        else:  # zero
            cam = np.zeros(7, dtype=np.float32)

        ekf_pos        = self.node.ekf.position
        ekf_sigma_norm = np.clip(self.node.ekf.uncertainty / 0.05, 0.0, 1.0)
        manip_obs      = manipulability_obs(q)

        prev_wall_velocity_norm = np.clip(
            self._last_wall_velocity / max(MAX_WALL_SPEED_M_S, 1e-9), -1.0, 1.0)
        # 6+3+2+1+2+1+7+2+2+3+2+2 = 33
        obs = np.concatenate([q_norm, tcp, dot_yz, [on_wall],
                              guidance_xy, [guidance_progress],
                              cam, ekf_pos, ekf_sigma_norm, manip_obs,
                              self._last_action, prev_wall_velocity_norm]).astype(np.float32)
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)

        if self._observation_mode == 'zero':
            assert np.allclose(obs[12:15], 0.0), \
                f"Mode zero non nul: {obs[12:15]}"

        return obs

    def _save_episode(self):
        ep = self._episode_num
        np.save(self._save_dir / f'ep{ep:05d}_target.npy', self.waypoints.astype(np.float32))
        if self._ep_dot_path:
            np.save(self._save_dir / f'ep{ep:05d}_dot.npy',
                    np.asarray(self._ep_dot_path, dtype=np.float32))

    def _curriculum_level(self) -> int:
        """Difficulty is advanced by competence, never by elapsed timesteps."""
        if getattr(self, '_minimal_profile', False):
            return 0
        if not self._curriculum_enabled:
            return 2
        return int(np.clip(self._curriculum_level_value, 0, 2))

    def _guided_reset_to_line(self, max_steps: int = 40, tolerance_m: float = 0.012) -> bool:
        """Move the laser close to the first waypoint without creating RL data."""
        target = np.asarray(self.waypoints[0], dtype=np.float64)
        for _ in range(max(1, int(max_steps))):
            dot = self._maybe_noisy_dot(self.node.get_laser_dot())
            if dot is None:
                return False
            error = target - np.asarray(dot, dtype=np.float64)
            if float(np.linalg.norm(error)) <= float(tolerance_m):
                self.node.stop()
                return True
            guided_max_speed = 0.05
            desired = 2.0 * error
            desired_norm = float(np.linalg.norm(desired))
            if desired_norm > guided_max_speed:
                desired *= guided_max_speed / desired_norm
            q = self.node.joint_pos.copy()
            jvel, _ = wall_action_to_joint_velocity(
                q, desired / guided_max_speed, previous_wall_velocity=None,
                max_speed=guided_max_speed)
            for i in range(6):
                hi = ELBOW_COMMAND_LIMIT_HIGH if i == 2 else JOINT_LIMITS_HIGH[i]
                if q[i] < JOINT_LIMITS_LOW[i] + 0.1 and jvel[i] < 0.0:
                    jvel[i] = 0.0
                if q[i] > hi - 0.1 and jvel[i] > 0.0:
                    jvel[i] = 0.0
            self._pulse(jvel)
        self.node.stop()
        dot = self._maybe_noisy_dot(self.node.get_laser_dot())
        return bool(dot is not None and np.linalg.norm(target - dot) <= 2.0 * tolerance_m)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        if self._episode_num > 0:
            self._save_episode()
        self._episode_num += 1
        self._tries_on_traj += 1
        self.node.stop()

        new_traj = (self._tries_on_traj > self._trials_per_traj) or (self._episode_num == 1)

        self.node.reset_world()
        self.node.wait_for_n_steps(n_steps=25, timeout=3.0)
        start_dot = self._maybe_noisy_dot(self.node.get_laser_dot())
        if start_dot is None:
            start_dot = DEFAULT_HOME_DOT.copy()
        start_dot = np.asarray(start_dot, dtype=np.float64)

        if self._minimal_profile:
            self.waypoints = straight_line_from_start(
                start_dot, length=self._fixed_line_length)
            self._tries_on_traj = 1
            if self._episode_num == 1 and self.node._visual_enabled:
                if not self.node.show_trajectory_with_retry(self.waypoints, attempts=3, delay_s=0.5):
                    raise RuntimeError('[reset] ligne minimale non affichée dans Gazebo')
                save_current_trajectory(self.waypoints)
        elif new_traj and self._random_traj:
            stored = load_current_trajectory() if self._episode_num == 1 else None
            stored_ok = (
                stored is not None
                and np.linalg.norm(np.asarray(stored[0], dtype=float) - start_dot) < 0.08
            )
            if stored_ok:
                self.waypoints = np.asarray(stored, dtype=np.float64)
                self._tries_on_traj = 1
                self._traj_count += 1
                print(
                    f'[trajectory] dessin initial déjà présent dans Gazebo '
                    f'#{self._traj_count} | longueur={arc_length(self.waypoints):.2f} m | '
                    f'{self._trials_per_traj} essais prévus'
                )
            else:
                level = self._curriculum_level()
                candidate = curriculum_line_from_start(self._rng, start_dot, level=level)
                shown = True
                if self.node._visual_enabled:
                    shown = self.node.show_trajectory_with_retry(candidate, attempts=3, delay_s=0.5)
                if shown:
                    self.waypoints = candidate
                    save_current_trajectory(self.waypoints)
                    self._tries_on_traj = 1
                    self._traj_count += 1
                    print(
                        f'[trajectory] nouveau dessin aléatoire #{self._traj_count} | '
                        f'longueur={arc_length(self.waypoints):.2f} m | curriculum={level} | '
                        f'{self._trials_per_traj} essais prévus'
                    )
                else:
                    fallback = load_current_trajectory()
                    fallback_ok = (
                        fallback is not None
                        and np.linalg.norm(np.asarray(fallback[0], dtype=float) - start_dot) < 0.08
                    )
                    if not fallback_ok:
                        raise RuntimeError(
                            '[reset] impossible de remplacer le dessin Gazebo et '
                            'aucune trajectoire affichée valide n est disponible')
                    self.waypoints = np.asarray(fallback, dtype=np.float64)
                    self._tries_on_traj = 1
                    self.node.get_logger().warning(
                        '[reset] remplacement du dessin indisponible : conservation '
                        'du dessin précédent pour garder vision et reward cohérentes.')
        elif self._random_traj:
            print(
                f'[trajectory] dessin aléatoire #{max(self._traj_count, 1)} | '
                f'essai {self._tries_on_traj}/{self._trials_per_traj}'
            )
        else:
            # Fixed shapes are translated/scaled so waypoint 0 always coincides
            # with the actual reset laser point.
            if self._episode_num == 1:
                self.waypoints = anchor_line_to_start(self.waypoints, start_dot)
                if self.node._visual_enabled:
                    shown = self.node.show_trajectory_with_retry(
                        self.waypoints, attempts=3, delay_s=0.5)
                    if not shown:
                        raise RuntimeError('[reset] dessin fixe non affiché dans Gazebo')
                    save_current_trajectory(self.waypoints)

        self._n_wp = len(self.waypoints)
        self._path_length = arc_length(self.waypoints)
        self._waypoint_s = waypoint_abscissae(self.waypoints)
        self.node.wait_for_n_steps(n_steps=25, timeout=3.0)

        if self._guided_reset_enabled:
            dot_now = self._maybe_noisy_dot(self.node.get_laser_dot())
            if dot_now is None or np.linalg.norm(dot_now - self.waypoints[0]) > 0.015:
                if not self._guided_reset_to_line():
                    self.node.get_logger().warning(
                        '[reset] approche guidée incomplète; épisode conservé mais diagnostic recommandé.')

        if self._observation_mode == 'real' and not self._cam_health_checked:
            health = self.node.check_camera_health(probe_duration=2.0)
            if not health.camera_transport_healthy:
                self.node.get_logger().warning(
                    '[reset] Première mesure caméra instable; seconde tentative automatique.')
                health = self.node.check_camera_health(probe_duration=3.0)
            if not health.camera_transport_healthy:
                raise RuntimeError(
                    f"[reset] Transport caméra défaillant.\n"
                    f"  /line_camera  : {health.camera_hz:.1f} Hz ({health.n_camera_recv} msg)\n"
                    f"  /line_detection: {health.detection_hz:.1f} Hz ({health.n_detection_recv} msg)\n"
                    f"  age camera={health.camera_age_s:.2f}s  age detection={health.detection_age_s:.2f}s\n"
                    f"Vérifier : ros2 topic hz /line_camera /line_detection\n"
                    f"           ros2 run ur7e_line_follower line_detector"
                )
            self._cam_health_checked = True
            if not health.visual_tracking_healthy:
                self.node.get_logger().warning(
                    f"[reset] Qualité visuelle faible : "
                    f"detection={health.detection_rate*100:.0f}%  laser={health.laser_rate*100:.0f}%\n"
                    "  Vérifier visibilité ligne bleue et laser sur /line_camera."
                )

        self._step = 0
        self._ep_rewards = []
        self._ep_distances = []
        self._ep_ordered_distances = []
        self._ep_offwall_count = 0
        self._ep_dot_path = []
        self._ep_reward_term_sums = {}
        self._last_action[:] = 0.0
        self._last_wall_velocity[:] = 0.0
        self._last_jvel_pre_filter[:] = 0.0
        self._last_reward_terms = {}
        self._stagnation_steps = 0
        self._consecutive_offwall_steps = 0
        self._episode_outcome_recorded = False
        self._reset_mono = time.monotonic()

        initial_dot = self._maybe_noisy_dot(self.node.get_laser_dot())
        if initial_dot is not None:
            initial_closest = closest_point_on_polyline(initial_dot, self.waypoints)
            initial_s = float(initial_closest['abscissa'])
        else:
            initial_s = 0.0
        self._last_abscissa = initial_s
        self._max_abscissa = initial_s
        # Waypoint zero is the reset anchor, not a learned achievement.
        self._wp_idx = 1 if self._n_wp > 1 else self._n_wp

        return self._get_obs(), {
            'training_profile': self._training_profile,
            'reward_profile': self._reward_profile,
            'curriculum_level': self._curriculum_level(),
        }

    def step(self, action: np.ndarray):
        action = np.clip(
            np.asarray(action, dtype=np.float64).reshape(ACTION_SPACE_DIM), -1.0, 1.0)
        previous_action = self._last_action.copy()
        q = self.node.joint_pos.copy()
        max_wall_speed = MAX_WALL_SPEED_M_S * self._action_speed_scale()
        jvel, wall_vel = wall_action_to_joint_velocity(
            q, action, previous_wall_velocity=self._last_wall_velocity,
            max_speed=max_wall_speed)
        self._last_wall_velocity = wall_vel.copy()
        for i in range(6):
            hi = ELBOW_COMMAND_LIMIT_HIGH if i == 2 else JOINT_LIMITS_HIGH[i]
            if q[i] < JOINT_LIMITS_LOW[i] + 0.1 and jvel[i] < 0.0:
                jvel[i] = 0.0
            if q[i] > hi - 0.1 and jvel[i] > 0.0:
                jvel[i] = 0.0
        self._last_jvel_pre_filter = jvel.copy()
        physics_ok = self._pulse(jvel)
        self._step += 1
        self._total_steps += 1

        dot = self._maybe_noisy_dot(self.node.get_laser_dot())
        on_wall = self._is_on_wall(dot)
        reward = 0.0
        terminated = False
        truncated = False
        q_cur = self.node.joint_pos.copy()

        sing_pen = singularity_penalty(q_cur, w_ref=0.015, p_max=0.25)
        action_delta = action - previous_action
        action_delta_penalty = (
            -ACTION_DELTA_PENALTY_GAIN * float(np.dot(action_delta, action_delta)))
        cmd_penalty = -0.005 * float(np.linalg.norm(self._last_cmd_out) ** 2)
        reward += sing_pen + action_delta_penalty + cmd_penalty

        distance_reward = -1.0
        progress_reward = 0.0
        offwall_pen = 0.0
        wp_bonus = 0.0
        completion_bonus = 0.0
        vision_penalty = 0.0
        stale_camera_penalty = 0.0
        stagnation_penalty = 0.0
        record_bonus = 0.0
        dist_global = MAX_DOT_DIST
        dist_ordered = MAX_DOT_DIST
        current_abscissa = self._last_abscissa

        if dot is not None:
            dot_arr = np.asarray(dot, dtype=np.float64)
            self._ep_dot_path.append(dot_arr.copy())
            closest_global = closest_point_on_polyline(dot_arr, self.waypoints)
            closest_ordered = closest_point_on_polyline(
                dot_arr, self.waypoints, start_idx=max(self._wp_idx - 1, 0),
                window=ORDERED_WINDOW)
            self._last_closest = closest_ordered
            dist_global = float(closest_global['distance'])
            dist_ordered = float(closest_ordered['distance'])
            self._ep_distances.append(min(dist_global, MAX_DOT_DIST))
            self._ep_ordered_distances.append(min(dist_ordered, MAX_DOT_DIST))

            distance_reward = tracking_reward(
                dist_ordered, near_scale_m=TRACKING_NEAR_SCALE_M,
                max_distance_m=MAX_DOT_DIST, profile=self._reward_profile)
            reward += distance_reward

            current_abscissa = float(closest_ordered['abscissa'])
            delta_s = current_abscissa - float(self._last_abscissa)
            progress_reward = gated_progress_reward(
                delta_s, dist_ordered, nominal_step_m=PROGRESS_STEP_SCALE_M,
                gain=PROGRESS_REWARD_GAIN, tracking_gate_m=TRACKING_GATE_M)
            reward += progress_reward

            # Coverage is earned only while the spot is genuinely tracking the
            # ordered line.  Otherwise a distant projection could jump to the end
            # and create a false completion.
            if (dist_ordered <= TRACKING_GATE_M
                    and current_abscissa > self._max_abscissa + 0.004):
                # Coverage is already rewarded by gated_progress_reward.  Keep
                # the historical maximum for diagnostics/success, without a
                # second dense bonus that could dominate the tracking term.
                self._max_abscissa = current_abscissa

            if delta_s > STAGNATION_PROGRESS_EPS_M and dist_ordered <= TRACKING_GATE_M:
                self._stagnation_steps = 0
            else:
                self._stagnation_steps += 1
                if self._stagnation_steps > STAGNATION_GRACE_STEPS:
                    excess = self._stagnation_steps - STAGNATION_GRACE_STEPS
                    stagnation_penalty = -min(
                        STAGNATION_PENALTY_MAX, 0.01 * excess)
                    reward += stagnation_penalty

            # The previous code stored max(history) here, making delta_s <= 0 for
            # most recovery steps.  Keep previous and maximum states separate.
            self._last_abscissa = current_abscissa
            advanced = 0
            if dist_ordered <= WAYPOINT_ACCEPTANCE_M:
                advanced = self._advance_waypoints(current_abscissa)
            if advanced > 0:
                wp_bonus = WAYPOINT_BONUS * float(advanced)
                reward += wp_bonus
        else:
            self._ep_distances.append(MAX_DOT_DIST)
            self._ep_ordered_distances.append(MAX_DOT_DIST)
            self._stagnation_steps += 1

        if on_wall:
            self._consecutive_offwall_steps = 0
        else:
            self._ep_offwall_count += 1
            self._consecutive_offwall_steps += 1
            outside = distance_outside_rectangle(
                dot, y_min=WALL_Y_MIN, y_max=WALL_Y_MAX,
                z_min=WALL_Z_MIN, z_max=WALL_Z_MAX)
            offwall_pen = compute_offwall_penalty(outside)
            reward += offwall_pen
            if self._consecutive_offwall_steps >= OFFWALL_ABORT_STEPS:
                reward += OFFWALL_TERMINAL_PENALTY
                offwall_pen += OFFWALL_TERMINAL_PENALTY
                terminated = True

        if self._observation_mode == 'real':
            cam_now = np.asarray(self.node.cam_detection, dtype=np.float64)
            if cam_now[0] < 0.5 or cam_now[6] < 0.5:
                vision_penalty = -VISION_LOSS_PENALTY
                reward += vision_penalty
            if not self._fresh_detection:
                stale_camera_penalty = -0.10
                reward += stale_camera_penalty

        recent_tracking_rmse = recent_rmse(
            self._ep_ordered_distances, window=SUCCESS_WINDOW_STEPS,
            default=MAX_DOT_DIST)
        coverage = float(np.clip(
            self._max_abscissa / max(self._path_length, 1e-9), 0.0, 1.0))
        success = bool(
            coverage >= 0.98
            and recent_tracking_rmse <= self._rmse_thresh
            and on_wall
        )
        if success:
            terminated = True
            completion_bonus = COMPLETION_BONUS
            reward += completion_bonus

        stagnation_abort = self._stagnation_steps >= STAGNATION_ABORT_STEPS
        if stagnation_abort:
            terminated = True
        if self._step >= self._max_steps or not physics_ok:
            truncated = True

        timeout_flags: dict[str, bool] = {}
        if self._observation_mode == 'real':
            if not self.node.camera_transport_alive:
                self._cam_health_checked = False
                truncated = True
                timeout_flags['camera_transport_timeout'] = True
            t_now = time.monotonic()
            if t_now > self._reset_mono + _VISUAL_GRACE_S:
                line_ref = max(float(self.node._last_line_seen_mono), self._reset_mono)
                laser_ref = max(float(self.node._last_laser_seen_mono), self._reset_mono)
                if t_now - line_ref > _LINE_TIMEOUT_S:
                    truncated = True
                    timeout_flags['line_lost_timeout'] = True
                if t_now - laser_ref > _LASER_TIMEOUT_S:
                    truncated = True
                    timeout_flags['laser_lost_timeout'] = True

        self._last_action = action.copy()
        terms = {
            'dist_ordered': distance_reward,
            'sing_penalty': sing_pen,
            'cmd_penalty': cmd_penalty,
            'offwall_penalty': offwall_pen,
            'waypoint_bonus': wp_bonus,
            'completion_bonus': completion_bonus,
            'progress_reward': progress_reward,
            'vision_penalty': vision_penalty,
            'stale_camera_penalty': stale_camera_penalty,
            'action_delta_penalty': action_delta_penalty,
            'stagnation_penalty': stagnation_penalty,
            'record_bonus': record_bonus,
        }
        self._accumulate_reward_terms(terms)
        reward *= REWARD_SCALE
        self._ep_rewards.append(float(reward))

        if terminated or truncated:
            self.node.stop()
            self._record_episode_outcome(success)

        info = self._build_info(
            q_cur, dist_ordered, sing_pen, cmd_penalty, offwall_pen,
            wp_bonus, completion_bonus)
        info.update(timeout_flags)
        info['is_success'] = success
        info['recent_rmse'] = recent_tracking_rmse
        info['physics_step_timeout'] = bool(not physics_ok)
        info['fresh_camera_frame'] = bool(self._fresh_detection)
        if stagnation_abort:
            info['stagnation_timeout'] = True
        if self._consecutive_offwall_steps >= OFFWALL_ABORT_STEPS:
            info['offwall_timeout'] = True

        return self._get_obs(), float(reward), terminated, truncated, info

    def _build_info(self, q_cur, dist_ordered, sing_pen, cmd_penalty,
                    offwall_penalty, wp_bonus, completion_bonus) -> dict:
        w_wall = float(yoshikawa(q_cur, task='wall'))
        Jw = wall_jacobian(q_cur)
        ep_rmse = self._current_rmse()
        progress = float(np.clip(
            self._max_abscissa / max(self._path_length, 1e-9), 0.0, 1.0))
        current_dot = self._maybe_noisy_dot(self.node.get_laser_dot())
        success_rate = (
            float(np.mean(self._recent_successes)) if self._recent_successes else 0.0)
        camera_age = float(getattr(self.node, 'camera_age_s', float('inf')))
        detection_age = float(getattr(self.node, 'detection_age_s', float('inf')))
        return {
            'is_success': False,  # overwritten by step() after recoverable success check
            'progress': progress,
            'on_wall': self._is_on_wall(current_dot),
            'laser_path': (
                np.array(self._ep_dot_path, dtype=np.float32)
                if self._ep_dot_path else np.zeros((0, 2), dtype=np.float32)),
            'target_waypoints': self.waypoints.astype(np.float32),
            'n_waypoints_done': self._wp_idx,
            'ep_rmse': ep_rmse,
            'recent_rmse': recent_rmse(
                self._ep_ordered_distances, window=SUCCESS_WINDOW_STEPS,
                default=MAX_DOT_DIST),
            'rmse_thresh': self._rmse_thresh,
            'traj_count': self._traj_count,
            'deviation_mean': (
                float(np.mean(self._ep_distances)) if self._ep_distances else MAX_DOT_DIST),
            'deviation_max': (
                float(np.max(self._ep_distances)) if self._ep_distances else MAX_DOT_DIST),
            'ordered_deviation_mean': (
                float(np.mean(self._ep_ordered_distances))
                if self._ep_ordered_distances else MAX_DOT_DIST),
            'offwall_ratio': self._ep_offwall_count / max(self._step, 1),
            'consecutive_offwall_steps': int(self._consecutive_offwall_steps),
            'yoshikawa_w': w_wall,
            'cond_wall': jacobian_condition(Jw),
            'sing_penalty': sing_pen,
            'wall_action_y': float(self._last_action[0]),
            'wall_action_z': float(self._last_action[1]),
            'wall_velocity_norm': float(np.linalg.norm(self._last_wall_velocity)),
            'action_speed_scale': float(self._action_speed_scale()),
            'cmd_raw_norm': float(np.linalg.norm(self._last_cmd_raw)),
            'cmd_lqr_norm': float(np.linalg.norm(self._last_cmd_lqr)),
            'cmd_null_norm': float(np.linalg.norm(self._last_cmd_null)),
            'cmd_out_norm': float(np.linalg.norm(self._last_cmd_out)),
            'pulse_duration_s': float(self._last_pulse_duration_s),
            'ekf_sigma_mean': float(np.mean(self.node.ekf.uncertainty)),
            'ekf_nis': (
                float(self.node.ekf.nis) if np.isfinite(self.node.ekf.nis) else float('nan')),
            'cam_detected': float(self.node.cam_detection[0]),
            'cam_laser_visible': float(self.node.cam_detection[6]),
            'camera_age_s': camera_age,
            'detection_age_s': detection_age,
            'visual_lookahead_u': float(
                getattr(self.node, 'cam_guidance', np.zeros(3))[0]),
            'visual_lookahead_v': float(
                getattr(self.node, 'cam_guidance', np.zeros(3))[1]),
            'visual_progress': float(
                getattr(self.node, 'cam_guidance', np.zeros(3))[2]),
            'stagnation_steps': int(self._stagnation_steps),
            'curriculum_level': int(self._curriculum_level()),
            'curriculum_success_rate': success_rate,
            'training_profile': self._training_profile,
            'reward_profile': self._reward_profile,
            **{f'reward_{k}': v for k, v in self._ep_reward_term_sums.items()},
            **{f'reward_last_{k}': v for k, v in self._last_reward_terms.items()},
        }

    def _current_rmse(self) -> float:
        if not self._ep_distances:
            return MAX_DOT_DIST
        return float(np.sqrt(np.mean(np.square(self._ep_distances))))

    def close(self):
        self.node.stop()
        try:
            self.node._dot_thread_running = False
            self.node.close_visual_helpers()
        except Exception:
            pass
        try:
            self.node._spin_executor.shutdown(wait_for_completion=False)
        except Exception:
            pass
        try:
            self.node.destroy_node()
        except Exception:
            pass

    @property
    def episode_deviations(self) -> np.ndarray:
        return np.asarray(self._ep_distances, dtype=np.float64) if self._ep_distances else np.zeros(1)
