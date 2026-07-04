"""
ROS2 bridge for UR7e line-follower avec caméra statique eye-to-hand et guidance visuelle lookahead.

  - No gripper, no cube, no DetachableJoint
  - Computes laser dot on wall analytically (FK + tool Z-axis geometry)
  - Publishes the analytical laser-wall intersection for deterministic simulation overlay
  - Camera health watchdog (two-level):
      transport : /line_camera Hz + /line_detection Hz + freshness → RuntimeError si défaillant
      visual    : detection_rate, laser_rate, coverage → warning seulement au démarrage
"""
import dataclasses
import subprocess
from pathlib import Path
import threading
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import Float64MultiArray, Float32MultiArray
from sensor_msgs.msg import JointState, Image

from .kinematics import fk_ur, fk_ur_toolz, laser_wall_dot
from .ekf import LaserDotEKF
from .singularity import (yoshikawa, null_space_manip_correction,
                           lqr_velocity_correction, check_known_singularities,
                           command_filter_diagnostics)
from .trajectory_store import save_current_model_name, load_current_model_name

JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]

HOME_POSITIONS = np.array([-0.133, -1.5708, 1.5708, 0.0, 1.5708, 0.0])

WORLD_NAME = 'line_follower'
WALL_X     = 1.0

JOINT_LIMITS_LOW  = np.array([-2*np.pi, -np.pi, 0.0,      -2*np.pi, -2*np.pi, -2*np.pi])
JOINT_LIMITS_HIGH = np.array([ 2*np.pi,  0.0,   np.pi,     2*np.pi,  2*np.pi,  2*np.pi])
ELBOW_COMMAND_LIMIT_HIGH = np.deg2rad(160.0)
JOINT_COMMAND_LIMIT_RAD_S = 0.35

# ── Seuils watchdog transport (bloquants) ─────────────────────────────────────
TRANSPORT_MIN_HZ    = 6.0   # was 10.0 — Gazebo ralentit après long run
TRANSPORT_MIN_COUNT = 20
TRANSPORT_MAX_AGE_S = 0.5

# ── Seuils watchdog visuel (non bloquants au démarrage) ──────────────────────
VISUAL_MIN_DETECTION_RATE = 0.20
VISUAL_MIN_LASER_RATE     = 0.10


@dataclasses.dataclass
class CameraHealth:
    """Rapport de santé caméra produit par check_camera_health()."""
    camera_hz:                float
    detection_hz:             float
    detection_rate:           float
    laser_rate:               float
    coverage_mean:            float
    camera_age_s:             float
    detection_age_s:          float
    n_camera_recv:            int
    n_detection_recv:         int
    camera_transport_healthy: bool
    visual_tracking_healthy:  bool


class LineFollowerBridge(Node):

    def __init__(self, visual_enabled: bool = True, node_name: str = 'ur7e_line_follower_bridge'):
        super().__init__(node_name)

        self.joint_pos = HOME_POSITIONS.copy()
        self.joint_vel = np.zeros(6)
        self.step_count = 0
        self._joint_name_map: dict = {}

        # /line_detection V4 : [detected, offset_n_norm, klt_confidence,
        #                        cos_t_directed, sin_t_directed, coverage_norm, laser_visible]
        self.cam_detection = np.zeros(7, dtype=np.float32)
        # /line_guidance V4 : [lookahead_du_norm, lookahead_dv_norm, visual_progress]
        self.cam_guidance = np.zeros(3, dtype=np.float32)

        # Compteurs séparés /line_camera (brut) et /line_detection (traité)
        self._camera_frame_count:    int   = 0
        self._detection_frame_count: int   = 0
        self._last_camera_ts:        float = 0.0
        self._last_detection_ts:     float = 0.0
        self._last_camera_mono:      float = 0.0
        self._last_detection_mono:   float = 0.0

        # Timestamps des dernières détections valides (timeouts visuels en step)
        self._last_line_seen_mono:  float = 0.0
        self._last_laser_seen_mono: float = 0.0

        # Historique circulaire : (detected, laser_vis, coverage)
        self._cam_history: deque = deque(maxlen=200)

        self._yoshikawa_w   = 0.0
        self.last_cmd_raw   = np.zeros(6, dtype=np.float64)
        self.last_cmd_lqr   = np.zeros(6, dtype=np.float64)
        self.last_cmd_null  = np.zeros(6, dtype=np.float64)
        self.last_cmd_out   = np.zeros(6, dtype=np.float64)
        self.last_cmd_diag  = {}

        self.ekf = LaserDotEKF(dt=1.0 / 250.0, wall_x=WALL_X)
        self._visual_enabled = bool(visual_enabled)
        self._control_ready = False

        self.vel_pub = self.create_publisher(
            Float64MultiArray, '/forward_velocity_controller/commands', 10)
        # Simulation-only measurement used by camera_line_detector to render the
        # laser spot directly in the image. This replaces slow per-step gz service
        # calls. On the real robot this topic is simply absent and the detector
        # uses the physical red laser visible in the camera.
        self.sim_laser_pub = self.create_publisher(
            Float32MultiArray, '/sim_laser_dot_yz', 10)
        self.create_subscription(JointState, '/joint_states', self._joint_cb, 10)
        # The phase-1/minimal profile does not consume camera observations.  Avoid
        # deserialising large Image messages in that mode: they can starve the
        # joint-state callback in a SingleThreadedExecutor and create false
        # physics_step_timeout failures.
        if self._visual_enabled:
            self.create_subscription(Float32MultiArray, '/line_detection', self._cam_cb, 5)
            self.create_subscription(Float32MultiArray, '/line_guidance', self._guidance_cb, 5)
            # Raw camera is used only for transport-health monitoring.
            self.create_subscription(Image, '/line_camera', self._raw_cam_cb,
                                     qos_profile_sensor_data)

        # Executor à thread unique — spin_once() appelé depuis le thread principal uniquement.
        # Un thread background causait une contention GIL avec les gradients SAC.
        self._spin_executor = SingleThreadedExecutor()
        self._spin_executor.add_node(self)

        self._dot_thread_running = False
        self._gz_service_lock = threading.Lock()
        self._gz_error_warned = False
        from .trajectory_visual import MODEL_NAME as _TRAJ_MODEL_NAME
        self._trajectory_model_name = _TRAJ_MODEL_NAME
        # Per-step Gazebo pose updates are intentionally disabled. The laser
        # spot is projected into the camera image by camera_line_detector from
        # /sim_laser_dot_yz, so robot control and KLT never depend on gz service.
        self._dot_warned_failure_count = 0
        time.sleep(1.5)
        self.get_logger().info(
            f'LineFollowerBridge ready (cam V4 — 7+3 valeurs, visual={self._visual_enabled})')
        # Pas de thread subprocess automatique et aucun appel bloquant :
        # env.step() ne fait que planifier une mise à jour visuelle best-effort.

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _raw_cam_cb(self, msg: Image):
        """Compte les frames /line_camera brutes (Hz monitoring uniquement)."""
        self._camera_frame_count += 1
        self._last_camera_ts   = time.time()
        self._last_camera_mono = time.monotonic()

    def _cam_cb(self, msg: Float32MultiArray):
        if len(msg.data) >= 7:
            self.cam_detection[:] = np.array(msg.data[:7], dtype=np.float32)
            self._detection_frame_count += 1
            t_now  = time.time()
            t_mono = time.monotonic()
            self._last_detection_ts   = t_now
            self._last_detection_mono = t_mono
            if msg.data[0] > 0.5:
                self._last_line_seen_mono = t_mono
            if msg.data[6] > 0.5:
                self._last_laser_seen_mono = t_mono
            self._cam_history.append((
                float(msg.data[0]),   # detected
                float(msg.data[6]),   # laser_visible
                float(msg.data[5]),   # coverage
            ))

    def _guidance_cb(self, msg: Float32MultiArray):
        if len(msg.data) >= 3:
            arr = np.asarray(msg.data[:3], dtype=np.float32)
            self.cam_guidance[:] = np.clip(np.nan_to_num(arr, nan=0.0), -1.0, 1.0)

    def _joint_cb(self, msg: JointState):
        if not self._joint_name_map:
            self._joint_name_map = {name: i for i, name in enumerate(msg.name)}
        for out_idx, jname in enumerate(JOINT_NAMES):
            src_idx = self._joint_name_map.get(jname)
            if src_idx is not None:
                self.joint_pos[out_idx] = msg.position[src_idx]
                self.joint_vel[out_idx] = msg.velocity[src_idx]

        dot = laser_wall_dot(self.joint_pos, WALL_X)
        sim_msg = Float32MultiArray()
        if dot is not None:
            sim_msg.data = [float(dot[0]), float(dot[1]), 1.0]
            self.sim_laser_pub.publish(sim_msg)
            if not self.ekf.initialized:
                self.ekf.reset(float(dot[0]), float(dot[1]))
            else:
                self.ekf.predict(self.joint_pos, self.joint_vel)
                self.ekf.update_fk(float(dot[0]), float(dot[1]))
        else:
            sim_msg.data = [0.0, 0.0, 0.0]
            self.sim_laser_pub.publish(sim_msg)

        self.step_count += 1

    # ── Watchdog caméra ───────────────────────────────────────────────────────

    @property
    def camera_transport_alive(self) -> bool:
        """True si image brute et détection ont toutes deux moins d'une seconde."""
        now = time.monotonic()
        return (
            self._camera_frame_count > 0
            and self._detection_frame_count > 0
            and (now - self._last_camera_mono) < 1.0
            and (now - self._last_detection_mono) < 1.0
        )

    @property
    def cam_is_alive(self) -> bool:
        """Alias rétrocompatible de camera_transport_alive."""
        return self.camera_transport_alive

    def check_camera_health(self, probe_duration: float = 3.0) -> CameraHealth:
        """
        Probe /line_camera et /line_detection pendant probe_duration secondes.
        Doit être appelé depuis le même thread que spin_once() (pas de spin background).

        Compteurs locaux → rates calculés uniquement sur la fenêtre du probe.
        """
        camera_start    = self._camera_frame_count
        detection_start = self._detection_frame_count

        t0 = time.monotonic()
        while time.monotonic() - t0 < probe_duration:
            self._spin_executor.spin_once(timeout_sec=0.05)
        elapsed = max(time.monotonic() - t0, 1e-9)

        n_camera_recv    = self._camera_frame_count    - camera_start
        n_detection_recv = self._detection_frame_count - detection_start

        camera_hz    = n_camera_recv    / elapsed
        detection_hz = n_detection_recv / elapsed

        now_mono = time.monotonic()
        camera_age    = (now_mono - self._last_camera_mono
                         if self._camera_frame_count > 0 else 999.0)
        detection_age = (now_mono - self._last_detection_mono
                         if self._detection_frame_count > 0 else 999.0)

        # Rates sur la fenêtre probe uniquement (derniers n_detection_recv messages)
        window = list(self._cam_history)[-n_detection_recv:] if n_detection_recv > 0 else []
        if window:
            detection_rate = float(np.mean([r[0] for r in window]))
            laser_rate     = float(np.mean([r[1] for r in window]))
            det_w = [r[2] for r in window if r[0] > 0.5]
            coverage_mean  = float(np.mean(det_w)) if det_w else 0.0
        else:
            detection_rate = laser_rate = coverage_mean = 0.0

        camera_transport_healthy = (
            camera_hz    >= TRANSPORT_MIN_HZ
            and detection_hz >= TRANSPORT_MIN_HZ
            and n_camera_recv    >= TRANSPORT_MIN_COUNT
            and n_detection_recv >= TRANSPORT_MIN_COUNT
            and camera_age    < TRANSPORT_MAX_AGE_S
            and detection_age < TRANSPORT_MAX_AGE_S
        )
        visual_tracking_healthy = (
            detection_rate >= VISUAL_MIN_DETECTION_RATE
            and laser_rate >= VISUAL_MIN_LASER_RATE
        )

        return CameraHealth(
            camera_hz                = camera_hz,
            detection_hz             = detection_hz,
            detection_rate           = detection_rate,
            laser_rate               = laser_rate,
            coverage_mean            = coverage_mean,
            camera_age_s             = camera_age,
            detection_age_s          = detection_age,
            n_camera_recv            = n_camera_recv,
            n_detection_recv         = n_detection_recv,
            camera_transport_healthy = camera_transport_healthy,
            visual_tracking_healthy  = visual_tracking_healthy,
        )

    # ── Commandes ─────────────────────────────────────────────────────────────

    def publish_velocity(self, joint_vels: np.ndarray,
                         apply_singularity_correction: bool = True):
        raw = np.asarray(joint_vels, dtype=np.float64).reshape(6).copy()
        cmd = raw.copy()
        self.last_cmd_raw  = raw.copy()
        self.last_cmd_lqr  = raw.copy()
        self.last_cmd_null = np.zeros(6, dtype=np.float64)
        self.last_cmd_out  = raw.copy()
        self.last_cmd_diag = {}

        if apply_singularity_correction and np.any(np.abs(cmd) > 1e-12):
            q = self.joint_pos.copy()
            # Use the physical joint-speed bound, not the largest component of
            # the current primary command.  The latter caused the secondary
            # term to be clipped independently on every joint and corrupted the
            # requested wall direction, notably making both +z and -z move down.
            diag = command_filter_diagnostics(
                q, cmd, q_dot_max=JOINT_COMMAND_LIMIT_RAD_S)
            self.last_cmd_diag = diag
            self.last_cmd_lqr  = np.asarray(diag['lqr_cmd'],  dtype=np.float64).copy()
            self.last_cmd_null = np.asarray(diag['null_cmd'], dtype=np.float64).copy()
            cmd = np.asarray(diag['out_cmd'], dtype=np.float64).copy()

        self.last_cmd_out = cmd.copy()
        msg = Float64MultiArray()
        msg.data = cmd.tolist()
        self.vel_pub.publish(msg)
        self._yoshikawa_w = yoshikawa(self.joint_pos, task='wall')

    def stop(self):
        self.publish_velocity(np.zeros(6), apply_singularity_correction=False)

    def get_laser_dot(self):
        return laser_wall_dot(self.joint_pos, WALL_X)

    # ── Visuels ───────────────────────────────────────────────────────────────

    _FLAG_X  = WALL_X - 0.015
    _FLAG_PH = 0.22
    _FLAG_FH = 0.055
    _FLAG_FW = 0.09
    _LINE_X  = WALL_X - 0.010
    _DOT_X   = WALL_X - 0.020
    _LINE_N  = 50

    def _run_gz_service(self, service: str, reqtype: str, reptype: str,
                        req: str, timeout_ms: int = 10000,
                        process_timeout_s: float = 12.0) -> tuple[bool, str]:
        """Call a Gazebo transport service and return (success, detail)."""
        try:
            with self._gz_service_lock:
                result = subprocess.run(
                    ['gz', 'service', '-s', service,
                     '--reqtype', reqtype, '--reptype', reptype,
                     '--timeout', str(int(timeout_ms)), '--req', req],
                    check=False, capture_output=True, text=True,
                    timeout=float(process_timeout_s),
                )
        except (subprocess.TimeoutExpired, OSError) as exc:
            return False, str(exc)
        text = ((result.stdout or '') + "\n" + (result.stderr or '')).strip()
        low = text.lower()
        ok = result.returncode == 0 and 'false' not in low and 'timed out' not in low
        return bool(ok), text or f'returncode={result.returncode}'

    def _remove_trajectory_model(self, model_name: str | None = None) -> bool:
        name = self._trajectory_model_name if model_name is None else str(model_name)
        # type: 2 = MODEL — obligatoire sous Gazebo Harmonic pour que le remove
        # soit effectif. Sans ce champ, le service retourne data:true mais
        # n'efface rien (l'entité n'est pas trouvée par nom seul).
        req = f'name: "{name}" type: 2'
        ok, detail = self._run_gz_service(
            f'/world/{WORLD_NAME}/remove', 'gz.msgs.Entity', 'gz.msgs.Boolean',
            req, timeout_ms=5000, process_timeout_s=7.0)
        if not ok:
            self.get_logger().debug(f'Retrait ancien dessin non confirmé ({name}): {detail[:200]}')
        return ok

    def update_trajectory_visual(self, waypoints: np.ndarray) -> bool:
        """Remplace le dessin + drapeaux en un seul modèle Gazebo.

        On utilise toujours le nom fixe TRAJECTORY_MODEL_NAME :
          1. Supprimer l'ancien (type:2 obligatoire sous Gazebo Harmonic).
          2. Créer le nouveau avec le même nom.
        Aucun nom temporaire → jamais deux modèles simultanés.
        """
        if not self._visual_enabled:
            return False
        from .trajectory_visual import write_trajectory_model, MODEL_NAME
        runtime_file = Path(f'/tmp/{MODEL_NAME}.sdf')
        write_trajectory_model(runtime_file, waypoints, model_name=MODEL_NAME)

        # Étape 1 : supprimer l'ancien dessin (best-effort — ok si absent).
        self._run_gz_service(
            f'/world/{WORLD_NAME}/remove', 'gz.msgs.Entity', 'gz.msgs.Boolean',
            f'name: "{MODEL_NAME}" type: 2', timeout_ms=5000, process_timeout_s=7.0)

        # Étape 2 : créer le nouveau dessin.
        req = (f'sdf_filename: "{runtime_file}" '
               f'name: "{MODEL_NAME}" allow_renaming: false')
        created, detail = self._run_gz_service(
            f'/world/{WORLD_NAME}/create', 'gz.msgs.EntityFactory',
            'gz.msgs.Boolean', req, timeout_ms=10000, process_timeout_s=12.0)
        if not created:
            self.get_logger().warning(
                f'Création drapeaux + dessin refusée par Gazebo: {detail[:300]}')
            return False

        self._trajectory_model_name = MODEL_NAME
        self._gz_error_warned = False
        return True

    def show_trajectory_with_retry(self, waypoints: np.ndarray, attempts: int = 3,
                                   delay_s: float = 0.5) -> bool:
        for attempt in range(max(1, int(attempts))):
            if self.update_trajectory_visual(waypoints):
                return True
            if attempt + 1 < attempts:
                time.sleep(max(0.1, float(delay_s)))
        return False

    # Legacy helpers kept for unit diagnostics only.
    def _flag_poses(self, waypoints: np.ndarray) -> list:
        x = self._FLAG_X
        ph, fh, fw = self._FLAG_PH, self._FLAG_FH, self._FLAG_FW
        result = []
        for (yw, zw), suffix in ((waypoints[0], 'start'), (waypoints[-1], 'end')):
            yw, zw = float(yw), float(zw)
            result.append((f'pole_{suffix}',  x, yw,             zw + ph / 2,         0., 0., 0., 1.))
            result.append((f'cloth_{suffix}', x, yw + fw/2+0.006, zw + ph - fh/2,    0., 0., 0., 1.))
        return result

    def _line_poses(self, waypoints: np.ndarray) -> list:
        lx = self._LINE_X
        wp = np.asarray(waypoints, dtype=float)
        diffs  = np.diff(wp, axis=0)
        dists  = np.linalg.norm(diffs, axis=1)
        cumlen = np.concatenate([[0.], np.cumsum(dists)])
        total  = cumlen[-1]
        ts_src = cumlen / (total + 1e-9)
        ts_dst = np.linspace(0., 1., self._LINE_N)
        ys = np.interp(ts_dst, ts_src, wp[:, 0])
        zs = np.interp(ts_dst, ts_src, wp[:, 1])
        return [(f'sph_{i:03d}', lx, float(ys[i]), float(zs[i]), 0., 0., 0., 1.)
                for i in range(self._LINE_N)]

    def start_laser_dot_thread(self, rate_hz: float = 15.0):
        """Compatibilité : boucle bornée utilisant la méthode synchronisée.

        Cette méthode n'est plus appelée automatiquement. Elle reste disponible
        pour les diagnostics, sans dupliquer l'appel subprocess.
        """
        if self._dot_thread_running:
            return
        self._dot_thread_running = True
        period = 1.0 / max(float(rate_hz), 1.0)

        def _dot_loop():
            while self._dot_thread_running and self._visual_enabled:
                cycle_start = time.monotonic()
                self.update_laser_dot_visual()
                remaining = period - (time.monotonic() - cycle_start)
                if remaining > 0.0:
                    time.sleep(remaining)

        threading.Thread(target=_dot_loop, daemon=True, name='laser-dot-visual').start()

    def update_laser_dot_visual(self) -> bool:
        """Mémorise la position FK courante — le thread laser_dot l'envoie à 3 Hz."""
        if not self._visual_enabled:
            return False
        from .kinematics import laser_wall_dot
        yz = laser_wall_dot(self.joint_pos)
        if yz is None:
            return False
        self._dot_yz = (float(yz[0]), float(yz[1]))
        return True

    def start_laser_dot_thread(self, rate_hz: float = 3.0):
        """Lance un thread qui envoie set_pose à Gazebo à rate_hz Hz."""
        if getattr(self, '_dot_thread_running', False):
            return
        self._dot_yz: tuple | None = None
        self._dot_thread_running = True

        def _loop():
            period = 1.0 / max(rate_hz, 0.5)
            while self._dot_thread_running:
                t0 = time.monotonic()
                yz = getattr(self, '_dot_yz', None)
                if yz is not None:
                    y, z = yz
                    req = (f'name: "laser_dot" '
                           f'position {{ x: 0.980 y: {y:.4f} z: {z:.4f} }} '
                           f'orientation {{ w: 1.0 }}')
                    try:
                        subprocess.run(
                            ['gz', 'service', '-s', f'/world/{WORLD_NAME}/set_pose',
                             '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean',
                             '--timeout', '800', '--req', req],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            timeout=1.5,
                        )
                    except Exception:
                        pass
                elapsed = time.monotonic() - t0
                remaining = period - elapsed
                if remaining > 0:
                    time.sleep(remaining)

        threading.Thread(target=_loop, daemon=True, name='laser-dot-visual').start()

    def stop_laser_dot_thread(self):
        self._dot_thread_running = False

    def close_visual_helpers(self) -> None:
        """No per-step Gazebo helper remains in V2.3.1."""
        return None

    def ensure_control_ready(self, timeout: float = 5.0) -> bool:
        """Wait once for joint states and the velocity-controller subscriber.

        ROS discovery can drop the first non-latched velocity command when a
        diagnostic process starts immediately.  This appeared as a false +y
        axis failure because +y is the first direction tested.
        """
        if self._control_ready:
            return True
        deadline = time.monotonic() + float(timeout)
        start_steps = int(self.step_count)
        while time.monotonic() < deadline:
            self._spin_executor.spin_once(timeout_sec=0.02)
            have_joint_states = self.step_count > start_steps or self.step_count > 0
            have_controller = self.vel_pub.get_subscription_count() > 0
            if have_joint_states and have_controller:
                self.stop()
                self._control_ready = True
                return True
        self.get_logger().warning(
            '[control] contrôleur/joint_states non confirmés avant timeout; '
            'les premières commandes peuvent être perdues.')
        return False

    def go_home(self, timeout: float = 3.0):
        KP = 3.0
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._spin_executor.spin_once(timeout_sec=0.004)
            err = HOME_POSITIONS - self.joint_pos
            if np.max(np.abs(err)) < 0.05:
                break
            vel = np.clip(KP * err, -2.0, 2.0)
            self.publish_velocity(vel, apply_singularity_correction=False)
        self.publish_velocity(np.zeros(6), apply_singularity_correction=False)

    def reset_world(self):
        self.ensure_control_ready(timeout=5.0)
        self.stop()
        self.go_home()
        self.joint_vel  = np.zeros(6)
        self.step_count = 0
        dot = laser_wall_dot(self.joint_pos, WALL_X)
        if dot is not None:
            self.ekf.reset(float(dot[0]), float(dot[1]))
        else:
            self.ekf.reset(0.0, 0.67)

    def drain_callbacks(self, max_cycles: int = 20) -> None:
        """Process queued ROS callbacks while the robot is stopped.

        The executor is intentionally single-threaded.  Draining before an action
        prevents an old queued detection from being mistaken for a post-action
        camera frame by the frame-count gate.
        """
        for _ in range(max(0, int(max_cycles))):
            self._spin_executor.spin_once(timeout_sec=0.0)

    def wait_for_n_steps(self, n_steps: int = 10, timeout: float = 3.0) -> bool:
        """Spin until ``n_steps`` fresh joint-state samples have arrived.

        The deadline is wall-clock only as a watchdog; the commanded duration is
        still defined by the number of joint-state samples.  A generous timeout
        therefore preserves deterministic simulated duration while tolerating a
        temporarily low Gazebo real-time factor.
        """
        target   = self.step_count + int(n_steps)
        deadline = time.monotonic() + float(timeout)
        while self.step_count < target:
            if time.monotonic() > deadline:
                return False
            self._spin_executor.spin_once(timeout_sec=0.01)
            # Drain a bounded number of already-ready callbacks so image and
            # guidance traffic cannot indefinitely sit ahead of joint_states.
            for _ in range(8):
                if self.step_count >= target:
                    break
                self._spin_executor.spin_once(timeout_sec=0.0)
        return True

    def wait_for_detection_after(self, frame_count: int, timeout: float = 0.20) -> bool:
        """Wait for a processed camera frame newer than ``frame_count``.

        This prevents the environment from pairing an action with a stale KLT
        observation.  It is best-effort: callers can log/penalise a timeout
        without blocking indefinitely.
        """
        deadline = time.monotonic() + float(timeout)
        while self._detection_frame_count <= int(frame_count):
            if time.monotonic() > deadline:
                return False
            self._spin_executor.spin_once(timeout_sec=0.004)
        return True

    @property
    def camera_age_s(self) -> float:
        if self._last_camera_mono <= 0.0:
            return float('inf')
        return max(0.0, time.monotonic() - self._last_camera_mono)

    @property
    def detection_age_s(self) -> float:
        if self._last_detection_mono <= 0.0:
            return float('inf')
        return max(0.0, time.monotonic() - self._last_detection_mono)
