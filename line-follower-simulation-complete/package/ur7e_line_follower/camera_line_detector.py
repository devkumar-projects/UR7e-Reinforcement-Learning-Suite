"""
camera_line_detector.py — Détection et suivi de la ligne bleue par caméra statique eye-to-hand.

Pipeline :
  1. Conversion ROS Image → numpy (sans cv_bridge, compatible NumPy 2.x)
  2. Resize à (W, H) = (320, 240) dans process_frame()
  3. Filtre couleur HSV → masque bleu (ligne) + masque rouge (laser)
  4. KLT (Lucas-Kanade) pour tracker les points de la ligne entre frames
  5. Ré-initialisation automatique si trop peu de points trackés
  6. Offset normal signé laser↔ligne, normalisé par SCALE_PX
  7. Orientation de la tangente vers le drapeau vert + EMA unitaire

Topics :
  Abonnement : /line_camera (Image), /sim_laser_dot_yz en simulation uniquement
  Publication : /line_detection (std_msgs/Float32MultiArray) — schéma V4
  Publication : /line_guidance  (std_msgs/Float32MultiArray) — guidance visuelle 3D
    data[0] : line_detected      (0.0 ou 1.0)
    data[1] : offset_n_norm      (écart normal signé, [-1,1], normalisé par SCALE_PX)
    data[2] : klt_confidence     (feature_ratio × retention_ratio, [0,1])
    data[3] : tangent_cos_theta  (tangente orientée vers le drapeau vert, [-1,1])
    data[4] : tangent_sin_theta  (tangente orientée vers le drapeau vert, [-1,1])
    data[5] : coverage_norm      (pixels bleus normalisés, [0,1])
    data[6] : laser_visible      (1.0 si point rouge détecté)

OBSERVATION_SCHEMA_VERSION = 4
"""

import dataclasses
import math
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

# ── Schéma ───────────────────────────────────────────────────────────────────
OBSERVATION_SCHEMA_VERSION: int = 4

# ── Paramètres couleur ────────────────────────────────────────────────────────
BLUE_HSV_LO = np.array([100,  60,  40], dtype=np.uint8)
BLUE_HSV_HI = np.array([140, 255, 255], dtype=np.uint8)

RED_HSV_LO1 = np.array([  0, 160, 160], dtype=np.uint8)
RED_HSV_HI1 = np.array([ 12, 255, 255], dtype=np.uint8)
RED_HSV_LO2 = np.array([165, 160, 160], dtype=np.uint8)
RED_HSV_HI2 = np.array([180, 255, 255], dtype=np.uint8)

GREEN_HSV_LO = np.array([ 40,  70,  40], dtype=np.uint8)
GREEN_HSV_HI = np.array([ 90, 255, 255], dtype=np.uint8)

# ── Paramètres KLT ───────────────────────────────────────────────────────────
LK_PARAMS = dict(
    winSize=(13, 13),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.02),
)
MIN_TRACK_POINTS = 6
MAX_KLT_POINTS   = 80
CONFIDENCE_FULL_POINTS = 30   # 30 points stables suffisent pour une confiance proche de 1
REINIT_INTERVAL  = 25

# ── Image ────────────────────────────────────────────────────────────────────
W, H = 320, 240
SCALE_PX: float = math.sqrt(W * W + H * H) / 2.0   # = 200.0 px

# ── EMA orientation ──────────────────────────────────────────────────────────
_EMA_ALPHA: float = 0.30   # poids de la nouvelle mesure dans l'EMA
LOOKAHEAD_PX: float = 28.0
LOOKAHEAD_NORM_PX: float = 60.0

# Static eye-to-hand camera geometry used only for the Gazebo laser overlay.
# Gazebo camera optical convention: local +X forward, +Y left, +Z up.
SIM_CAMERA_POSITION = np.array([-0.20, -1.30, 1.30], dtype=np.float64)
SIM_CAMERA_RPY = np.array([0.0, 0.30, 0.83], dtype=np.float64)
SIM_CAMERA_HFOV = 1.3090
SIM_LASER_X = 0.980
SIM_LASER_MAX_AGE_S = 0.50


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


SIM_CAMERA_ROTATION = _rpy_matrix(*SIM_CAMERA_RPY)


def project_sim_laser_to_pixel(y: float, z: float, width: int, height: int) -> "np.ndarray | None":
    """Project the analytical wall point into the static Gazebo camera image."""
    point_world = np.array([SIM_LASER_X, float(y), float(z)], dtype=np.float64)
    point_cam = SIM_CAMERA_ROTATION.T @ (point_world - SIM_CAMERA_POSITION)
    depth = float(point_cam[0])
    if depth <= 1e-6:
        return None
    focal = (float(width) * 0.5) / math.tan(SIM_CAMERA_HFOV * 0.5)
    # Local +Y points to image left and local +Z points upward.
    u = float(width) * 0.5 - focal * float(point_cam[1]) / depth
    v = float(height) * 0.5 - focal * float(point_cam[2]) / depth
    if not (0.0 <= u < float(width) and 0.0 <= v < float(height)):
        return None
    return np.array([u, v], dtype=np.float32)


@dataclasses.dataclass
class ProcessResult:
    """Résultat structuré de process_frame(). Valeurs brutes + normalisées."""
    line_detected:  bool
    blue_px:        int
    line_coverage:  float
    laser_uv:       "np.ndarray | None"
    laser_visible:  bool
    line_pts:       "np.ndarray | None"
    offset_n_px:    float    # erreur normale signée en pixels (→ detection_vector)
    offset_l_px:    float    # erreur longitudinale en pixels (diagnostic — instable)
    angle_deg:      float    # angle brut avant EMA (diagnostic)
    offset_n_norm:  float    # offset_n_px / SCALE_PX, clampé [-1, 1]
    klt_confidence: float    # feature_ratio × retention_ratio [0, 1]
    tangent_cos_t:  float    # EMA cos(θ) orientée vers le drapeau vert [-1, 1]
    tangent_sin_t:  float    # EMA sin(θ) orientée vers le drapeau vert [-1, 1]
    coverage_norm:  float    # pixels bleus normalisés [0, 1]
    lookahead_du_norm: float   # vecteur image laser -> point futur, axe u
    lookahead_dv_norm: float   # vecteur image laser -> point futur, axe v
    visual_progress:   float   # progression estimée entre début visuel et drapeau vert

    @property
    def detection_vector(self) -> list:
        """Vecteur /line_detection V4 — 7 valeurs.
        offset_l_px intentionnellement exclu : instable selon le sous-ensemble KLT actif.
        """
        return [
            float(self.line_detected),
            float(self.offset_n_norm),
            float(self.klt_confidence),
            float(self.tangent_cos_t),
            float(self.tangent_sin_t),
            float(self.coverage_norm),
            float(self.laser_visible),
        ]

    @property
    def guidance_vector(self) -> list:
        return [
            float(self.lookahead_du_norm),
            float(self.lookahead_dv_norm),
            float(self.visual_progress),
        ]


def _imgmsg_to_bgr(msg: Image) -> np.ndarray:
    """Conversion ROS Image → BGR numpy sans cv_bridge (compatible NumPy 2.x)."""
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8).copy()
    enc  = msg.encoding.lower()
    if enc in ('bgr8', 'rgb8', '8uc3'):
        img = data.reshape(msg.height, msg.width, 3)
        if enc == 'rgb8':
            img = img[:, :, ::-1].copy()
        return img
    if enc in ('mono8', '8uc1'):
        img = data.reshape(msg.height, msg.width)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = data.reshape(msg.height, msg.width, -1)
    return img


class CameraLineDetector(Node):

    def __init__(self):
        super().__init__('camera_line_detector')
        self._pub = self.create_publisher(Float32MultiArray, '/line_detection', 10)
        self._guidance_pub = self.create_publisher(Float32MultiArray, '/line_guidance', 10)
        self.create_subscription(Image, '/line_camera', self._img_cb,
                                 qos_profile_sensor_data)
        self.declare_parameter('use_sim_laser_overlay', False)
        self._use_sim_laser_overlay = bool(
            self.get_parameter('use_sim_laser_overlay').value)
        self._sim_laser_yz = None
        self._sim_laser_last_mono = 0.0
        if self._use_sim_laser_overlay:
            self.create_subscription(
                Float32MultiArray, '/sim_laser_dot_yz', self._sim_laser_cb, 10)
        self._init_state()
        self.get_logger().info(
            f'CameraLineDetector prêt → /line_detection + /line_guidance (schéma V{OBSERVATION_SCHEMA_VERSION})')

    # ── État interne ──────────────────────────────────────────────────────────

    def _init_state(self) -> None:
        """Initialise ou réinitialise tout l'état KLT + EMA."""
        self._prev_gray:    "np.ndarray | None" = None
        self._prev_pts:     "np.ndarray | None" = None
        self._frame_idx:    int   = 0
        self._n_init_pts:   int   = 0
        self._ema_c: "float | None" = None   # None → init directe sans biais à 0°
        self._ema_s: "float | None" = None
        self._prev_tangent: "np.ndarray | None" = None
        self._prev_laser_uv: "np.ndarray | None" = None

    # ── Simulation laser overlay ─────────────────────────────────────────────

    def _sim_laser_cb(self, msg: Float32MultiArray):
        if len(msg.data) >= 3 and float(msg.data[2]) > 0.5:
            self._sim_laser_yz = np.array(msg.data[:2], dtype=np.float64)
            self._sim_laser_last_mono = time.monotonic()
        else:
            self._sim_laser_yz = None
            self._sim_laser_last_mono = 0.0

    def _overlay_sim_laser(self, frame: np.ndarray) -> np.ndarray:
        """Draw the simulated red spot into the camera frame without Gazebo IPC."""
        if not bool(getattr(self, '_use_sim_laser_overlay', False)):
            return frame
        yz = getattr(self, '_sim_laser_yz', None)
        age = time.monotonic() - float(getattr(self, '_sim_laser_last_mono', 0.0))
        if yz is None or age > SIM_LASER_MAX_AGE_S:
            return frame
        uv = project_sim_laser_to_pixel(float(yz[0]), float(yz[1]),
                                        int(frame.shape[1]), int(frame.shape[0]))
        if uv is None:
            return frame
        out = frame.copy()
        radius = max(5, int(round(frame.shape[1] / 64.0)))
        cv2.circle(out, (int(round(uv[0])), int(round(uv[1]))),
                   radius, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        return out

    # ── Callback image (wrapper léger) ────────────────────────────────────────

    def _img_cb(self, msg: Image):
        """Conversion ROS→numpy puis délègue entièrement à process_frame()."""
        try:
            frame = _imgmsg_to_bgr(msg)
        except Exception as e:
            self.get_logger().warning(f'img decode: {e}')
            return
        frame = self._overlay_sim_laser(frame)
        # Le resize est dans process_frame() pour que tests et ROS utilisent le même pipeline
        result = self.process_frame(frame)
        out = Float32MultiArray()
        out.data = result.detection_vector
        self._pub.publish(out)
        guidance = Float32MultiArray()
        guidance.data = result.guidance_vector
        self._guidance_pub.publish(guidance)

    # ── Pipeline principal (public, stateful) ─────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> ProcessResult:
        """
        Traite une frame BGR de la caméra statique (taille quelconque → resizée à W×H).
        Mute l'état interne : _prev_gray, _prev_pts, _n_init_pts,
                               _ema_c, _ema_s, _prev_tangent, _frame_idx.
        """
        if frame.shape[1] != W or frame.shape[0] != H:
            frame = cv2.resize(frame, (W, H))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        blue_mask = cv2.inRange(hsv, BLUE_HSV_LO, BLUE_HSV_HI)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kern)
        blue_mask = cv2.morphologyEx(
            blue_mask, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        blue_mask = cv2.dilate(blue_mask, kern)
        blue_mask = self._select_line_component(blue_mask)

        red1 = cv2.inRange(hsv, RED_HSV_LO1, RED_HSV_HI1)
        red2 = cv2.inRange(hsv, RED_HSV_LO2, RED_HSV_HI2)
        red_mask = cv2.bitwise_or(red1, red2)
        green_mask = cv2.inRange(hsv, GREEN_HSV_LO, GREEN_HSV_HI)

        blue_px       = int(np.count_nonzero(blue_mask))
        line_coverage = float(blue_px) / (W * H)
        line_detected = blue_px > 40
        coverage_norm = float(np.clip(line_coverage * 20.0, 0.0, 1.0))

        laser_uv, laser_visible = self._find_laser(red_mask)
        goal_uv, _goal_visible = self._find_goal(green_mask)
        line_pts, klt_confidence = self._update_klt(gray, blue_mask, line_detected)

        offset_n_px   = 0.0
        offset_l_px   = 0.0
        angle_deg     = 0.0
        offset_n_norm = 0.0
        lookahead_du_norm = 0.0
        lookahead_dv_norm = 0.0
        visual_progress = 0.0
        pts2d = None

        if line_detected and laser_visible and laser_uv is not None:
            if line_pts is not None and len(line_pts) >= 4:
                pts2d = line_pts.reshape(-1, 2)
            else:
                ys, xs = np.where(blue_mask > 0)
                if len(xs) >= 4:
                    step = max(1, len(xs) // 60)
                    pts2d = np.stack([xs[::step], ys[::step]], axis=1).astype(np.float32)

            if pts2d is not None and len(pts2d) >= 4:
                offset_n_px, offset_l_px, angle_deg = self._compute_offset(laser_uv, pts2d, goal_uv=goal_uv)
                offset_n_norm = float(np.clip(offset_n_px / SCALE_PX, -1.0, 1.0))

                theta = math.radians(angle_deg)
                c = math.cos(theta)
                s = math.sin(theta)
                if self._ema_c is None:
                    self._ema_c, self._ema_s = c, s
                else:
                    self._ema_c = (1.0 - _EMA_ALPHA) * self._ema_c + _EMA_ALPHA * c
                    self._ema_s = (1.0 - _EMA_ALPHA) * self._ema_s + _EMA_ALPHA * s
                ema_norm = math.hypot(self._ema_c, self._ema_s)
                if ema_norm > 1e-9:
                    self._ema_c /= ema_norm
                    self._ema_s /= ema_norm

        cos_t = self._ema_c if self._ema_c is not None else 0.0
        sin_t = self._ema_s if self._ema_s is not None else 0.0
        if (line_detected and laser_visible and laser_uv is not None
                and pts2d is not None and len(pts2d) >= 4
                and math.hypot(cos_t, sin_t) > 1e-6):
            lookahead_du_norm, lookahead_dv_norm, visual_progress = self._compute_guidance(
                laser_uv, pts2d, np.array([cos_t, sin_t], dtype=np.float32), goal_uv)

        self._frame_idx += 1

        return ProcessResult(
            line_detected  = line_detected,
            blue_px        = blue_px,
            line_coverage  = line_coverage,
            laser_uv       = laser_uv,
            laser_visible  = laser_visible,
            line_pts       = line_pts,
            offset_n_px    = offset_n_px,
            offset_l_px    = offset_l_px,
            angle_deg      = angle_deg,
            offset_n_norm  = offset_n_norm,
            klt_confidence = klt_confidence,
            tangent_cos_t  = float(np.clip(cos_t, -1.0, 1.0)),
            tangent_sin_t  = float(np.clip(sin_t, -1.0, 1.0)),
            coverage_norm  = coverage_norm,
            lookahead_du_norm = lookahead_du_norm,
            lookahead_dv_norm = lookahead_dv_norm,
            visual_progress = visual_progress,
        )

    # ── KLT ──────────────────────────────────────────────────────────────────

    def _update_klt(self, gray: np.ndarray, blue_mask: np.ndarray,
                    line_detected: bool) -> "tuple[np.ndarray | None, float]":
        """
        Retourne (line_pts, klt_confidence).

        Réinit réussie : confidence = min(n_pts / CONFIDENCE_FULL_POINTS, 1.0)
        Réinit échouée : confidence = 0.0, état réinitialisé
        Suivi          : confidence = clip(feature_ratio × retention_ratio, 0, 1)
          density_ratio   = n_tracked / CONFIDENCE_FULL_POINTS
          retention_ratio = n_tracked / n_init_pts
          confidence      = density_ratio × (0.5 + 0.5 × retention_ratio)
        """
        need_reinit = (
            self._prev_gray is None
            or self._prev_pts is None
            or len(self._prev_pts) < MIN_TRACK_POINTS
            or self._frame_idx % REINIT_INTERVAL == 0
        )

        if line_detected and need_reinit:
            pts = self._sample_line_points(blue_mask)
            if pts is None or len(pts) < MIN_TRACK_POINTS:
                self._prev_pts   = None
                self._n_init_pts = 0
                return None, 0.0
            self._prev_pts   = pts
            self._prev_gray  = gray.copy()
            self._n_init_pts = len(pts)
            return pts, float(np.clip(len(pts) / CONFIDENCE_FULL_POINTS, 0.0, 1.0))

        if (self._prev_gray is not None
                and self._prev_pts is not None
                and len(self._prev_pts) >= MIN_TRACK_POINTS):
            tracked = self._klt_track(self._prev_gray, gray, self._prev_pts, blue_mask)
            self._prev_gray = gray.copy()
            if tracked is not None and len(tracked) >= MIN_TRACK_POINTS:
                self._prev_pts = tracked
                density_ratio   = min(float(len(tracked)) / CONFIDENCE_FULL_POINTS, 1.0)
                retention_ratio = min(float(len(tracked)) / max(self._n_init_pts, 1), 1.0)
                confidence = density_ratio * (0.5 + 0.5 * retention_ratio)
                return tracked, float(np.clip(confidence, 0.0, 1.0))
            self._prev_pts   = None
            self._n_init_pts = 0
            return None, 0.0

        self._prev_gray = gray.copy()
        return None, 0.0

    def _klt_track(self, prev_gray: np.ndarray, gray: np.ndarray,
                   prev_pts: np.ndarray, blue_mask: np.ndarray) -> "np.ndarray | None":
        pts_cv = prev_pts.reshape(-1, 1, 2).astype(np.float32)
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, pts_cv, None, **LK_PARAMS)
        if next_pts is None or status is None:
            return None
        good = [
            next_pts[i, 0]
            for i, s in enumerate(status)
            if s[0] == 1
            and 0 <= int(next_pts[i, 0, 0]) < W
            and 0 <= int(next_pts[i, 0, 1]) < H
            and blue_mask[int(next_pts[i, 0, 1]), int(next_pts[i, 0, 0])] > 0
        ]
        if len(good) < MIN_TRACK_POINTS:
            return None
        return np.array(good, dtype=np.float32)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _select_line_component(self, blue_mask: np.ndarray) -> np.ndarray:
        """Conserve la composante bleue la plus compatible avec une longue ligne.

        La caméra eye-to-hand peut voir des pièces bleues du robot. Une ligne cible
        occupe une grande diagonale avec un faible taux de remplissage, contrairement
        à un capot compact.
        """
        n, labels, stats, _ = cv2.connectedComponentsWithStats(blue_mask, connectivity=8)
        best_label = 0
        best_score = -1.0
        for label in range(1, n):
            x, y, w, h, area = stats[label]
            area = float(area)
            if area < 30.0:
                continue
            diag = math.hypot(float(w), float(h))
            fill = area / max(float(w * h), 1.0)
            elongation = max(float(w), float(h)) / max(min(float(w), float(h)), 1.0)
            score = diag * (1.6 - min(fill, 1.0)) + 4.0 * min(elongation, 8.0)
            if score > best_score:
                best_score = score
                best_label = label
        if best_label == 0:
            return np.zeros_like(blue_mask)
        out = np.zeros_like(blue_mask)
        out[labels == best_label] = 255
        return out

    def _find_laser(self, red_mask: np.ndarray):
        """Sélectionne le spot laser circulaire sans confondre le drapeau rouge.

        Avec la caméra statique, le drapeau de départ rouge est toujours visible.
        On classe donc les composantes rouges par circularité, taille plausible et
        continuité temporelle, au lieu de prendre aveuglément la plus grande aire.
        """
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < 4.0 or area > 1200.0:
                continue
            perimeter = float(cv2.arcLength(c, True))
            if perimeter <= 1e-9:
                continue
            circularity = float(4.0 * math.pi * area / (perimeter * perimeter))
            M = cv2.moments(c)
            if M['m00'] == 0:
                continue
            uv = np.array([M['m10'] / M['m00'], M['m01'] / M['m00']], dtype=np.float32)
            x, y, w, h = cv2.boundingRect(c)
            aspect = min(w, h) / max(w, h, 1)
            # Le spot est petit, compact et presque circulaire.
            score = 2.0 * circularity + 0.7 * aspect - 0.0007 * abs(area - 60.0)
            if self._prev_laser_uv is not None:
                score -= 0.012 * float(np.linalg.norm(uv - self._prev_laser_uv))
            candidates.append((score, uv))
        if not candidates:
            self._prev_laser_uv = None
            return None, False
        _, uv = max(candidates, key=lambda item: item[0])
        self._prev_laser_uv = uv.copy()
        return uv, True


    def _find_goal(self, green_mask: np.ndarray):
        """Centroïde du drapeau vert d'arrivée, visible par la caméra fixe."""
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < 12.0 or area > 5000.0:
                continue
            M = cv2.moments(c)
            if M['m00'] <= 1e-9:
                continue
            uv = np.array([M['m10'] / M['m00'], M['m01'] / M['m00']], dtype=np.float32)
            if best is None or area > best[0]:
                best = (area, uv)
        return (best[1], True) if best is not None else (None, False)

    def _sample_line_points(self, blue_mask: np.ndarray) -> "np.ndarray | None":
        pts = cv2.goodFeaturesToTrack(
            blue_mask,
            maxCorners=MAX_KLT_POINTS,
            qualityLevel=0.01,
            minDistance=5,
            blockSize=7,
            useHarrisDetector=False,
        )
        if pts is not None and len(pts) >= MIN_TRACK_POINTS:
            return pts.reshape(-1, 2).astype(np.float32)
        ys, xs = np.where(blue_mask > 0)
        if len(xs) < MIN_TRACK_POINTS:
            return None
        idx = np.linspace(0, len(xs) - 1, min(MAX_KLT_POINTS, len(xs)), dtype=int)
        return np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)

    def _compute_guidance(self, laser_uv: np.ndarray, line_pts: np.ndarray,
                          tangent: np.ndarray,
                          goal_uv: "np.ndarray | None") -> "tuple[float, float, float]":
        """Guidage visuel réel, sans waypoint Gazebo privilégié.

        Le point lookahead est choisi parmi les points bleus situés devant le
        laser selon la tangente locale orientée vers le drapeau vert. La
        progression est une estimation image entre l'extrémité opposée au
        drapeau vert et le drapeau vert.
        """
        laser = np.asarray(laser_uv, dtype=np.float32).reshape(2)
        pts = np.asarray(line_pts, dtype=np.float32).reshape(-1, 2)
        t = np.asarray(tangent, dtype=np.float32).reshape(2)
        tn = float(np.linalg.norm(t))
        if len(pts) < 2 or tn < 1e-8:
            return 0.0, 0.0, 0.0
        t = t / tn
        n = np.array([-t[1], t[0]], dtype=np.float32)
        delta = pts - laser
        forward = delta @ t
        lateral = np.abs(delta @ n)
        valid = np.where(forward > 2.0)[0]
        target = None
        if len(valid):
            score = np.abs(forward[valid] - LOOKAHEAD_PX) + 0.35 * lateral[valid]
            target = pts[valid[int(np.argmin(score))]]
        elif goal_uv is not None:
            target = np.asarray(goal_uv, dtype=np.float32).reshape(2)
        if target is None:
            return 0.0, 0.0, 0.0
        guide = target - laser
        du = float(np.clip(guide[0] / LOOKAHEAD_NORM_PX, -1.0, 1.0))
        dv = float(np.clip(guide[1] / LOOKAHEAD_NORM_PX, -1.0, 1.0))

        progress = 0.0
        if goal_uv is not None:
            goal = np.asarray(goal_uv, dtype=np.float32).reshape(2)
            start = pts[int(np.argmax(np.linalg.norm(pts - goal, axis=1)))]
            axis = goal - start
            den = float(np.dot(axis, axis))
            if den > 1e-6:
                progress = float(np.clip(np.dot(laser - start, axis) / den, 0.0, 1.0))
        return du, dv, progress


    def _compute_offset(self, laser_uv: np.ndarray,
                        line_pts: np.ndarray,
                        goal_uv: "np.ndarray | None" = None) -> "tuple[float, float, float]":
        """
        Distance signée laser → ligne + angle tangentiel.
        Renvoie (offset_normal_px, offset_longitudinal_px, angle_deg).

        Orientation déterministe (deux étapes) :
          1. Canonique : tangent[0] >= 0. Si tangent[0] ≈ 0 : tangent[1] >= 0.
             Garantit un signe cohérent entre épisodes et indépendamment de l'ordre
             des points entrés dans la SVD.
          2. Continuité temporelle : aligne avec _prev_tangent pour éviter les
             sauts 180° à chaque frame (appliqué après la canonicalisation).
        """
        line_pts = np.asarray(line_pts, dtype=np.float32).reshape(-1, 2)
        laser_uv = np.asarray(laser_uv, dtype=np.float32).reshape(2)
        if len(line_pts) < 2:
            return 0.0, 0.0, 0.0

        dists = np.linalg.norm(line_pts - laser_uv, axis=1)
        order = np.argsort(dists)
        N = min(18, len(line_pts))
        neighbors = line_pts[order[:N]]
        centroid = neighbors.mean(axis=0)
        centered = neighbors - centroid
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        tangent = Vt[0].astype(np.float32)
        tn = float(np.linalg.norm(tangent))
        tangent = np.array([1.0, 0.0], dtype=np.float32) if tn < 1e-9 else tangent / tn

        # La tangente doit indiquer le sens départ -> arrivée. La caméra statique
        # voit le drapeau vert : on choisit donc le signe pointant vers son centroïde.
        # Sans drapeau visible, repli déterministe + continuité temporelle.
        if goal_uv is not None:
            goal_vec = np.asarray(goal_uv, dtype=np.float32).reshape(2) - centroid
            if float(np.linalg.norm(goal_vec)) > 1e-6 and float(np.dot(tangent, goal_vec)) < 0.0:
                tangent = -tangent
        else:
            if tangent[0] < 0.0 or (abs(float(tangent[0])) < 1e-6 and tangent[1] < 0.0):
                tangent = -tangent
            if self._prev_tangent is not None and np.dot(tangent, self._prev_tangent) < 0.0:
                tangent = -tangent

        self._prev_tangent = tangent.copy()

        normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
        delta = laser_uv - centroid   # centroïde, pas nearest (stable)
        offset_normal = float(np.dot(delta, normal))
        offset_long   = float(np.dot(delta, tangent))
        angle_deg = float(np.degrees(math.atan2(float(tangent[1]), float(tangent[0]))))
        return offset_normal, offset_long, angle_deg


def main(args=None):
    rclpy.init(args=args)
    node = CameraLineDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
