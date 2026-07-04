"""Physical-camera detector for a blue drawing and a red laser spot.

Outputs
-------
/line_detection (7):
  [line, signed_offset_norm, KLT_confidence, tangent_cos, tangent_sin,
   coverage_norm, laser_visible]
/line_guidance (3):
  [lookahead_du_norm, lookahead_dv_norm, visual_progress]
/line_measurement (12):
  [valid, laser_u, laser_v, line_u, line_v, tangent_u, tangent_v,
   KLT_confidence, signed_offset_px, progress, laser_visible, line_detected]
/camera_wall_measurement (11):
  [valid, laser_y, laser_z, line_y, line_z, tangent_y, tangent_z,
   confidence, cross_y, cross_z, progress]
/line_debug: BGR overlay for recording / rqt_image_view.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

W, H = 640, 480
SCALE_PX = math.hypot(W, H) / 2.0
MIN_TRACK_POINTS = 8
MAX_TRACK_POINTS = 120
REINIT_INTERVAL = 20
LOOKAHEAD_PX = 55.0
LOOKAHEAD_NORM = 120.0


def imgmsg_to_bgr(msg: Image) -> np.ndarray:
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8).copy()
    enc = msg.encoding.lower()
    if enc in ('bgr8', '8uc3'):
        return data.reshape(msg.height, msg.width, 3)
    if enc == 'rgb8':
        return data.reshape(msg.height, msg.width, 3)[:, :, ::-1].copy()
    if enc in ('mono8', '8uc1'):
        gray = data.reshape(msg.height, msg.width)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    arr = data.reshape(msg.height, msg.width, -1)
    if arr.shape[2] >= 3:
        return arr[:, :, :3].copy()
    raise ValueError(f'Unsupported image encoding {msg.encoding!r}')


def bgr_to_imgmsg(frame: np.ndarray, stamp, frame_id: str) -> Image:
    frame = np.ascontiguousarray(frame, dtype=np.uint8)
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = frame.shape[:2]
    msg.encoding = 'bgr8'
    msg.is_bigendian = 0
    msg.step = int(frame.shape[1] * 3)
    msg.data = frame.tobytes()
    return msg


def _as_hsv(values, default) -> np.ndarray:
    arr = np.asarray(values if len(values) == 3 else default, dtype=np.int32)
    return np.clip(arr, 0, 255).astype(np.uint8)


class VisualDetector(Node):
    def __init__(self) -> None:
        super().__init__('ur7e_visual_detector')
        self.declare_parameter('homography_file', '')
        self.declare_parameter('blue_hsv_lo', [95, 55, 35])
        self.declare_parameter('blue_hsv_hi', [145, 255, 255])
        self.declare_parameter('red_hsv_lo1', [0, 140, 140])
        self.declare_parameter('red_hsv_hi1', [14, 255, 255])
        self.declare_parameter('red_hsv_lo2', [164, 140, 140])
        self.declare_parameter('red_hsv_hi2', [180, 255, 255])
        self.declare_parameter('green_hsv_lo', [38, 60, 35])
        self.declare_parameter('green_hsv_hi', [92, 255, 255])
        self.declare_parameter('min_blue_pixels', 90)
        self.declare_parameter('debug_overlay', True)

        gp = self.get_parameter
        self.blue_lo = _as_hsv(gp('blue_hsv_lo').value, [95, 55, 35])
        self.blue_hi = _as_hsv(gp('blue_hsv_hi').value, [145, 255, 255])
        self.red_lo1 = _as_hsv(gp('red_hsv_lo1').value, [0, 140, 140])
        self.red_hi1 = _as_hsv(gp('red_hsv_hi1').value, [14, 255, 255])
        self.red_lo2 = _as_hsv(gp('red_hsv_lo2').value, [164, 140, 140])
        self.red_hi2 = _as_hsv(gp('red_hsv_hi2').value, [180, 255, 255])
        self.green_lo = _as_hsv(gp('green_hsv_lo').value, [38, 60, 35])
        self.green_hi = _as_hsv(gp('green_hsv_hi').value, [92, 255, 255])
        self.min_blue_pixels = int(gp('min_blue_pixels').value)
        self.debug_overlay = bool(gp('debug_overlay').value)
        self.homography_path = Path(str(gp('homography_file').value)).expanduser()

        self.pub_det = self.create_publisher(Float32MultiArray, '/line_detection', 10)
        self.pub_guidance = self.create_publisher(Float32MultiArray, '/line_guidance', 10)
        self.pub_raw = self.create_publisher(Float32MultiArray, '/line_measurement', 10)
        self.pub_wall = self.create_publisher(Float32MultiArray, '/camera_wall_measurement', 10)
        self.pub_debug = self.create_publisher(Image, '/line_debug', qos_profile_sensor_data)
        self.create_subscription(Image, '/line_camera', self.image_cb, qos_profile_sensor_data)

        self.prev_gray = None
        self.prev_pts = None
        self.prev_tangent = None
        self.prev_laser = None
        self.n_init = 0
        self.frame_idx = 0
        self.ema_tangent = None
        self.H_pix_to_wall = None
        self.homography_mtime = -1.0
        self._load_homography(force=True)
        self.get_logger().info('Visual detector ready: blue line + red laser + KLT + metric homography')

    def _load_homography(self, force: bool = False) -> None:
        path = self.homography_path
        if not path or not str(path) or not path.exists():
            self.H_pix_to_wall = None
            return
        mtime = path.stat().st_mtime
        if not force and mtime == self.homography_mtime:
            return
        try:
            data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
            Hm = np.asarray(data.get('homography', []), dtype=np.float64).reshape(3, 3)
            if not np.all(np.isfinite(Hm)):
                raise ValueError('non-finite homography')
            self.H_pix_to_wall = Hm
            self.homography_mtime = mtime
            self.get_logger().info(f'Loaded camera-wall homography: {path}')
        except Exception as exc:
            self.H_pix_to_wall = None
            self.get_logger().warning(f'Cannot load homography {path}: {exc}')

    def pixel_to_wall(self, uv: np.ndarray) -> np.ndarray | None:
        if self.H_pix_to_wall is None:
            return None
        p = np.array([float(uv[0]), float(uv[1]), 1.0])
        q = self.H_pix_to_wall @ p
        if abs(float(q[2])) < 1e-10:
            return None
        yz = q[:2] / q[2]
        return yz if np.all(np.isfinite(yz)) else None

    def image_cb(self, msg: Image) -> None:
        try:
            frame = imgmsg_to_bgr(msg)
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
            result = self.process(frame)
            self.publish_result(result, frame, msg)
        except Exception as exc:
            self.get_logger().warning(f'detector frame rejected: {exc}')

    def _select_line_component(self, mask: np.ndarray) -> np.ndarray:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        best, best_score = 0, -1.0
        for label in range(1, n):
            x, y, w, h, area = stats[label]
            if area < self.min_blue_pixels:
                continue
            diag = math.hypot(float(w), float(h))
            fill = float(area) / max(float(w * h), 1.0)
            elong = max(w, h) / max(min(w, h), 1)
            score = diag * (1.7 - min(fill, 1.0)) + 5.0 * min(float(elong), 10.0)
            if score > best_score:
                best, best_score = label, score
        out = np.zeros_like(mask)
        if best:
            out[labels == best] = 255
        return out

    def _find_laser(self, mask: np.ndarray) -> np.ndarray | None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if not (3.0 <= area <= 2500.0):
                continue
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 1e-6:
                continue
            circ = 4.0 * math.pi * area / (perimeter * perimeter)
            M = cv2.moments(contour)
            if M['m00'] <= 1e-9:
                continue
            uv = np.array([M['m10'] / M['m00'], M['m01'] / M['m00']], dtype=np.float32)
            x, y, w, h = cv2.boundingRect(contour)
            aspect = min(w, h) / max(w, h, 1)
            score = 2.2 * circ + 0.6 * aspect - 0.00025 * abs(area - 90.0)
            if self.prev_laser is not None:
                score -= 0.006 * float(np.linalg.norm(uv - self.prev_laser))
            candidates.append((score, uv))
        if not candidates:
            self.prev_laser = None
            return None
        uv = max(candidates, key=lambda x: x[0])[1]
        self.prev_laser = uv.copy()
        return uv

    @staticmethod
    def _find_goal(mask: np.ndarray) -> np.ndarray | None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if not (15.0 <= area <= 15000.0):
                continue
            M = cv2.moments(contour)
            if M['m00'] <= 1e-9:
                continue
            uv = np.array([M['m10'] / M['m00'], M['m01'] / M['m00']], dtype=np.float32)
            if best is None or area > best[0]:
                best = (area, uv)
        return None if best is None else best[1]

    def _sample_points(self, mask: np.ndarray) -> np.ndarray | None:
        pts = cv2.goodFeaturesToTrack(mask, maxCorners=MAX_TRACK_POINTS, qualityLevel=0.008,
                                      minDistance=5, blockSize=7)
        if pts is not None and len(pts) >= MIN_TRACK_POINTS:
            return pts.reshape(-1, 2).astype(np.float32)
        ys, xs = np.where(mask > 0)
        if len(xs) < MIN_TRACK_POINTS:
            return None
        idx = np.linspace(0, len(xs) - 1, min(MAX_TRACK_POINTS, len(xs)), dtype=int)
        return np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)

    def _update_klt(self, gray: np.ndarray, mask: np.ndarray, line_valid: bool) -> tuple[np.ndarray | None, float]:
        need = (self.prev_gray is None or self.prev_pts is None or
                len(self.prev_pts) < MIN_TRACK_POINTS or self.frame_idx % REINIT_INTERVAL == 0)
        if line_valid and need:
            pts = self._sample_points(mask)
            self.prev_gray = gray.copy()
            self.prev_pts = pts
            self.n_init = 0 if pts is None else len(pts)
            conf = 0.0 if pts is None else min(len(pts) / 45.0, 1.0)
            return pts, float(conf)
        if self.prev_gray is not None and self.prev_pts is not None:
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_pts.reshape(-1, 1, 2), None,
                winSize=(17, 17), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.02),
            )
            self.prev_gray = gray.copy()
            if nxt is not None and status is not None:
                good = []
                for i, ok in enumerate(status.reshape(-1)):
                    if not ok:
                        continue
                    u, v = nxt[i, 0]
                    ui, vi = int(round(u)), int(round(v))
                    if 0 <= ui < W and 0 <= vi < H and mask[vi, ui] > 0:
                        good.append([u, v])
                if len(good) >= MIN_TRACK_POINTS:
                    pts = np.asarray(good, dtype=np.float32)
                    self.prev_pts = pts
                    density = min(len(pts) / 45.0, 1.0)
                    retention = min(len(pts) / max(self.n_init, 1), 1.0)
                    return pts, float(density * (0.5 + 0.5 * retention))
            self.prev_pts = None
            self.n_init = 0
            return None, 0.0
        self.prev_gray = gray.copy()
        return None, 0.0

    def _local_geometry(self, laser: np.ndarray, pts: np.ndarray, goal: np.ndarray | None):
        d = np.linalg.norm(pts - laser, axis=1)
        neighbors = pts[np.argsort(d)[:min(24, len(pts))]]
        center = neighbors.mean(axis=0)
        _, _, vt = np.linalg.svd(neighbors - center, full_matrices=False)
        tangent = vt[0].astype(np.float64)
        tangent /= max(float(np.linalg.norm(tangent)), 1e-9)
        if goal is not None and np.dot(tangent, goal - center) < 0.0:
            tangent = -tangent
        elif goal is None and self.prev_tangent is not None and np.dot(tangent, self.prev_tangent) < 0.0:
            tangent = -tangent
        elif goal is None and (tangent[0] < 0 or (abs(tangent[0]) < 1e-6 and tangent[1] < 0)):
            tangent = -tangent
        self.prev_tangent = tangent.copy()
        if self.ema_tangent is None:
            self.ema_tangent = tangent.copy()
        else:
            self.ema_tangent = 0.72 * self.ema_tangent + 0.28 * tangent
            self.ema_tangent /= max(float(np.linalg.norm(self.ema_tangent)), 1e-9)
        tangent = self.ema_tangent.copy()
        normal = np.array([-tangent[1], tangent[0]])
        signed_offset = float(np.dot(laser - center, normal))
        closest = neighbors[int(np.argmin(np.linalg.norm(neighbors - laser, axis=1)))]

        delta = pts - laser
        forward = delta @ tangent
        lateral = np.abs(delta @ normal)
        valid = np.where(forward > 3.0)[0]
        lookahead = closest
        if len(valid):
            score = np.abs(forward[valid] - LOOKAHEAD_PX) + 0.3 * lateral[valid]
            lookahead = pts[valid[int(np.argmin(score))]]
        elif goal is not None:
            lookahead = goal
        guide = lookahead - laser

        progress = 0.0
        if goal is not None:
            start = pts[int(np.argmax(np.linalg.norm(pts - goal, axis=1)))]
            axis = goal - start
            den = float(np.dot(axis, axis))
            if den > 1e-6:
                progress = float(np.clip(np.dot(laser - start, axis) / den, 0.0, 1.0))
        return center, closest, tangent, signed_offset, guide, progress

    def process(self, frame: np.ndarray) -> dict:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        blue = cv2.inRange(hsv, self.blue_lo, self.blue_hi)
        blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, kernel)
        blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        blue = cv2.dilate(blue, kernel)
        blue = self._select_line_component(blue)
        red = cv2.bitwise_or(cv2.inRange(hsv, self.red_lo1, self.red_hi1),
                            cv2.inRange(hsv, self.red_lo2, self.red_hi2))
        green = cv2.inRange(hsv, self.green_lo, self.green_hi)

        blue_px = int(np.count_nonzero(blue))
        line_valid = blue_px >= self.min_blue_pixels
        laser = self._find_laser(red)
        goal = self._find_goal(green)
        pts, klt = self._update_klt(gray, blue, line_valid)
        if pts is None and line_valid:
            ys, xs = np.where(blue > 0)
            if len(xs) >= MIN_TRACK_POINTS:
                step = max(1, len(xs) // 150)
                pts = np.stack([xs[::step], ys[::step]], axis=1).astype(np.float32)

        valid = bool(line_valid and laser is not None and pts is not None and len(pts) >= MIN_TRACK_POINTS)
        closest = np.zeros(2)
        tangent = np.array([1.0, 0.0])
        signed_offset = 0.0
        guide = np.zeros(2)
        progress = 0.0
        if valid:
            _, closest, tangent, signed_offset, guide, progress = self._local_geometry(laser, pts, goal)
        coverage = float(np.clip((blue_px / float(W * H)) * 24.0, 0.0, 1.0))

        wall = None
        self._load_homography()
        if valid and self.H_pix_to_wall is not None:
            laser_yz = self.pixel_to_wall(laser)
            line_yz = self.pixel_to_wall(closest)
            tangent_end = self.pixel_to_wall(closest + 25.0 * tangent)
            if laser_yz is not None and line_yz is not None and tangent_end is not None:
                tangent_yz = tangent_end - line_yz
                tn = float(np.linalg.norm(tangent_yz))
                if tn > 1e-9:
                    tangent_yz /= tn
                    cross = line_yz - laser_yz
                    wall = (laser_yz, line_yz, tangent_yz, cross)

        self.frame_idx += 1
        return {
            'valid': valid, 'line': line_valid, 'laser': laser, 'goal': goal,
            'pts': pts, 'closest': closest, 'tangent': tangent,
            'signed_offset': signed_offset, 'guide': guide, 'progress': progress,
            'klt': float(klt), 'coverage': coverage, 'blue_mask': blue, 'wall': wall,
        }

    def publish_result(self, r: dict, frame: np.ndarray, src: Image) -> None:
        valid = r['valid']
        laser = r['laser'] if r['laser'] is not None else np.zeros(2)
        tangent = r['tangent']
        det = Float32MultiArray()
        det.data = [
            float(r['line']), float(np.clip(r['signed_offset'] / SCALE_PX, -1., 1.)),
            float(np.clip(r['klt'], 0., 1.)), float(np.clip(tangent[0], -1., 1.)),
            float(np.clip(tangent[1], -1., 1.)), float(r['coverage']),
            float(r['laser'] is not None),
        ]
        self.pub_det.publish(det)

        guide = Float32MultiArray()
        guide.data = [float(np.clip(r['guide'][0] / LOOKAHEAD_NORM, -1., 1.)),
                      float(np.clip(r['guide'][1] / LOOKAHEAD_NORM, -1., 1.)),
                      float(np.clip(r['progress'], 0., 1.))]
        self.pub_guidance.publish(guide)

        raw = Float32MultiArray()
        raw.data = [
            float(valid), float(laser[0]), float(laser[1]),
            float(r['closest'][0]), float(r['closest'][1]),
            float(tangent[0]), float(tangent[1]), float(r['klt']),
            float(r['signed_offset']), float(r['progress']),
            float(r['laser'] is not None), float(r['line']),
        ]
        self.pub_raw.publish(raw)

        metric = Float32MultiArray()
        if r['wall'] is None:
            metric.data = [0.0] * 11
        else:
            laser_yz, line_yz, tangent_yz, cross = r['wall']
            metric.data = [
                1.0, float(laser_yz[0]), float(laser_yz[1]),
                float(line_yz[0]), float(line_yz[1]),
                float(tangent_yz[0]), float(tangent_yz[1]),
                float(np.clip(r['klt'], 0., 1.)),
                float(cross[0]), float(cross[1]), float(r['progress']),
            ]
        self.pub_wall.publish(metric)

        if self.debug_overlay:
            dbg = frame.copy()
            contours, _ = cv2.findContours(r['blue_mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(dbg, contours, -1, (255, 180, 0), 1)
            if r['laser'] is not None:
                p = tuple(np.round(r['laser']).astype(int))
                cv2.circle(dbg, p, 9, (0, 0, 255), 2)
            if valid:
                c = tuple(np.round(r['closest']).astype(int))
                e = tuple(np.round(r['closest'] + 50.0 * tangent).astype(int))
                cv2.circle(dbg, c, 6, (0, 255, 255), -1)
                cv2.arrowedLine(dbg, c, e, (0, 255, 0), 2, tipLength=0.25)
            text = f"KLT={r['klt']:.2f} offset={r['signed_offset']:.1f}px progress={100*r['progress']:.0f}% H={'OK' if self.H_pix_to_wall is not None else 'MISSING'}"
            cv2.putText(dbg, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 30), 3)
            cv2.putText(dbg, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1)
            self.pub_debug.publish(bgr_to_imgmsg(dbg, src.header.stamp, src.header.frame_id))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisualDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
