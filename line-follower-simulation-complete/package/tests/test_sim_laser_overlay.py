"""Tests for deterministic simulated laser projection into the camera image."""
import time
import cv2
import numpy as np

from ur7e_line_follower.camera_line_detector import (
    CameraLineDetector,
    project_sim_laser_to_pixel,
    W,
    H,
)


def _detector_with_overlay(y=0.0, z=0.488):
    det = CameraLineDetector.__new__(CameraLineDetector)
    det._init_state()
    det._use_sim_laser_overlay = True
    det._sim_laser_yz = np.array([y, z], dtype=np.float64)
    det._sim_laser_last_mono = time.monotonic()
    return det


def test_home_laser_projects_inside_camera():
    uv = project_sim_laser_to_pixel(0.0, 0.488, W, H)
    assert uv is not None
    assert 0 <= uv[0] < W
    assert 0 <= uv[1] < H


def test_overlay_is_detected_as_red_laser():
    det = _detector_with_overlay()
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    augmented = det._overlay_sim_laser(frame)
    hsv = cv2.cvtColor(augmented, cv2.COLOR_BGR2HSV)
    assert np.count_nonzero(hsv[:, :, 0] < 12) > 20
    result = det.process_frame(augmented)
    assert result.laser_visible
    assert result.laser_uv is not None


def test_stale_overlay_is_not_drawn():
    det = _detector_with_overlay()
    det._sim_laser_last_mono = time.monotonic() - 2.0
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    augmented = det._overlay_sim_laser(frame)
    assert np.array_equal(augmented, frame)


def test_use_sim_overlay_is_opt_in():
    det = _detector_with_overlay()
    det._use_sim_laser_overlay = False
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    assert np.array_equal(det._overlay_sim_laser(frame), frame)
