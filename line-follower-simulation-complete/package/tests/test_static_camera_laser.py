"""Le drapeau rouge statique ne doit pas être confondu avec le spot laser."""
import cv2
import numpy as np

from ur7e_line_follower.camera_line_detector import CameraLineDetector, W, H


def _det():
    d = CameraLineDetector.__new__(CameraLineDetector)
    d._init_state()
    return d


def test_circular_laser_selected_over_large_red_flag():
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(mask, (20, 30), (75, 70), 255, -1)  # drapeau rouge
    cv2.circle(mask, (210, 145), 6, 255, -1)          # spot laser
    uv, visible = _det()._find_laser(mask)
    assert visible
    np.testing.assert_allclose(uv, [210, 145], atol=2.0)


def test_laser_tracking_prefers_temporal_continuity():
    d = _det()
    m1 = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(m1, (20, 30), (75, 70), 255, -1)
    cv2.circle(m1, (180, 120), 6, 255, -1)
    uv1, _ = d._find_laser(m1)
    m2 = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(m2, (20, 30), (75, 70), 255, -1)
    cv2.circle(m2, (184, 123), 6, 255, -1)
    uv2, _ = d._find_laser(m2)
    assert np.linalg.norm(uv2 - uv1) < 10.0


def test_long_blue_line_selected_over_compact_blue_robot_part():
    d = _det()
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(mask, (15, 150), (95, 220), 255, -1)  # pièce robot compacte
    cv2.line(mask, (130, 210), (295, 25), 255, 7)       # ligne cible longue
    selected = d._select_line_component(mask)
    assert selected[100, 228] > 0 or np.count_nonzero(selected[:, 120:]) > 500
    assert np.count_nonzero(selected[150:220, 15:95]) < 50
