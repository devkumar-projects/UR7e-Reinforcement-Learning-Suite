"""Tests unitaires pour la formule klt_confidence de _update_klt()."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from ur7e_line_follower.camera_line_detector import CameraLineDetector

def make_det():
    det = CameraLineDetector.__new__(CameraLineDetector)
    det._init_state()
    return det


@pytest.fixture
def gray():
    return np.zeros((240, 320), dtype=np.uint8)


@pytest.fixture
def full_mask():
    m = np.ones((240, 320), dtype=np.uint8) * 255
    return m


@pytest.fixture
def empty_mask():
    return np.zeros((240, 320), dtype=np.uint8)


def _pts(n: int) -> np.ndarray:
    return np.random.rand(n, 2).astype(np.float32) * [300, 220] + [10, 10]


def test_conf_reinit_80_pts(monkeypatch, gray, full_mask):
    """80 points reinit → conf = 1.0."""
    det = make_det()
    pts80 = _pts(80)
    monkeypatch.setattr(det, '_sample_line_points', lambda mask: pts80)
    _, conf = det._update_klt(gray, full_mask, line_detected=True)
    assert abs(conf - 1.0) < 1e-6, f"conf={conf}, attendu=1.0"


def test_conf_reinit_6_pts(monkeypatch, gray, full_mask):
    """6 points reinit → conf = 6/30 = 0.20, pas 1.0."""
    det = make_det()
    pts6 = _pts(6)
    monkeypatch.setattr(det, '_sample_line_points', lambda mask: pts6)
    _, conf = det._update_klt(gray, full_mask, line_detected=True)
    expected = 6 / 30
    assert abs(conf - expected) < 1e-4, f"conf={conf}, attendu={expected}"


def test_conf_reinit_none(monkeypatch, gray, empty_mask):
    """Sample retourne None → conf = 0, état réinitialisé."""
    det = make_det()
    monkeypatch.setattr(det, '_sample_line_points', lambda mask: None)
    _, conf = det._update_klt(gray, empty_mask, line_detected=True)
    assert conf == 0.0
    assert det._prev_pts is None
    assert det._n_init_pts == 0


def test_conf_tracking_80to40(monkeypatch, gray, full_mask):
    """Tracking : 80 pts init, 40 trackés → densité=1, rétention=0.5, conf=0.75."""
    det = make_det()
    pts80 = _pts(80)
    monkeypatch.setattr(det, '_sample_line_points', lambda m: pts80)
    det._update_klt(gray, full_mask, True)   # reinit

    det._prev_gray = gray.copy()
    det._frame_idx = 1

    pts40 = _pts(40)
    monkeypatch.setattr(det, '_klt_track', lambda *_: pts40)
    _, conf = det._update_klt(gray, full_mask, True)
    expected = 1.0 * (0.5 + 0.5 * (40 / 80))
    assert abs(conf - expected) < 1e-4, f"conf={conf}, attendu={expected:.4f}"


def test_conf_tracking_no_pts(monkeypatch, gray, full_mask):
    """Tracking échoue (None) → conf = 0, état réinitialisé."""
    det = make_det()
    pts80 = _pts(80)
    monkeypatch.setattr(det, '_sample_line_points', lambda m: pts80)
    det._update_klt(gray, full_mask, True)   # reinit

    det._prev_gray = gray.copy()
    det._frame_idx = 1
    monkeypatch.setattr(det, '_klt_track', lambda *_: None)
    _, conf = det._update_klt(gray, full_mask, True)
    assert conf == 0.0


def test_conf_no_detection(gray, empty_mask):
    """line_detected=False → conf = 0."""
    det = make_det()
    _, conf = det._update_klt(gray, empty_mask, line_detected=False)
    assert conf == 0.0


def test_conf_clip_max(monkeypatch, gray, full_mask):
    """Plus de MAX_KLT_POINTS tracked → conf clampé à 1.0."""
    det = make_det()
    from ur7e_line_follower.camera_line_detector import MAX_KLT_POINTS
    pts_over = _pts(MAX_KLT_POINTS + 10)
    monkeypatch.setattr(det, '_sample_line_points', lambda m: pts_over)
    _, conf = det._update_klt(gray, full_mask, True)
    assert conf <= 1.0 + 1e-9
