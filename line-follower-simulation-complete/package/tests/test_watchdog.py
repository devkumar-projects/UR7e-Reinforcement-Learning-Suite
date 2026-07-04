"""
Tests watchdog caméra — appelle la vraie check_camera_health() avec horloge et
compteurs mockés (pas de ROS, pas de Gazebo).

Stratégie :
  - Instancier LineFollowerBridge.__new__() pour éviter __init__ (ROS).
  - Pré-remplir les compteurs et l'historique à la main.
  - Monkeypatch time.monotonic() avec une séquence [T0, T_mid, T_end]
    pour que la boucle while dans check_camera_health() tourne au moins une fois.
  - spin_once() injecte les messages lors du premier appel.
  - Appeler bridge.check_camera_health(probe_duration=1.0) et inspecter CameraHealth.
"""
import sys
import pathlib
import time
from collections import deque

import pytest
import numpy as np

_pkg_root = pathlib.Path(__file__).resolve().parents[1]
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

import ur7e_line_follower.bridge as bridge_mod
from ur7e_line_follower.bridge import (
    CameraHealth, LineFollowerBridge,
    TRANSPORT_MIN_HZ, TRANSPORT_MIN_COUNT, TRANSPORT_MAX_AGE_S,
    VISUAL_MIN_DETECTION_RATE, VISUAL_MIN_LASER_RATE,
)

print(f"[watchdog tests] bridge module: {bridge_mod.__file__}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_bridge():
    b = LineFollowerBridge.__new__(LineFollowerBridge)
    b._camera_frame_count    = 1000   # base, incrémenté par spin_once
    b._detection_frame_count = 1000
    b._last_camera_ts        = 0.0
    b._last_detection_ts     = 0.0
    b._last_detection_mono   = 0.0
    b._last_line_seen_mono   = 0.0
    b._last_laser_seen_mono  = 0.0
    b._cam_history           = deque(maxlen=200)

    class _FakeExec:
        def spin_once(self, timeout_sec=0): pass
    b._spin_executor = _FakeExec()
    return b


def _probe(b, monkeypatch,
           n_cam: int, n_det: int,
           cam_age: float = 0.1, det_age: float = 0.1,
           history: list | None = None,
           probe_duration: float = 1.0) -> CameraHealth:
    """
    Simule check_camera_health(probe_duration) avec n_cam/n_det messages injectés.

    Séquence time.monotonic() :
      [T0, T0+0.05, T0+probe_duration+0.01]
      → while loop tourne une fois (spin_once appelé une fois)
      → elapsed ≈ probe_duration → Hz = n_det/probe_duration
    """
    T0    = 1_000.0
    T_MID = T0 + 0.05                    # still < T0+probe_duration → while runs
    T_END = T0 + probe_duration + 0.01   # > T0+probe_duration → loop exits

    # Appels attendus : t0, condition, condition de sortie, elapsed, freshness.
    tseq = iter([T0, T_MID, T_END, T_END, T_END])
    monkeypatch.setattr(time, 'monotonic', lambda: next(tseq, T_END))

    T_NOW = 5_000.0
    monkeypatch.setattr(time, 'time', lambda: T_NOW)

    injected = [False]

    def _inject(timeout_sec=0):
        if not injected[0]:
            b._camera_frame_count    += n_cam
            b._detection_frame_count += n_det
            b._last_camera_ts      = T_NOW - cam_age
            b._last_detection_ts   = T_NOW - det_age
            b._last_camera_mono    = T_END - cam_age
            b._last_detection_mono = T_END - det_age
            if history:
                for entry in history:
                    b._cam_history.append(entry)
            injected[0] = True

    b._spin_executor.spin_once = _inject
    return b.check_camera_health(probe_duration=probe_duration)


# ── Scénario 1 : tout OK ──────────────────────────────────────────────────────

def test_both_healthy(monkeypatch):
    b = _make_bridge()
    hist = [(1.0, 1.0, 0.8)] * 45
    h = _probe(b, monkeypatch, n_cam=45, n_det=45,
               cam_age=0.1, det_age=0.1, history=hist)
    assert h.camera_transport_healthy, f"Attendu healthy: {h}"
    assert h.visual_tracking_healthy


# ── Scénario 2 : pas de /line_camera ─────────────────────────────────────────

def test_no_camera_transport_fails(monkeypatch):
    b = _make_bridge()
    h = _probe(b, monkeypatch, n_cam=0, n_det=45,
               cam_age=999.0, det_age=0.1, history=[(1.0, 1.0, 0.8)] * 45)
    assert not h.camera_transport_healthy
    assert h.n_camera_recv == 0


# ── Scénario 3 : pas de /line_detection ──────────────────────────────────────

def test_no_detection_transport_fails(monkeypatch):
    b = _make_bridge()
    h = _probe(b, monkeypatch, n_cam=45, n_det=0,
               cam_age=0.1, det_age=999.0, history=[])
    assert not h.camera_transport_healthy
    assert h.n_detection_recv == 0


# ── Scénario 4 : frames trop vieux ───────────────────────────────────────────

def test_stale_frames_transport_fails(monkeypatch):
    b = _make_bridge()
    hist = [(1.0, 1.0, 0.8)] * 45
    h = _probe(b, monkeypatch, n_cam=45, n_det=45,
               cam_age=1.5, det_age=1.5, history=hist)
    assert not h.camera_transport_healthy
    assert h.camera_age_s > TRANSPORT_MAX_AGE_S
    assert h.detection_age_s > TRANSPORT_MAX_AGE_S


# ── Scénario 5 : Hz insuffisant (5 msg en 1 s → 5 Hz < 10 Hz) ───────────────

def test_low_hz_transport_fails(monkeypatch):
    b = _make_bridge()
    h = _probe(b, monkeypatch, n_cam=5, n_det=5,
               cam_age=0.1, det_age=0.1, history=[(1.0, 1.0, 0.8)] * 5,
               probe_duration=1.0)
    assert not h.camera_transport_healthy
    assert h.camera_hz < TRANSPORT_MIN_HZ
    assert h.detection_hz < TRANSPORT_MIN_HZ


# ── Scénario 6 : visual faible (non bloquant) ────────────────────────────────

def test_low_visual_not_blocking(monkeypatch):
    b = _make_bridge()
    hist = [(0.0, 0.0, 0.0)] * 97 + [(1.0, 0.0, 0.5)] * 3
    h = _probe(b, monkeypatch, n_cam=45, n_det=45,
               cam_age=0.1, det_age=0.1, history=hist)
    assert h.camera_transport_healthy, "Transport doit rester OK malgré visual faible"
    assert not h.visual_tracking_healthy
    assert h.detection_rate < VISUAL_MIN_DETECTION_RATE


# ── Scénario 7 : exactement au seuil visuel ──────────────────────────────────

def test_visual_healthy_at_or_above_threshold(monkeypatch):
    b = _make_bridge()
    # 100 messages : 20 détections, dont 10 avec laser.
    hist = ([(1.0, 1.0, 0.5)] * 10
            + [(1.0, 0.0, 0.3)] * 10
            + [(0.0, 0.0, 0.0)] * 80)
    h = _probe(b, monkeypatch, n_cam=100, n_det=100,
               cam_age=0.1, det_age=0.1, history=hist)
    assert h.camera_transport_healthy
    assert h.detection_rate == pytest.approx(0.20)
    assert h.laser_rate == pytest.approx(0.10)
    assert h.visual_tracking_healthy


# ── camera_transport_alive ──────────────────────────────────────────────────

def test_camera_transport_alive_fresh(monkeypatch):
    b = _make_bridge()
    b._camera_frame_count = 5
    b._detection_frame_count = 5
    T = 1000.0
    b._last_camera_mono = T - 0.2
    b._last_detection_mono = T - 0.3
    monkeypatch.setattr(time, 'monotonic', lambda: T)
    assert b.camera_transport_alive is True
    assert b.cam_is_alive is True


def test_camera_transport_fails_when_raw_stale(monkeypatch):
    b = _make_bridge()
    b._camera_frame_count = 5
    b._detection_frame_count = 5
    T = 1000.0
    b._last_camera_mono = T - 2.0
    b._last_detection_mono = T - 0.1
    monkeypatch.setattr(time, 'monotonic', lambda: T)
    assert b.camera_transport_alive is False


def test_camera_transport_fails_when_detection_stale(monkeypatch):
    b = _make_bridge()
    b._camera_frame_count = 5
    b._detection_frame_count = 5
    T = 1000.0
    b._last_camera_mono = T - 0.1
    b._last_detection_mono = T - 2.0
    monkeypatch.setattr(time, 'monotonic', lambda: T)
    assert b.camera_transport_alive is False


def test_camera_transport_fails_without_either_stream(monkeypatch):
    b = _make_bridge()
    b._camera_frame_count = 0
    b._detection_frame_count = 5
    monkeypatch.setattr(time, 'monotonic', lambda: 1000.0)
    assert b.camera_transport_alive is False
