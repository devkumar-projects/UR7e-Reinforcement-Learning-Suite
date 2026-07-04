"""Tests tangente orientée vers le drapeau vert + continuité angulaire."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from ur7e_line_follower.camera_line_detector import CameraLineDetector

def make_det():
    det = CameraLineDetector.__new__(CameraLineDetector)
    det._init_state()
    return det

def make_line_frame(angle_deg: float, width: int = 320, height: int = 240) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cx, cy = width // 2, height // 2
    theta = np.radians(angle_deg)
    import cv2
    p1 = (int(cx - 120 * np.cos(theta)), int(cy - 120 * np.sin(theta)))
    p2 = (int(cx + 120 * np.cos(theta)), int(cy + 120 * np.sin(theta)))
    cv2.line(frame, p1, p2, (200, 20, 10), thickness=5)
    frame[cy - 3:cy + 3, cx - 3:cx + 3] = [20, 20, 200]
    gu = int(cx + 105 * np.cos(theta)); gv = int(cy + 105 * np.sin(theta))
    if 8 <= gu < width - 8 and 8 <= gv < height - 8:
        import cv2
        cv2.circle(frame, (gu, gv), 7, (20, 220, 20), thickness=-1)
    return frame


def test_canonical_sign_deterministic():
    """Mêmes points, ordre aléatoire → même signe d'offset_n."""
    from ur7e_line_follower.camera_line_detector import CameraLineDetector
    pts_horiz = np.array([[float(u), 120.0] for u in range(10, 310, 5)], dtype=np.float32)
    laser = np.array([160.0, 110.0], dtype=np.float32)  # au-dessus de la ligne

    signs = []
    for seed in range(6):
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(pts_horiz)
        det = CameraLineDetector.__new__(CameraLineDetector)
        det._init_state()
        off_n, _, _ = det._compute_offset(laser, shuffled)
        signs.append(int(np.sign(off_n)))
    assert all(s == signs[0] for s in signs), \
        f"Signe offset_n incohérent selon l'ordre des points: {signs}"


def test_canonical_sign_after_reset():
    """Après _init_state(), même sign quel que soit l'ordre."""
    from ur7e_line_follower.camera_line_detector import CameraLineDetector
    pts = np.array([[float(u), 120.0] for u in range(10, 310, 5)], dtype=np.float32)
    laser = np.array([160.0, 130.0], dtype=np.float32)  # en-dessous de la ligne

    signs = []
    for _ in range(4):
        det = CameraLineDetector.__new__(CameraLineDetector)
        det._init_state()
        off_n, _, _ = det._compute_offset(laser, pts)
        signs.append(int(np.sign(off_n)))
    assert len(set(signs)) == 1, f"Signe change après reset: {signs}"


def test_tangent_89_91_continuity():
    """89°→91° : vecteur dirigé reste proche (distance < 0.2), pas de saut ±1."""
    det89 = make_det()
    r89 = det89.process_frame(make_line_frame(89))
    det91 = make_det()
    r91 = det91.process_frame(make_line_frame(91))

    v89 = np.array([r89.tangent_cos_t, r89.tangent_sin_t])
    v91 = np.array([r91.tangent_cos_t, r91.tangent_sin_t])
    dist = float(np.linalg.norm(v89 - v91))
    assert dist < 0.2, \
        f"|v89-v91|={dist:.4f}: saut à ±90° détecté\n  v89={v89}  v91={v91}"


def test_double_angle_no_discontinuity():
    """cos(θ) et sin(θ) ne sautent pas entre 44° et 46°."""
    det44 = make_det()
    r44 = det44.process_frame(make_line_frame(44))
    det46 = make_det()
    r46 = det46.process_frame(make_line_frame(46))

    v44 = np.array([r44.tangent_cos_t, r44.tangent_sin_t])
    v46 = np.array([r46.tangent_cos_t, r46.tangent_sin_t])
    dist = float(np.linalg.norm(v44 - v46))
    assert dist < 0.3, f"|v44-v46|={dist:.4f}"


def test_green_flag_orients_tangent_to_goal():
    det = make_det()
    r = det.process_frame(make_line_frame(0))
    assert r.tangent_cos_t > 0.7, (r.tangent_cos_t, r.tangent_sin_t)


def test_detection_vector_length():
    """process_frame() retourne un detection_vector de longueur 7."""
    det = make_det()
    r = det.process_frame(make_line_frame(45))
    dv = r.detection_vector
    assert len(dv) == 7, f"len(detection_vector)={len(dv)}, attendu=7"


def test_detection_vector_bounds():
    """Toutes les valeurs du detection_vector sont dans [-1, 1] sauf detected et laser_visible."""
    det = make_det()
    r = det.process_frame(make_line_frame(30))
    dv = r.detection_vector
    # indices 1..5 (offset, klt_conf, cos_2t, sin_2t, coverage)
    for i in [1, 2, 3, 4, 5]:
        assert -1.0 - 1e-6 <= dv[i] <= 1.0 + 1e-6, \
            f"detection_vector[{i}]={dv[i]} hors [-1,1]"
    # indices 0 et 6 (detected, laser_visible) : 0.0 ou 1.0
    assert dv[0] in (0.0, 1.0)
    assert dv[6] in (0.0, 1.0)


def test_guidance_vector_length_and_bounds():
    det = make_det()
    r = det.process_frame(make_line_frame(0))
    gv = r.guidance_vector
    assert len(gv) == 3
    assert all(-1.0 - 1e-6 <= v <= 1.0 + 1e-6 for v in gv)

def test_visual_lookahead_points_toward_green_flag():
    det = make_det()
    r = det.process_frame(make_line_frame(0))
    # Drapeau vert placé à droite dans make_line_frame(0).
    assert r.lookahead_du_norm >= -1e-4, r.guidance_vector


def test_red_laser_does_not_split_blue_path_or_zero_progress():
    """Le spot rouge superposé ne doit pas couper la ligne en deux composantes.

    Sans le closing 9x9, seule la moitié située après le laser était conservée et
    la progression visuelle restait artificiellement à zéro.
    """
    import cv2
    det = make_det()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    curve = np.array(
        [[150, 360], [210, 320], [280, 285], [350, 255], [430, 210], [505, 150]],
        dtype=np.int32,
    )
    cv2.polylines(frame, [curve], False, (220, 30, 20), thickness=10)
    cv2.rectangle(frame, (120, 340), (140, 410), (20, 20, 220), thickness=-1)
    cv2.rectangle(frame, (510, 115), (535, 185), (20, 220, 20), thickness=-1)
    cv2.circle(frame, (282, 282), 8, (20, 20, 240), thickness=-1)

    result = det.process_frame(frame)
    assert result.line_detected and result.laser_visible
    assert result.line_pts[:, 0].min() < 100.0, result.line_pts[:, 0].min()
    assert result.line_pts[:, 0].max() > 230.0, result.line_pts[:, 0].max()
    assert result.visual_progress > 0.05, result.guidance_vector
