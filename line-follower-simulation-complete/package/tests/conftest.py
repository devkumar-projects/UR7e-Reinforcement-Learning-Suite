"""
conftest.py — Fixtures partagées pour les tests ur7e_line_follower.

Portabilité : sys.path est calculé depuis __file__, jamais en dur.
"""
import sys
import pathlib
import numpy as np
import pytest

# Ajoute la racine du dépôt (ur7e_line_follower/) au sys.path
# pour que `import ur7e_line_follower` fonctionne sans installation préalable.
_pkg_root = pathlib.Path(__file__).resolve().parents[1]  # racine du package
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))


def make_det():
    """Instancie CameraLineDetector sans ROS (sans __init__)."""
    from ur7e_line_follower.camera_line_detector import CameraLineDetector
    det = CameraLineDetector.__new__(CameraLineDetector)
    det._init_state()
    return det


def make_line_frame(angle_deg: float, width: int = 320, height: int = 240) -> np.ndarray:
    """Frame BGR synthétique : ligne bleue à angle_deg + laser rouge au centre."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cx, cy = width // 2, height // 2
    theta = np.radians(angle_deg)
    for t in range(-120, 120):
        u = int(cx + t * np.cos(theta))
        v = int(cy + t * np.sin(theta))
        if 0 <= u < width and 0 <= v < height:
            frame[v, u] = [200, 20, 10]   # BGR bleu
    frame[cy - 3:cy + 3, cx - 3:cx + 3] = [20, 20, 200]   # laser rouge
    return frame


@pytest.fixture
def det():
    return make_det()
