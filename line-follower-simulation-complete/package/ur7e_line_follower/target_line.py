"""
Target line definitions for the line-following task.
All lines are in wall frame: (y, z) coordinates.
  y : horizontal on wall, robot's right is +y, range ≈ [-0.5, 0.5]
  z : vertical on wall, up is +z, range ≈ [0.35, 0.95]
"""
import numpy as np
from pathlib import Path

WALL_X    = 1.0
WALL_Y_MIN, WALL_Y_MAX = -0.65, 0.65
WALL_Z_MIN, WALL_Z_MAX =  0.20, 1.30

N_WAYPOINTS = 50


def s_curve(n: int = N_WAYPOINTS) -> np.ndarray:
    """One full period of a sine wave, rising from bottom to top of wall."""
    t = np.linspace(0, 1, n)
    y = 0.40 * np.sin(2 * np.pi * t)
    z = 0.40 + 0.55 * t
    return np.stack([y, z], axis=1)


def zigzag(n: int = N_WAYPOINTS) -> np.ndarray:
    """Triangle wave."""
    t = np.linspace(0, 1, n)
    y = 0.35 * (2 * np.abs(2 * (t - np.floor(t + 0.5))) - 1)
    z = 0.40 + 0.55 * t
    return np.stack([y, z], axis=1)


def circle_arc(n: int = N_WAYPOINTS, radius: float = 0.28) -> np.ndarray:
    """Semi-circle."""
    angles = np.linspace(-np.pi / 2, np.pi / 2, n)
    y = radius * np.cos(angles)
    z = 0.67 + radius * np.sin(angles)
    return np.stack([y, z], axis=1)


def figure_eight(n: int = N_WAYPOINTS, scale: float = 0.25) -> np.ndarray:
    """Lemniscate of Bernoulli."""
    t = np.linspace(0, 2 * np.pi, n)
    denom = 1 + np.sin(t) ** 2
    y = scale * np.sqrt(2) * np.cos(t) / denom
    z = 0.67 + scale * np.sqrt(2) * np.sin(t) * np.cos(t) / denom
    return np.stack([y, z], axis=1)


def horizontal_line(n: int = N_WAYPOINTS) -> np.ndarray:
    y = np.linspace(-0.40, 0.40, n)
    z = np.full(n, 0.67)
    return np.stack([y, z], axis=1)


def diagonal(n: int = N_WAYPOINTS) -> np.ndarray:
    y = np.linspace(-0.35, 0.35, n)
    z = np.linspace(0.40, 0.95, n)
    return np.stack([y, z], axis=1)


SHAPES = {
    's_curve':        s_curve,
    'zigzag':         zigzag,
    'circle_arc':     circle_arc,
    'figure_eight':   figure_eight,
    'horizontal':     horizontal_line,
    'diagonal':       diagonal,
}
DEFAULT_SHAPE = 's_curve'

# Seuil RMSE pour déclarer une trajectoire réussie (mètres)
RMSE_SUCCESS_THRESH = 0.04   # 4 cm — assez exigeant pour forcer un bon suivi

# Longueur minimale d'une trajectoire aléatoire
MIN_PATH_LENGTH = 0.50   # 50 cm


def straight_line_from_start(start_point: np.ndarray, length: float = 0.25,
                             n: int = N_WAYPOINTS) -> np.ndarray:
    """Short vertical line anchored at the current laser point.

    The direction is selected to stay inside the wall.  This is the deterministic
    trajectory used by the minimal convergence profile.
    """
    start = np.asarray(start_point, dtype=np.float64).reshape(2).copy()
    margin = 0.05
    start[0] = np.clip(start[0], WALL_Y_MIN + margin, WALL_Y_MAX - margin)
    start[1] = np.clip(start[1], WALL_Z_MIN + margin, WALL_Z_MAX - margin)
    length = float(max(length, 0.05))
    up_room = WALL_Z_MAX - margin - start[1]
    down_room = start[1] - (WALL_Z_MIN + margin)
    signed_length = min(length, up_room) if up_room >= min(length, down_room) else -min(length, down_room)
    end = start + np.array([0.0, signed_length], dtype=np.float64)
    t = np.linspace(0.0, 1.0, int(n))[:, None]
    pts = start[None, :] * (1.0 - t) + end[None, :] * t
    pts[0] = start
    return pts.astype(np.float32)


def anchor_line_to_start(waypoints: np.ndarray, start_point: np.ndarray,
                         margin: float = 0.02) -> np.ndarray:
    """Translate a fixed trajectory so its first point matches the reset dot.

    A final uniform scale around the start point keeps the complete path inside
    the wall without changing its topology.
    """
    wp = np.asarray(waypoints, dtype=np.float64).copy()
    start = np.asarray(start_point, dtype=np.float64).reshape(2)
    wp += start - wp[0]
    delta = wp - start
    scale = 1.0
    for axis, lo, hi in ((0, WALL_Y_MIN + margin, WALL_Y_MAX - margin),
                         (1, WALL_Z_MIN + margin, WALL_Z_MAX - margin)):
        positive = delta[:, axis] > 1e-12
        negative = delta[:, axis] < -1e-12
        if np.any(positive):
            scale = min(scale, float((hi - start[axis]) / np.max(delta[positive, axis])))
        if np.any(negative):
            scale = min(scale, float((lo - start[axis]) / np.min(delta[negative, axis])))
    scale = float(np.clip(scale, 0.05, 1.0))
    wp = start + scale * delta
    wp[0] = start
    return wp.astype(np.float32)


def waypoint_abscissae(waypoints: np.ndarray) -> np.ndarray:
    """Cumulative arc-length coordinate of every waypoint."""
    wp = np.asarray(waypoints, dtype=np.float64)
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(wp, axis=0), axis=1))])


def _catmull_rom_spline(ctrl: np.ndarray, n: int) -> np.ndarray:
    """
    Interpolation Catmull-Rom à travers les points de contrôle ctrl (N_ctrl, 2).
    Retourne n points régulièrement espacés en longueur d'arc.
    """
    # Points fantômes aux extrémités
    p = np.vstack([
        ctrl[0]  + (ctrl[0]  - ctrl[1]),
        ctrl,
        ctrl[-1] + (ctrl[-1] - ctrl[-2]),
    ])
    segments = len(ctrl) - 1
    dense = []
    pts_per_seg = max(10, n)  # sur-échantillonnage pour longueur d'arc
    for i in range(segments):
        p0, p1, p2, p3 = p[i], p[i + 1], p[i + 2], p[i + 3]
        t = np.linspace(0.0, 1.0, pts_per_seg)[:, None]
        seg = 0.5 * (
            2 * p1
            + (-p0 + p2) * t
            + (2*p0 - 5*p1 + 4*p2 - p3) * t**2
            + (-p0 + 3*p1 - 3*p2 + p3) * t**3
        )
        dense.append(seg[:-1] if i < segments - 1 else seg)
    dense = np.vstack(dense)

    # Reparamétrisation par longueur d'arc → n points équidistants
    dists = np.concatenate([[0.0],
                             np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))])
    total = dists[-1]
    s_target = np.linspace(0.0, total, n)
    pts = np.empty((n, 2), dtype=np.float64)
    for j, s in enumerate(s_target):
        idx = np.searchsorted(dists, s, side='right') - 1
        idx = int(np.clip(idx, 0, len(dense) - 2))
        if dists[idx + 1] - dists[idx] < 1e-10:
            pts[j] = dense[idx]
        else:
            t_loc = (s - dists[idx]) / (dists[idx + 1] - dists[idx])
            pts[j] = dense[idx] * (1 - t_loc) + dense[idx + 1] * t_loc
    return pts


def random_line(rng: np.random.Generator | None = None,
                n: int = N_WAYPOINTS,
                min_length: float = MIN_PATH_LENGTH,
                max_attempts: int = 60) -> np.ndarray:
    """
    Génère une trajectoire aléatoire fluide (Catmull-Rom) sur le mur.

    Contraintes :
    - Longueur d'arc totale >= min_length (défaut 50 cm)
    - Tous les points dans les limites du mur
    - 3 à 5 points de contrôle aléatoires

    Retourne un tableau (n, 2) de [y, z] en coordonnées mur.
    """
    if rng is None:
        rng = np.random.default_rng()

    MARGIN = 0.06  # marge par rapport aux bords du mur

    for _ in range(max_attempts):
        n_ctrl = int(rng.integers(3, 6))  # 3, 4 ou 5 points de contrôle

        # Points de contrôle aléatoires dans les limites du mur
        ctrl_y = rng.uniform(WALL_Y_MIN + MARGIN, WALL_Y_MAX - MARGIN, n_ctrl)
        ctrl_z = rng.uniform(WALL_Z_MIN + MARGIN, WALL_Z_MAX - MARGIN, n_ctrl)

        # Tri par z pour favoriser une progression naturelle (bas→haut ou haut→bas)
        direction = rng.choice([-1, 1])
        order = np.argsort(ctrl_z)[::direction]
        ctrl_y = ctrl_y[order]
        ctrl_z = ctrl_z[order]
        ctrl = np.stack([ctrl_y, ctrl_z], axis=1)

        pts = _catmull_rom_spline(ctrl, n)

        # Vérification des limites
        if not (np.all(pts[:, 0] >= WALL_Y_MIN + 0.01) and
                np.all(pts[:, 0] <= WALL_Y_MAX - 0.01) and
                np.all(pts[:, 1] >= WALL_Z_MIN + 0.01) and
                np.all(pts[:, 1] <= WALL_Z_MAX - 0.01)):
            continue

        # Vérification longueur d'arc
        arc_length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        if arc_length >= min_length:
            return pts.astype(np.float32)

    # Fallback : s_curve classique si aucune tentative valide
    return s_curve(n).astype(np.float32)

# Point laser attendu à HOME (coordonnées mur y,z). Utilisé pour que chaque
# dessin aléatoire commence exactement là où le robot est réinitialisé.
DEFAULT_HOME_DOT = np.array([0.0, 0.488], dtype=np.float64)


def random_line_from_start(rng: np.random.Generator | None = None,
                           start_point: np.ndarray | None = None,
                           n: int = N_WAYPOINTS,
                           min_length: float = MIN_PATH_LENGTH,
                           max_attempts: int = 100) -> np.ndarray:
    """Trajectoire aléatoire fluide ancrée au point laser de départ.

    Le premier waypoint est exactement ``start_point``. Les points de contrôle
    progressent ensuite majoritairement vers le haut du mur afin de fournir une
    direction départ→arrivée non ambiguë et une tâche apprenable après chaque
    reset du robot.
    """
    if rng is None:
        rng = np.random.default_rng()
    start = np.asarray(DEFAULT_HOME_DOT if start_point is None else start_point,
                       dtype=np.float64).reshape(2).copy()
    margin = 0.07
    start[0] = np.clip(start[0], WALL_Y_MIN + margin, WALL_Y_MAX - margin)
    start[1] = np.clip(start[1], WALL_Z_MIN + margin, WALL_Z_MAX - margin)

    for _ in range(max_attempts):
        n_ctrl = int(rng.integers(4, 7))
        z_end_min = min(max(start[1] + 0.45, WALL_Z_MIN + 0.55), WALL_Z_MAX - margin)
        if z_end_min >= WALL_Z_MAX - margin - 1e-3:
            z_end = max(WALL_Z_MIN + margin, start[1] - 0.55)
            z_values = np.linspace(start[1], z_end, n_ctrl)
        else:
            z_end = float(rng.uniform(z_end_min, WALL_Z_MAX - margin))
            z_values = np.linspace(start[1], z_end, n_ctrl)
        z_jitter = rng.normal(0.0, 0.025, n_ctrl)
        z_jitter[[0, -1]] = 0.0
        z_values = np.maximum.accumulate(z_values + z_jitter)
        z_values[0] = start[1]
        z_values[-1] = z_end

        y_values = rng.uniform(WALL_Y_MIN + margin, WALL_Y_MAX - margin, n_ctrl)
        y_values[0] = start[0]
        # Évite un premier virage brutal tout en gardant la forme aléatoire.
        y_values[1] = np.clip(start[0] + rng.normal(0.0, 0.16),
                              WALL_Y_MIN + margin, WALL_Y_MAX - margin)
        ctrl = np.stack([y_values, z_values], axis=1)
        pts = _catmull_rom_spline(ctrl, n)
        pts[0] = start

        in_bounds = (
            np.all(pts[:, 0] >= WALL_Y_MIN + 0.01) and
            np.all(pts[:, 0] <= WALL_Y_MAX - 0.01) and
            np.all(pts[:, 1] >= WALL_Z_MIN + 0.01) and
            np.all(pts[:, 1] <= WALL_Z_MAX - 0.01)
        )
        length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        if in_bounds and length >= min_length:
            pts[0] = start
            return pts.astype(np.float32)

    fallback = s_curve(n).astype(np.float64)
    fallback += start - fallback[0]
    fallback[:, 0] = np.clip(fallback[:, 0], WALL_Y_MIN + 0.01, WALL_Y_MAX - 0.01)
    fallback[:, 1] = np.clip(fallback[:, 1], WALL_Z_MIN + 0.01, WALL_Z_MAX - 0.01)
    fallback[0] = start
    return fallback.astype(np.float32)


def arc_length(waypoints: np.ndarray) -> float:
    """Longueur d'arc totale d'une trajectoire (n, 2)."""
    return float(np.sum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1)))


def load_line(path=None, shape: str = DEFAULT_SHAPE) -> np.ndarray:
    """
    Load a target line.
    - path=None : use the named shape
    - path=*.npy : load numpy array of shape (N, 2)
    - path=*.png : extract centerline from a hand-drawn line image
    """
    if path is not None:
        path = Path(path)
        if path.suffix == '.npy':
            pts = np.load(str(path))
            assert pts.ndim == 2 and pts.shape[1] == 2, "Line must be (N,2) array of [y,z]"
            return pts.astype(np.float32)
        if path.suffix in ('.png', '.jpg', '.jpeg'):
            return _from_image(path)
    return SHAPES[shape]().astype(np.float32)


def _from_image(path: Path) -> np.ndarray:
    """Extract and normalize a line from a hand-drawn PNG (dark line on white bg)."""
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("pip install Pillow  to load line images")
    try:
        from skimage.morphology import skeletonize
    except ImportError:
        raise ImportError("pip install scikit-image  to load line images")

    img = np.array(Image.open(path).convert('L'))
    binary = img < 128
    skel = skeletonize(binary)
    ys_px, xs_px = np.where(skel)
    if len(ys_px) == 0:
        raise ValueError(f"No line found in {path}. Draw a dark line on a white background.")

    # Sort top-to-bottom (increasing y pixel = decreasing real-world z)
    order = np.argsort(ys_px)
    ys_px, xs_px = ys_px[order], xs_px[order]

    idx = np.linspace(0, len(ys_px) - 1, N_WAYPOINTS, dtype=int)
    ys_px, xs_px = ys_px[idx], xs_px[idx]

    H, W = img.shape
    y = (xs_px / W - 0.5) * (WALL_Y_MAX - WALL_Y_MIN)
    z = WALL_Z_MAX - (ys_px / H) * (WALL_Z_MAX - WALL_Z_MIN)
    return np.stack([y, z], axis=1).astype(np.float32)


def save_line(waypoints: np.ndarray, path):
    """Save waypoints to .npy file."""
    np.save(str(path), waypoints.astype(np.float32))
    print(f"[line] Saved {len(waypoints)} waypoints → {path}")


def nearest_point_on_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Closest point on segment [a,b] to point p, and signed distance."""
    ab = b - a
    ab_len2 = np.dot(ab, ab)
    if ab_len2 < 1e-12:
        return a, np.linalg.norm(p - a)
    t = np.clip(np.dot(p - a, ab) / ab_len2, 0.0, 1.0)
    closest = a + t * ab
    return closest, np.linalg.norm(p - closest)


def distance_to_line(dot: np.ndarray, waypoints: np.ndarray) -> float:
    """Minimum distance from dot (y,z) to the polyline defined by waypoints."""
    min_d = float('inf')
    for i in range(len(waypoints) - 1):
        _, d = nearest_point_on_segment(dot, waypoints[i], waypoints[i + 1])
        if d < min_d:
            min_d = d
    return float(min_d)


def closest_point_on_polyline(p: np.ndarray, waypoints: np.ndarray, start_idx: int = 0,
                              window: int | None = None):
    """Point le plus proche sur une fenêtre ordonnée de la polyligne.

    Retourne dict: distance, closest, segment_index, abscissa.
    """
    p = np.asarray(p, dtype=np.float64).reshape(2)
    wp = np.asarray(waypoints, dtype=np.float64)
    nseg = max(len(wp) - 1, 1)
    if window is None:
        i0, i1 = 0, nseg
    else:
        i0 = max(0, int(start_idx) - int(window))
        i1 = min(nseg, int(start_idx) + int(window) + 1)
    best = (float('inf'), wp[0].copy(), i0, 0.0)
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(wp, axis=0), axis=1))])
    for i in range(i0, i1):
        a, b = wp[i], wp[i + 1]
        ab = b - a
        den = float(np.dot(ab, ab))
        t = 0.0 if den < 1e-12 else float(np.clip(np.dot(p - a, ab) / den, 0.0, 1.0))
        c = a + t * ab
        d = float(np.linalg.norm(p - c))
        if d < best[0]:
            s = float(cum[i] + t * np.linalg.norm(ab))
            best = (d, c, i, s)
    return {'distance': best[0], 'closest': best[1], 'segment_index': best[2], 'abscissa': best[3]}


def resample_by_arclength(waypoints: np.ndarray, n: int = N_WAYPOINTS) -> np.ndarray:
    wp = np.asarray(waypoints, dtype=np.float64)
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(wp, axis=0), axis=1))])
    if d[-1] < 1e-12:
        return np.repeat(wp[:1], n, axis=0).astype(np.float32)
    s = np.linspace(0.0, d[-1], n)
    y = np.interp(s, d, wp[:, 0])
    z = np.interp(s, d, wp[:, 1])
    return np.stack([y, z], axis=1).astype(np.float32)


def curriculum_line_from_start(rng: np.random.Generator | None = None,
                               start_point: np.ndarray | None = None,
                               level: int = 0,
                               n: int = N_WAYPOINTS) -> np.ndarray:
    """Generate an anchored trajectory with increasing difficulty.

    level 0: straight / gentle curve, small lateral excursion.
    level 1: moderate Catmull-Rom curve.
    level 2+: full random generator used by the final task.
    """
    if rng is None:
        rng = np.random.default_rng()
    start = np.asarray(DEFAULT_HOME_DOT if start_point is None else start_point,
                       dtype=np.float64).reshape(2).copy()
    margin = 0.08
    start[0] = np.clip(start[0], WALL_Y_MIN + margin, WALL_Y_MAX - margin)
    start[1] = np.clip(start[1], WALL_Z_MIN + margin, WALL_Z_MAX - margin)
    level = int(max(0, level))
    if level >= 2:
        return random_line_from_start(rng, start, n=n)

    for _ in range(100):
        if level == 0:
            n_ctrl = 3
            dz = float(rng.uniform(0.50, 0.68))
            z_end = min(start[1] + dz, WALL_Z_MAX - margin)
            if z_end - start[1] < 0.42:
                z_end = max(WALL_Z_MIN + margin, start[1] - dz)
            z = np.linspace(start[1], z_end, n_ctrl)
            drift = float(rng.uniform(-0.18, 0.18))
            y_end = np.clip(start[0] + drift, WALL_Y_MIN + margin, WALL_Y_MAX - margin)
            bend = float(rng.uniform(-0.08, 0.08))
            y_mid = np.clip((start[0] + y_end) * 0.5 + bend,
                            WALL_Y_MIN + margin, WALL_Y_MAX - margin)
            ctrl = np.array([[start[0], start[1]], [y_mid, z[1]], [y_end, z[2]]])
        else:
            n_ctrl = 4
            dz = float(rng.uniform(0.55, 0.78))
            z_end = min(start[1] + dz, WALL_Z_MAX - margin)
            if z_end - start[1] < 0.45:
                z_end = max(WALL_Z_MIN + margin, start[1] - dz)
            z = np.linspace(start[1], z_end, n_ctrl)
            y = np.empty(n_ctrl, dtype=np.float64)
            y[0] = start[0]
            y[1:] = np.clip(start[0] + rng.normal(0.0, 0.22, n_ctrl - 1),
                            WALL_Y_MIN + margin, WALL_Y_MAX - margin)
            ctrl = np.column_stack((y, z))

        pts = _catmull_rom_spline(ctrl, n)
        pts[0] = start
        in_bounds = (
            np.all(pts[:, 0] >= WALL_Y_MIN + 0.01)
            and np.all(pts[:, 0] <= WALL_Y_MAX - 0.01)
            and np.all(pts[:, 1] >= WALL_Z_MIN + 0.01)
            and np.all(pts[:, 1] <= WALL_Z_MAX - 0.01)
        )
        length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        if in_bounds and length >= MIN_PATH_LENGTH:
            return pts.astype(np.float32)
    return random_line_from_start(rng, start, n=n)
