#!/usr/bin/env python3
"""
Tests unitaires standalone pour ur7e_line_follower.
Ne nécessite PAS ROS ni Gazebo.

Composants testés :
  1. MGI / Cinématique (kinematics.py)   — 7 tests
  2. EKF / Filtre de Kalman (ekf.py)     — 7 tests
  3. LQR / Singularités (singularity.py) — 7 tests
  4. Monte Carlo / Lignes (target_line)  — 6 tests
  5. RL / Inférence modèle (train.py)    — 5 tests
  6. KLT / Caméra (camera_line_detector) — 7 tests

Usage :
    python3 test_components.py
    python3 test_components.py --skip-rl    (si pas de checkpoint disponible)
"""
import sys
import pathlib
import argparse
import traceback
import numpy as np

# Package root (…/ros2_ws/src) calculé depuis __file__ — pas de chemin en dur.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

PASS  = '\033[92m[PASS]\033[0m'
FAIL  = '\033[91m[FAIL]\033[0m'
WARN  = '\033[93m[WARN]\033[0m'
SEP   = '─' * 60

results: list[tuple[str, bool, str]] = []


def record_test(name: str, ok: bool, detail: str = ''):
    tag = PASS if ok else FAIL
    msg = f'  {tag} {name}'
    if detail:
        msg += f'  ->  {detail}'
    print(msg)
    results.append((name, ok, detail))
    return ok


def section(title: str):
    print(f'\n{SEP}\n  {title}\n{SEP}')


# ─────────────────────────────────────────────────────────────────
#  1. MGI / Cinématique
# ─────────────────────────────────────────────────────────────────
def test_kinematics():
    section('1. MGI / Cinématique inverse (kinematics.py)')
    from ur7e_line_follower.kinematics import (
        fk_ur, fk_ur_toolz, laser_wall_dot, jacobian, wall_jacobian,
        cartesian_to_joint_vel,
    )

    # HOME réelle issue de bridge.py — laser orienté vers le mur x=1.0
    HOME = np.array([-0.133, -1.5708, 1.5708, 0.0, 1.5708, 0.0])

    # T1 : FK retourne un vecteur (3,) sans NaN
    pos = fk_ur(HOME)
    record_test('FK home -> vecteur 3D valide',
         pos.shape == (3,) and not np.any(np.isnan(pos)),
         f'pos={np.round(pos, 3)}')

    # T2 : TCP doit être devant le robot (x > 0)
    record_test('FK home -> TCP devant le robot (x>0)',
         float(pos[0]) > 0.0,
         f'x={pos[0]:.3f}')

    # T3 : Outil Z (direction laser) normalisé
    toolz = fk_ur_toolz(HOME)
    norm = np.linalg.norm(toolz)
    record_test('Tool-Z normalisé (|v|~1)',
         abs(norm - 1.0) < 1e-4,
         f'|v|={norm:.6f}')

    # T4 : Intersection laser/mur — la home doit toucher le mur x=1.0
    dot = laser_wall_dot(HOME, wall_x=1.0)
    record_test('Laser touche le mur (x=1.0) depuis HOME',
         dot is not None,
         f'dot={np.round(dot, 3) if dot is not None else "None"}')

    # T5 : Jacobien forme (3, 6)
    J = jacobian(HOME)
    record_test('Jacobien shape (3,6)',
         J.shape == (3, 6),
         f'shape={J.shape}')

    # T6 : Jacobien du mur (2, 6) — vitesse laser sur le mur
    Jw = wall_jacobian(HOME, wall_x=1.0)
    record_test('Jacobien mur shape (2,6)',
         Jw.shape == (2, 6) and not np.all(Jw == 0),
         f'|J_wall|={np.linalg.norm(Jw):.4f}')

    # T7 : MGI vitesse cartésienne -> articulaire (saturation)
    tcp_vel = np.array([0.0, 0.05, 0.0])
    dq = cartesian_to_joint_vel(HOME, tcp_vel, max_jvel=1.5)
    record_test('MGI cartesian->joint (|dq| <= max_jvel)',
         dq.shape == (6,) and np.linalg.norm(dq) <= 1.5 + 1e-6,
         f'|dq|={np.linalg.norm(dq):.4f}')


# ─────────────────────────────────────────────────────────────────
#  2. EKF / Filtre de Kalman
# ─────────────────────────────────────────────────────────────────
def test_ekf():
    section('2. EKF / Filtre de Kalman Étendu (ekf.py)')
    from ur7e_line_follower.ekf import LaserDotEKF

    ekf = LaserDotEKF(dt=0.004, wall_x=1.0)
    HOME = np.array([-0.133, -1.5708, 1.5708, 0.0, 1.5708, 0.0])

    # T1 : non initialisé au départ
    record_test('EKF non initialisé avant reset',
         not ekf.initialized)

    # T2 : reset -> initialisé
    ekf.reset(y=0.05, z=0.60)
    record_test('EKF initialisé après reset',
         ekf.initialized,
         f'pos={ekf.position}')

    # T3 : position après reset cohérente
    pos = ekf.position
    record_test('Position EKF correcte après reset',
         abs(pos[0] - 0.05) < 1e-6 and abs(pos[1] - 0.60) < 1e-6,
         f'y={pos[0]:.4f} z={pos[1]:.4f}')

    # T4 : predict ne plante pas et conserve la position (~même vitesse nulle)
    ekf.predict(HOME, np.zeros(6))
    pos2 = ekf.position
    record_test('Predict (vitesse nulle) -> position stable',
         np.linalg.norm(pos2 - pos) < 0.01,
         f'dérive={np.linalg.norm(pos2 - pos)*1000:.2f}mm')

    # T5 : update FK réduit l'incertitude
    sigma_before = ekf.uncertainty.copy()
    ekf.update_fk(0.05, 0.60)
    sigma_after = ekf.uncertainty
    record_test('Update FK réduit sigma (covariance décroît)',
         np.all(sigma_after <= sigma_before + 1e-10),
         f'sigma_avant={sigma_before} sigma_après={sigma_after}')

    # T6 : un offset KLT relatif ne doit pas être injecté comme position absolue
    accepted_relative = ekf.update_camera(0.05, 0.60, offset_y=0.01, offset_z=-0.02)
    record_test('EKF refuse les offsets KLT comme mesure absolue',
         accepted_relative is False,
         'update_camera retourne False comme prévu')

    # T6b : une vraie position caméra métrique calibrée reste acceptée
    accepted_absolute = ekf.update_camera_absolute(0.051, 0.599)
    record_test('EKF accepte une mesure caméra absolue calibrée',
         accepted_absolute is True,
         f'pos={np.round(ekf.position, 4)}')

    # T7 : convergence sur 50 steps vers une mesure stable
    ekf2 = LaserDotEKF(dt=0.004)
    ekf2.reset(y=0.0, z=0.5)
    for _ in range(50):
        ekf2.predict(HOME, np.zeros(6))
        ekf2.update_fk(0.10, 0.65)   # vraie position
    pos_f = ekf2.position
    record_test('EKF converge vers la mesure en 50 steps',
         abs(pos_f[0] - 0.10) < 0.005 and abs(pos_f[1] - 0.65) < 0.005,
         f'y={pos_f[0]:.4f} (cible 0.10) z={pos_f[1]:.4f} (cible 0.65)')


# ─────────────────────────────────────────────────────────────────
#  3. LQR / Singularités
# ─────────────────────────────────────────────────────────────────
def test_lqr():
    section('3. LQR + Singularités (singularity.py)')
    from ur7e_line_follower.singularity import (
        yoshikawa, singular_values, manipulability_obs,
        lqr_gains, lqr_velocity_correction,
        null_space_manip_correction, singularity_penalty,
        check_known_singularities,
    )

    HOME       = np.array([-0.133, -1.5708, 1.5708, 0.0, 1.5708, 0.0])
    ELBOW_SING = np.array([-0.133, -1.5708, 0.0,   0.0, 1.5708, 0.0])  # q3~0 -> coude

    # T1 : Yoshikawa > 0 à HOME
    w = yoshikawa(HOME)
    record_test('Yoshikawa > 0 à HOME',
         w > 0,
         f'w={w:.5f}')

    # T2 : Yoshikawa proche de 0 près d'une singularité de coude
    w_sing = yoshikawa(ELBOW_SING)
    record_test('Yoshikawa < yoshikawa(HOME) près singularité coude',
         w_sing < w,
         f'w_sing={w_sing:.5f} vs w_home={w:.5f}')

    # T3 : valeurs singulières (3,) décroissantes
    sv = singular_values(HOME)
    record_test('Valeurs singulières (3,) décroissantes',
         sv.shape == (3,) and sv[0] >= sv[1] >= sv[2] - 1e-9,
         f'sv={np.round(sv, 4)}')

    # T4 : manipulability_obs retourne 3 scalaires in [0,1]
    obs = manipulability_obs(HOME)
    record_test('manipulability_obs in [0,1]^3',
         obs.shape == (3,) and np.all(obs >= 0) and np.all(obs <= 1.0),
         f'obs={np.round(obs, 4)}')

    # T5 : LQR gains > 0 et forme (6,)
    K = lqr_gains(HOME, w)
    record_test('LQR gains (6,) tous positifs',
         K.shape == (6,) and np.all(K > 0),
         f'K={np.round(K, 3)}')

    # T6 : LQR réduit la commande (K > 0 -> division par (1+K))
    dq_des = np.ones(6)
    dq_cmd = lqr_velocity_correction(HOME, dq_des, w)
    record_test('LQR réduit les vitesses articulaires désirées',
         np.all(np.abs(dq_cmd) <= np.abs(dq_des) + 1e-9),
         f'|dq_des|={np.linalg.norm(dq_des):.3f} -> |dq_cmd|={np.linalg.norm(dq_cmd):.3f}')

    # T7 : Correction noyau retourne (6,) sans NaN
    q_null = null_space_manip_correction(HOME)
    record_test('Correction noyau (null-space) retourne (6,) sans NaN',
         q_null.shape == (6,) and not np.any(np.isnan(q_null)),
         f'|q_null|={np.linalg.norm(q_null):.5f}')


# ─────────────────────────────────────────────────────────────────
#  4. Monte Carlo / Lignes cibles
# ─────────────────────────────────────────────────────────────────
def test_montecarlo():
    section('4. Monte Carlo / Lignes cibles (target_line.py)')
    from ur7e_line_follower.target_line import (
        s_curve, random_line, arc_length, distance_to_line,
        WALL_Y_MIN, WALL_Y_MAX, WALL_Z_MIN, WALL_Z_MAX,
    )

    # T1 : s_curve retourne (50, 2) dans les limites du mur
    sc = s_curve()
    ok = (sc.shape == (50, 2) and
          np.all(sc[:, 0] >= WALL_Y_MIN) and np.all(sc[:, 0] <= WALL_Y_MAX) and
          np.all(sc[:, 1] >= WALL_Z_MIN) and np.all(sc[:, 1] <= WALL_Z_MAX))
    record_test('s_curve (50,2) dans les limites du mur', ok,
         f'yin[{sc[:,0].min():.2f},{sc[:,0].max():.2f}] zin[{sc[:,1].min():.2f},{sc[:,1].max():.2f}]')

    # T2 : longueur d'arc s_curve >= 50 cm
    length = arc_length(sc)
    record_test("s_curve longueur d'arc >= 50 cm",
         length >= 0.50,
         f'L={length*100:.1f}cm')

    # T3 : 10 tirages Monte Carlo de random_line — toutes valides
    rng = np.random.default_rng(42)
    ok_count = 0
    for _ in range(10):
        rl = random_line(rng=rng)
        l  = arc_length(rl)
        in_bounds = (np.all(rl[:, 0] >= WALL_Y_MIN) and
                     np.all(rl[:, 0] <= WALL_Y_MAX) and
                     np.all(rl[:, 1] >= WALL_Z_MIN) and
                     np.all(rl[:, 1] <= WALL_Z_MAX))
        if l >= 0.50 and in_bounds:
            ok_count += 1
    record_test(f'Monte Carlo 10x random_line : {ok_count}/10 valides (L>=50cm, dans mur)',
         ok_count == 10,
         f'{ok_count}/10')

    # T4 : distance_to_line nul sur la ligne elle-même
    pt_on_line = sc[25]     # waypoint exact
    d = distance_to_line(pt_on_line, sc)
    record_test('distance_to_line ~ 0 pour un point sur la ligne',
         d < 1e-4,
         f'd={d*1000:.3f}mm')

    # T5 : distance_to_line > 0 pour un point hors de toute la polyline
    # On prend un coin du mur qui est très loin de la s_curve
    pt_off = np.array([WALL_Y_MAX - 0.01, WALL_Z_MIN + 0.01])
    d2 = distance_to_line(pt_off, sc)
    record_test('distance_to_line > 0 pour un point loin de la ligne',
         d2 > 0.05,
         f'd={d2*100:.2f}cm (attendu >> 5cm)')

    # T6 : diversité Monte Carlo — toutes les lignes différentes
    lines = [random_line(rng=np.random.default_rng(i)) for i in range(6)]
    unique = all(not np.allclose(lines[i], lines[j])
                 for i in range(6) for j in range(i+1, 6))
    record_test('Monte Carlo 6 lignes distinctes (seed différentes)',
         unique)


# ─────────────────────────────────────────────────────────────────
#  5. RL / Inférence modèle SAC
# ─────────────────────────────────────────────────────────────────
def test_rl(ckpt_dir=None):
    section('5. RL / Inférence modèle SAC (stable-baselines3)')
    from pathlib import Path
    from stable_baselines3 import SAC

    if ckpt_dir is None:
        ckpt_dir = Path.home() / '.ros' / 'ur7e_line_follower' / 'checkpoints'

    # Trouver le dernier checkpoint
    zips = sorted(Path(ckpt_dir).glob('*.zip'))
    if not zips:
        print(f'  {WARN} Aucun checkpoint dans {ckpt_dir} — tests RL sautés')
        return

    # Chercher un checkpoint V2 (avec .meta.json)
    import json
    v2_ckpts = [z for z in zips
                if Path(str(z).removesuffix('.zip') + '.meta.json').exists()]
    if not v2_ckpts:
        print(f'  {WARN} Aucun checkpoint V2 (meta.json) dans {ckpt_dir}')
        print(f'  {WARN} Les checkpoints existants sont V1 (28D) — non compatibles V2. Tests RL sautés.')
        record_test('Checkpoint V2 disponible',
             False, 'Aucun checkpoint V2 — réentraîner après migration')
        return

    ckpt = v2_ckpts[-1]
    print(f'  Checkpoint V2.2 : {ckpt.name}')

    # T1 : chargement sans erreur
    try:
        model = SAC.load(str(ckpt))
        ok = True
        detail = 'OK'
    except Exception as e:
        ok = False
        detail = str(e)
    record_test('Chargement du checkpoint SAC V2', ok, detail)
    if not ok:
        return

    # T2 : espace d'observation = (29,) schéma V3
    obs_space = model.observation_space
    record_test('Espace observation = Box(29,)',
         obs_space.shape == (29,),
         f'shape={obs_space.shape}')

    # T3 : espace d'action = (2,) cartésien mur dans [-1, 1]
    act_space = model.action_space
    record_test('Espace action = Box(2,) dans [-1,1]',
         act_space.shape == (2,) and
         np.allclose(act_space.low, -1) and np.allclose(act_space.high, 1),
         f'shape={act_space.shape} low={act_space.low[0]:.1f} high={act_space.high[0]:.1f}')

    # T4 : inférence déterministe sur obs fictive -> action (2,) in [-1,1]
    dummy_obs = np.zeros(29, dtype=np.float32)
    dummy_obs[11] = 1.0   # on_wall = True
    action, _ = model.predict(dummy_obs, deterministic=True)
    record_test('Inférence déterministe -> action (2,) in [-1,1]',
         action.shape == (2,) and np.all(action >= -1) and np.all(action <= 1),
         f'action={np.round(action, 3)}')

    # T5 : 5 inférences stochastiques -> dispersion non nulle
    actions = np.array([model.predict(dummy_obs, deterministic=False)[0]
                        for _ in range(5)])
    std_actions = np.std(actions, axis=0)
    record_test('Inférence stochastique 5x -> variance non nulle',
         np.any(std_actions > 1e-6),
         f'std_mean={std_actions.mean():.4f}')


# ─────────────────────────────────────────────────────────────────
#  6. KLT / Caméra
# ─────────────────────────────────────────────────────────────────
def _make_frame(W, H, line_v, laser_u, laser_v, line_color=(200, 30, 10)):
    """Construit une image BGR synthétique avec une ligne bleue horizontale et un laser rouge."""
    import cv2
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    for x in range(20, W - 20):
        cv2.circle(frame, (x, line_v), 2, line_color, -1)
    cv2.circle(frame, (laser_u, laser_v), 5, (0, 0, 255), -1)
    return frame


def test_klt():
    section('6. KLT / Caméra (camera_line_detector.py)')
    import cv2
    from ur7e_line_follower.camera_line_detector import (
        CameraLineDetector, W, H, MIN_TRACK_POINTS, LK_PARAMS,
        BLUE_HSV_LO, BLUE_HSV_HI, RED_HSV_LO1, RED_HSV_HI1,
        RED_HSV_LO2, RED_HSV_HI2,
    )

    LINE_V   = H // 2          # ligne à v=120
    LASER1_U = W // 3          # laser frame1 à u=106, sur la ligne
    LASER2_U = W // 3 + 20    # laser frame2 décalé de 20px à droite

    # ── Frame 1 : ligne + laser à (LASER1_U, LINE_V) ─────────────────
    frame1 = _make_frame(W, H, LINE_V, LASER1_U, LINE_V)

    # ── T1 : masque HSV bleu détecte la ligne ─────────────────────────
    hsv1      = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    blue_mask1 = cv2.inRange(hsv1, BLUE_HSV_LO, np.array([140,255,255],dtype=np.uint8))
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    blue_mask1 = cv2.morphologyEx(blue_mask1, cv2.MORPH_OPEN, kern)
    blue_mask1 = cv2.dilate(blue_mask1, kern)
    blue_px1 = int(np.count_nonzero(blue_mask1))
    record_test('T1 Masque HSV bleu detecte la ligne synthetique',
         blue_px1 > 200, f'{blue_px1} pixels bleus')

    # ── T2 : masque rouge detecte le laser ────────────────────────────
    red1_ = cv2.inRange(hsv1, RED_HSV_LO1, RED_HSV_HI1)
    red2_ = cv2.inRange(hsv1, RED_HSV_LO2, RED_HSV_HI2)
    red_mask1 = cv2.bitwise_or(red1_, red2_)
    record_test('T2 Masque HSV rouge detecte le laser',
         int(np.count_nonzero(red_mask1)) > 5,
         f'{int(np.count_nonzero(red_mask1))} pixels rouges')

    # ── T3 : _find_laser retourne centroide proche de la vraie position ─
    det = CameraLineDetector.__new__(CameraLineDetector)
    det._init_state()

    laser_uv, laser_vis = det._find_laser(red_mask1)
    record_test('T3 _find_laser : laser visible et centroide valide',
         laser_vis and laser_uv is not None,
         f'uv={laser_uv}')
    if laser_uv is not None:
        err_centroid = np.linalg.norm(laser_uv - np.array([LASER1_U, LINE_V], dtype=np.float32))
        record_test('T3b Centroide laser a moins de 3px de la position reelle',
             err_centroid < 3.0,
             f'erreur={err_centroid:.2f}px (laser reel=({LASER1_U},{LINE_V}))')

    # ── T4 : _compute_offset avec pts2d construit analytiquement ──────
    # On construit une ligne horizontale parfaite : pts2d[i] = (i*5, LINE_V)
    # Le laser est placé EXACTEMENT sur l'un de ces points.
    # Ainsi nearest == laser et off_normal doit être == 0.0 exactement.
    pts_analytic = np.array([[float(u), float(LINE_V)]
                              for u in range(10, W - 10, 5)], dtype=np.float32)
    laser_exact  = np.array([float(pts_analytic[10, 0]), float(LINE_V)], dtype=np.float32)

    off_n, off_l, angle_deg = det._compute_offset(laser_exact, pts_analytic)
    # Valeurs théoriques :
    #   tangente ACP = [1, 0] (ligne horizontale pure)
    #   normale      = [0, 1]
    #   delta = laser - nearest = [0, 0]  car laser_exact EST dans pts_analytic
    #   off_normal = dot([0,0],[0,1]) = 0.0
    #   off_longit = dot([0,0],[1,0]) = 0.0
    #   angle_deg  = atan2(0,1) = 0° ou 180° selon signe tangente
    record_test('T4 _compute_offset : offset normal = 0 (laser exact sur pts)',
         abs(off_n) < 1e-3,
         f'off_normal={off_n:.6f}px (theorique=0.0)')
    record_test('T4b _compute_offset : angle tangente ~0° ou ~180° (ligne horiz)',
         abs(abs(angle_deg) - 0.0) < 1.0 or abs(abs(angle_deg) - 180.0) < 1.0,
         f'angle={angle_deg:.4f}deg')

    # ── T5 : _compute_offset avec laser a 10px au-dessus de la ligne ──
    # On place le laser 10px au-dessus (v = LINE_V - 10).
    # Theorique : off_normal = -10px (sens normal vers le bas = [0,+1]),
    #             donc dot([-10 en v] -> delta=[0,-10], normal=[0,1]) = -10.0
    laser_above = np.array([float(pts_analytic[10, 0]), float(LINE_V - 10)], dtype=np.float32)
    off_n2, off_l2, _ = det._compute_offset(laser_above, pts_analytic)
    # nearest = point le plus proche dans pts_analytic. La ligne est a v=LINE_V.
    # Le point le plus proche de (u, LINE_V-10) est (u, LINE_V) -> dist=10px.
    # delta = laser_above - nearest = [0, -10]
    # normal tangente horiz = [0, +1] ou [0, -1] selon signe stabilisation
    # off_n = dot([0,-10], [0,±1]) = ±10
    record_test('T5 _compute_offset : |off_normal| = 10px pour laser a 10px de la ligne',
         abs(abs(off_n2) - 10.0) < 1.5,
         f'|off_normal|={abs(off_n2):.3f}px (theorique=10.0)')

    # ── T6 : bout-en-bout via process_frame() ────────────────────────
    # Frame 1 : ligne a LINE_V, laser a LASER1_U
    det2 = CameraLineDetector.__new__(CameraLineDetector)
    det2._init_state()
    r1 = det2.process_frame(frame1)
    record_test('T6a Frame1 : ligne detectee',
         r1.line_detected, f'{r1.blue_px} pixels bleus')
    record_test('T6b Frame1 : laser visible',
         r1.laser_visible, f'uv={r1.laser_uv}')
    n_pts_frame1 = len(r1.line_pts) if r1.line_pts is not None else 0
    record_test('T6c Frame1 : >= MIN_TRACK_POINTS points initialises',
         n_pts_frame1 >= MIN_TRACK_POINTS, f'{n_pts_frame1} points')

    # Frame 2 : meme ligne, laser decale de +20px en u (deplacement connu)
    frame2 = _make_frame(W, H, LINE_V, LASER2_U, LINE_V)
    r2 = det2.process_frame(frame2)
    record_test('T6d Frame2 : ligne toujours detectee apres deplacement laser',
         r2.line_detected, f'{r2.blue_px} pixels bleus')
    record_test('T6e Frame2 : laser visible a nouvelle position',
         r2.laser_visible, f'uv={r2.laser_uv}')

    # Verifier que le centroide laser a bien suivi le deplacement (+20px en u)
    if r2.laser_uv is not None:
        delta_u = float(r2.laser_uv[0]) - LASER1_U
        record_test('T6f Frame2 : centroide laser decale de ~20px en u',
             abs(delta_u - 20.0) < 3.0,
             f'delta_u={delta_u:.1f}px (attendu=+20px)')

    # T7 : detection_vector (7 valeurs) normalise dans [-1,1]
    dv = r2.detection_vector
    record_test('T7 detection_vector longueur 7 (schema V3)',
         len(dv) == 7,
         f'len={len(dv)} val={[round(x,3) for x in dv]}')
    record_test('T7b offset_n_norm in [-1,1]',
         -1.0 <= dv[1] <= 1.0,
         f'offset_n={dv[1]:.4f}')


# ─────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-rl', action='store_true',
                        help='Sauter les tests RL (pas de checkpoint)')
    args = parser.parse_args()

    print('\n' + '═' * 60)
    print('  TESTS COMPOSANTS UR7e Line Follower')
    print('═' * 60)

    suites = [
        ('Cinématique MGI', test_kinematics),
        ('EKF Kalman',      test_ekf),
        ('LQR Singularités', test_lqr),
        ('Monte Carlo',     test_montecarlo),
        ('KLT Caméra',      test_klt),
    ]
    if not args.skip_rl:
        suites.insert(4, ('RL SAC', test_rl))

    for name, fn in suites:
        try:
            fn()
        except Exception:
            print(f'\n  {FAIL} EXCEPTION dans {name} :')
            traceback.print_exc()

    # ── Résumé ───────────────────────────────────────────────────
    print(f'\n{"═"*60}')
    print('  RÉSUMÉ')
    print(f'{"═"*60}')
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    total  = len(results)
    print(f'  Total : {total} tests  |  {PASS} {passed}  |  {FAIL} {failed}')

    if failed:
        print('\n  Tests ÉCHOUÉS :')
        for name, ok, detail in results:
            if not ok:
                print(f'    • {name}')
                if detail:
                    print(f'      {detail}')
    print()
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
