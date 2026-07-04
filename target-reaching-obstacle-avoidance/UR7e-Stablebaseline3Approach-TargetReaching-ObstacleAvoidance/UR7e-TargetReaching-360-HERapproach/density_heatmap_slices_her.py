# ==============================================================================
# FICHIER : density_heatmap_slices_her.py
# RÔLE : Density heatmaps DYNAMIQUES par tranches du modèle SAC+HER 360°.
#        Deux vues (esthétique "Vega density heatmap", colormap viridis) :
#          - plan XY : un curseur balaie l'altitude z (sections horizontales)
#          - plan XZ : un curseur balaie y (sections verticales)
#        La grandeur affichée = distance finale moyenne locale (mm).
#        Échelle : 50 mm (5 cm) = excellent (vert) -> 150 mm (15 cm) = mauvais.
#
# MASQUAGE STRICT : aucune donnée affichée hors de l'espace atteignable
#   (singularité / hors de portée : trop près ou trop loin du cobot). Un pixel
#   n'est coloré que si une cible IK-valide existe dans son voisinage.
#
# CIBLES : 5000 points en grille HOMOGÈNE dans la boîte de travail, filtrés
#   par atteignabilité IK exacte. Départs ALÉATOIRES non singuliers (hérités
#   du moteur via reset). Évaluation avec le modèle HER (wrapper goal-cond.).
#
# Sortie : resultats_xp/figures_360/density_heatmap_slices_her.html (autonome).
# Usage : pip install plotly scipy ; python density_heatmap_slices_her.py
# ==============================================================================

import os
import json
import numpy as np
from stable_baselines3 import SAC
from ur7e_wrapper_her import UR7eReachHEREnv, SUCCESS_RADIUS

# ============================= PARAMÈTRES ====================================
MODEL_PATH = "sac_ur7e_her_360_reach"
N_TARGETS = 5000                 # cibles homogènes visées (après filtrage IK)
SEED = 4242
MAX_STEPS = 300
N_SLICES = 15                    # tranches par axe (résolution des curseurs)
GRID_XY = 120                    # résolution spatiale d'une tranche (champ continu fin)
SIGMA = 0.07                     # rayon de moyennation gaussienne (m) — lissage continu
MASK_RADIUS = 0.09               # rayon max cible<->pixel pour qu'un pixel soit "couvert"
SLICE_HALF_FACTOR = 1.6          # demi-épaisseur de tranche élargie (recouvrement -> continuité)
DIST_GOOD = 50.0                 # mm : borne basse échelle couleur (vert)
DIST_BAD = 150.0                 # mm : borne haute échelle couleur (rouge)
OUT_HTML = os.path.join("resultats_xp", "figures_360",
                        "density_heatmap_slices_her.html")
# boîte de travail symétrique (360°), avant filtrage IK
X0, X1 = -0.85, 0.85
Y0, Y1 = -0.85, 0.85
Z0, Z1 = 0.0, 1.2
# =============================================================================


def homogeneous_reachable_targets(env, n_target, seed):
    """
    Génère EXACTEMENT n_target cibles ATTEIGNABLES, réparties de façon HOMOGÈNE
    (grille régulière jittered) dans la boîte de travail. On construit une grille
    homogène volontairement sur-échantillonnée (beaucoup de points seront rejetés
    car hors de la coquille atteignable), on la mélange pour que l'acceptation
    reste spatialement homogène, puis on accepte jusqu'à atteindre n_target.
    Validation par IK exacte (PyBullet renvoie toujours une solution : on
    l'applique et on vérifie la position réelle de l'effecteur).
    """
    import pybullet as p
    rng = np.random.RandomState(seed)
    engine = env.engine
    robot, ee_index = engine.robot, engine.ee_index
    joints = engine.CONTROLLABLE_JOINTS

    # Grille homogène sur-échantillonnée. Le taux d'acceptation (coquille / boîte)
    # est ~25-35% ; on vise donc une grille ~4x plus dense que n_target pour avoir
    # une marge confortable, quitte à la regénérer plus dense si insuffisant.
    def make_grid(n_lin_factor):
        vol = (X1 - X0) * (Y1 - Y0) * (Z1 - Z0)
        n_lin = int(np.ceil((n_target * n_lin_factor / vol) ** (1 / 3)))
        xs = np.linspace(X0, X1, n_lin)
        ys = np.linspace(Y0, Y1, n_lin)
        nz = max(4, int(n_lin * (Z1 - Z0) / (X1 - X0)))
        zs = np.linspace(Z0, Z1, nz)
        step = xs[1] - xs[0] if len(xs) > 1 else 0.1
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        cells = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
        # jitter homogène dans chaque cellule -> cadrillage uniforme, non rigide
        cells = cells + rng.uniform(-0.5, 0.5, cells.shape) * step
        rng.shuffle(cells)   # mélange : l'acceptation reste spatialement homogène
        return cells

    accepted = []
    tested = 0
    factor = 6
    while len(accepted) < n_target:
        cells = make_grid(factor)
        for cand in cells:
            if len(accepted) >= n_target:
                break
            tested += 1
            sol = p.calculateInverseKinematics(robot, ee_index, cand.tolist())
            for j, q in zip(joints, sol):
                p.resetJointState(robot, j, q)
            ee = np.array(p.getLinkState(robot, ee_index)[0])
            if np.linalg.norm(ee - cand) < 0.005:     # atteignable à 5 mm
                accepted.append(cand.copy())
        if len(accepted) < n_target:
            factor *= 2          # grille insuffisante -> on densifie et on recommence
            print(f"  ... {len(accepted)}/{n_target} acceptées, densification de la grille")

    for j in joints:
        p.resetJointState(robot, j, 0.0)
    accepted = np.array(accepted[:n_target])
    print(f"  -> {len(accepted)} cibles atteignables homogènes "
          f"({tested} candidates testées, taux {100*len(accepted)/max(tested,1):.0f}%).")
    return accepted


def run_episode(model, env, target):
    """Joue un épisode (départ aléatoire non singulier hérité du reset)."""
    obs, _ = env.reset()
    env.engine.target = np.array(target, dtype=np.float64)
    obs = env._make_obs()
    done = False
    info = {}
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        done = term or trunc
    # distance finale réelle effecteur<->cible
    return float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))


def gaussian_slice(points2d, values, gx, gy, sigma, mask_radius):
    """
    Champ continu lissé sur une grille 2D à partir de points épars (d'UNE tranche).
    Renvoie (field, mask) : field=valeur moyennée, mask=True si pixel COUVERT
    (au moins une cible proche) -> sinon NaN (zone non atteignable, transparente).
    """
    from scipy.spatial import cKDTree
    nodes = np.column_stack([gx.ravel(), gy.ravel()])
    field = np.full(len(nodes), np.nan)
    if len(points2d) == 0:
        return field.reshape(gx.shape)
    tree = cKDTree(points2d)
    # pixels couverts : au moins une cible dans mask_radius
    covered = tree.query_ball_point(nodes, r=mask_radius)
    neigh = tree.query_ball_point(nodes, r=3 * sigma)
    for i, (cov, idx) in enumerate(zip(covered, neigh)):
        if not cov or not idx:
            continue   # pixel non atteignable -> reste NaN (transparent)
        d2 = np.sum((points2d[idx] - nodes[i])**2, axis=1)
        w = np.exp(-d2 / (2 * sigma**2))
        field[i] = np.sum(w * values[idx]) / np.sum(w)
    return field.reshape(gx.shape)


def build_slices(targets, dist_mm, axis, slice_centers, half):
    """
    Construit N_SLICES tranches le long de `axis` (2=z pour XY, 1=y pour XZ).
    Chaque tranche agrège les cibles dont la coord `axis` tombe dans [c-half, c+half].
    Retourne (liste de grilles 2D, axes a, b).
    """
    if axis == 2:      # plan XY (balayage z) : a=x, b=y
        ia, ib = 0, 1
        a0, a1, b0, b1 = X0, X1, Y0, Y1
    else:              # plan XZ (balayage y) : a=x, b=z
        ia, ib = 0, 2
        a0, a1, b0, b1 = X0, X1, Z0, Z1

    av = np.linspace(a0, a1, GRID_XY)
    bv = np.linspace(b0, b1, GRID_XY)
    ga, gb = np.meshgrid(av, bv, indexing="ij")

    grids = []
    for c in slice_centers:
        # bande élargie (recouvrante) : assez de cibles par tranche -> champ plein
        sel = np.abs(targets[:, axis] - c) <= half * SLICE_HALF_FACTOR
        pts = targets[sel][:, [ia, ib]]
        vals = dist_mm[sel]
        field = gaussian_slice(pts, vals, ga, gb, SIGMA, MASK_RADIUS)
        grids.append(field)
    return grids, av, bv


def main():
    try:
        import plotly.graph_objects as go
        from plotly.offline import plot
    except ImportError:
        print("plotly requis : pip install plotly")
        return

    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    print(f"Chargement du modèle {MODEL_PATH}...")
    env = UR7eReachHEREnv(render_mode=None, max_episode_len=MAX_STEPS,
                          random_start=True, seed=SEED)
    model = SAC.load(MODEL_PATH, env=env, device="cpu")  # env requis (HER)

    print(f"Génération de ~{N_TARGETS} cibles homogènes atteignables (IK)...")
    targets = homogeneous_reachable_targets(env, N_TARGETS, SEED)

    print(f"Évaluation du modèle HER sur {len(targets)} cibles...")
    import time
    t_start = time.time()
    n_total = len(targets)
    dists = np.empty(n_total)
    for i, t in enumerate(targets):
        dists[i] = run_episode(model, env, t)
        # état d'avancement actualisé sur place (toutes les 25 cibles)
        if (i + 1) % 25 == 0 or (i + 1) == n_total:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (n_total - (i + 1)) / rate if rate > 0 else 0
            print(f"\r  {i+1}/{n_total} cibles  "
                  f"({100*(i+1)/n_total:.0f}%)  |  "
                  f"{rate:.1f} cibles/s  |  écoulé {elapsed:.0f}s  "
                  f"ETA {eta:.0f}s   ", end="", flush=True)
    print()   # saut de ligne final après la barre de progression
    env.close()
    dist_mm = dists * 1000.0
    print(f"  distance finale : médiane {np.median(dist_mm):.0f} mm | "
          f"<50mm: {(dist_mm<50).mean()*100:.0f}% | <150mm: {(dist_mm<150).mean()*100:.0f}%")

    # --- tranches XY (balayage z) et XZ (balayage y) ---
    half_z = (Z1 - Z0) / (2 * N_SLICES)
    half_y = (Y1 - Y0) / (2 * N_SLICES)
    z_centers = np.linspace(Z0 + half_z, Z1 - half_z, N_SLICES)
    y_centers = np.linspace(Y0 + half_y, Y1 - half_y, N_SLICES)

    print("Construction des tranches XY (balayage z)...")
    xy_grids, xax, yax = build_slices(targets, dist_mm, 2, z_centers, half_z)
    print("Construction des tranches XZ (balayage y)...")
    xz_grids, xax2, zax = build_slices(targets, dist_mm, 1, y_centers, half_y)

    # --- Figure Plotly : 2 sous-graphes + 2 sliders indépendants ---
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Plan XY — section à z fixé",
                                        "Plan XZ — section à y fixé"),
                        horizontal_spacing=0.12)

    # Palette Viridis NON inversée (comme la capture Vega : jaune = "haut").
    # Pour que JAUNE = BIEN (faible distance) et VIOLET = MAUVAIS (grande
    # distance), on affiche la "qualité" = DIST_BAD - distance, de sorte que
    # les bonnes cibles (faible distance) prennent la valeur haute -> jaune.
    # La colorbar est ré-étiquetée en mm de distance réelle pour rester lisible.
    cs = "Viridis"
    qmin, qmax = 0.0, (DIST_BAD - DIST_GOOD)      # qualité in [0, 100] mm
    # ticks colorbar : on affiche la distance réelle correspondante
    tick_dists = [DIST_GOOD, 75, 100, 125, DIST_BAD]            # mm
    tick_vals = [DIST_BAD - d for d in tick_dists]              # position sur l'échelle qualité
    tick_text = [f"≤{int(DIST_GOOD)}" if d == DIST_GOOD else
                 (f"≥{int(DIST_BAD)}" if d == DIST_BAD else f"{int(d)}")
                 for d in tick_dists]

    def quality(g):
        # NaN reste NaN (zones non atteignables -> transparentes)
        return DIST_BAD - g

    # heatmaps XY (une trace par tranche, visible une à la fois)
    for k, g in enumerate(xy_grids):
        fig.add_trace(go.Heatmap(
            z=quality(g).T, x=xax, y=yax, colorscale=cs, zmin=qmin, zmax=qmax,
            zsmooth="best", visible=(k == N_SLICES // 2),
            colorbar=dict(title="dist. moy.<br>locale (mm)", x=0.46, len=0.9,
                          tickmode="array", tickvals=tick_vals, ticktext=tick_text),
            customdata=g.T,
            hovertemplate="x=%{x:.2f}  y=%{y:.2f}<br>%{customdata:.0f} mm<extra></extra>",
            name=f"z={z_centers[k]:.2f}"), row=1, col=1)
    # heatmaps XZ
    for k, g in enumerate(xz_grids):
        fig.add_trace(go.Heatmap(
            z=quality(g).T, x=xax2, y=zax, colorscale=cs, zmin=qmin, zmax=qmax,
            zsmooth="best", visible=(k == N_SLICES // 2), showscale=False,
            customdata=g.T,
            hovertemplate="x=%{x:.2f}  z=%{y:.2f}<br>%{customdata:.0f} mm<extra></extra>",
            name=f"y={y_centers[k]:.2f}"), row=1, col=2)

    n = N_SLICES
    # steps du slider XY : n'agit que sur les traces XY (indices 0..n-1)
    xy_steps = []
    for k in range(n):
        vis = [False] * (2 * n)
        vis[k] = True                      # la tranche XY k
        vis[n + (n // 2)] = True           # garde la tranche XZ centrale visible
        xy_steps.append(dict(method="update",
                             args=[{"visible": vis}],
                             label=f"{z_centers[k]:.2f}"))
    xz_steps = []
    for k in range(n):
        vis = [False] * (2 * n)
        vis[n // 2] = True                 # garde la tranche XY centrale visible
        vis[n + k] = True                  # la tranche XZ k
        xz_steps.append(dict(method="update",
                             args=[{"visible": vis}],
                             label=f"{y_centers[k]:.2f}"))

    fig.update_layout(
        title="Density heatmaps dynamiques — SAC+HER 360° "
              "(distance finale moyenne, zones non atteignables masquées)",
        sliders=[
            dict(active=n // 2, pad={"t": 30}, x=0.0, len=0.46,
                 currentvalue={"prefix": "z = ", "suffix": " m"},
                 steps=xy_steps),
            dict(active=n // 2, pad={"t": 30}, x=0.54, len=0.46,
                 currentvalue={"prefix": "y = ", "suffix": " m"},
                 steps=xz_steps),
        ],
        margin=dict(l=40, r=40, t=80, b=40),
        plot_bgcolor="white",
    )
    # repères orthonormés + axes propres
    fig.update_xaxes(title_text="x (m)", row=1, col=1, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(title_text="y (m)", row=1, col=1)
    fig.update_xaxes(title_text="x (m)", row=1, col=2, scaleanchor="y2", scaleratio=1)
    fig.update_yaxes(title_text="z (m)", row=1, col=2)

    plot(fig, filename=OUT_HTML, auto_open=False, include_plotlyjs="cdn")
    print(f"\nFigure sauvegardée : {OUT_HTML}")
    print("Deux curseurs : gauche balaie z (vues XY), droite balaie y (vues XZ).")


if __name__ == "__main__":
    main()
