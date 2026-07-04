# ==============================================================================
# FICHIER : density_3d_volume_360.py
# RÔLE : "3D heatmap density plot" CONTINU du modèle SAC ÉTALON 360°.
#        Volume 3D rempli (rendu volumétrique translucide) obtenu en moyennant
#        le voisinage de N cibles atteignables réparties sur TOUT l'espace
#        (360° autour de la base), AVEC le nuage de points superposé.
#
# Différences avec density_3d_volume.py (version "avant uniquement") :
#   1. Pointe sur le modèle 360° (sac_ur7e_360_reach) et le wrapper 360.
#   2. Échantillonne les cibles sur TOUT l'espace (x,y ∈ [-0.85, 0.85]),
#      cohérent avec le domaine d'entraînement ; validation IK exacte.
#   3. Pose de départ FIXE au test (random_start=False) -> reproductible.
#   4. Échelle de couleur calée sur le MIN et le MAX RÉELS des données
#      (et non sur des percentiles).
#
# Réutilise le modèle 360° entraîné : AUCUN réentraînement.
# Usage : pip install plotly scipy ; python density_3d_volume_360.py
# ==============================================================================

import os
import numpy as np
from stable_baselines3 import SAC
from ur7e_wrapper_360 import UR7eReach360Env

# ============================= PARAMÈTRES ====================================
MODEL_PATH = "sac_ur7e_360_reach"   # modèle étalon 360°
N_CIBLES = 3000                     # cibles atteignables visées
SEED_CIBLES = 99999
MAX_STEPS = 300
GRID = 32                           # résolution de la grille (32^3 ≈ 33k nœuds)
SIGMA = 0.06                        # rayon de moyennation gaussienne (m)
IK_TOL = 0.005                      # tolérance (m) pour juger une cible atteignable
OUT_HTML = os.path.join("resultats_xp", "figures_360",
                        "density_3d_volume_sac_360.html")
# boîte d'échantillonnage SYMÉTRIQUE (360° autour de la base) — avant filtrage IK
X0, X1 = -0.85, 0.85
Y0, Y1 = -0.85, 0.85
Z0, Z1 = 0.0, 1.2
# =============================================================================


def reachable_targets(env, n, seed, tol=IK_TOL):
    """
    Génère n cibles ATTEIGNABLES sur TOUT l'espace (360°), via vérification IK
    exacte. PyBullet renvoie toujours une solution IK même hors d'atteinte, donc
    on applique la solution et on vérifie la position réelle de l'effecteur.
    """
    import pybullet as p
    rng = np.random.RandomState(seed)
    engine = env.engine
    robot = engine.robot
    ee_index = engine.ee_index
    joints = engine.CONTROLLABLE_JOINTS

    accepted = []
    n_tested = 0
    while len(accepted) < n:
        cand = np.array([rng.uniform(X0, X1),
                         rng.uniform(Y0, Y1),
                         rng.uniform(Z0, Z1)])
        n_tested += 1
        sol = p.calculateInverseKinematics(robot, ee_index, cand.tolist())
        for j, q in zip(joints, sol):
            p.resetJointState(robot, j, q)
        ee = np.array(p.getLinkState(robot, ee_index)[0])
        if np.linalg.norm(ee - cand) < tol:
            accepted.append(cand)
        if n_tested % 5000 == 0:
            print(f"  ... {len(accepted)}/{n} cibles atteignables "
                  f"({n_tested} testées, taux {100*len(accepted)/n_tested:.0f}%)")
    for j in joints:
        p.resetJointState(robot, j, 0.0)
    print(f"  -> {n} cibles atteignables retenues sur {n_tested} candidates "
          f"({100*n/n_tested:.0f}% d'acceptation).")
    return np.array(accepted)


def run_episode(model, env, target):
    obs, _ = env.reset()                       # pose de départ FIXE (random_start=False)
    env.engine.target = np.array(target, dtype=np.float64)
    obs = env.engine._get_observation()
    done = False
    info = {}
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        done = term or trunc
    return info["distance"]


def gaussian_field(points, values, gx, gy, gz, sigma):
    """Moyennation gaussienne : champ continu lissé à partir des points épars."""
    from scipy.spatial import cKDTree
    nodes = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    tree = cKDTree(points)
    radius = 3 * sigma
    field = np.full(len(nodes), np.nan)
    neighbors = tree.query_ball_point(nodes, r=radius)
    for i, idx in enumerate(neighbors):
        if not idx:
            continue
        d2 = np.sum((points[idx] - nodes[i])**2, axis=1)
        w = np.exp(-d2 / (2 * sigma**2))
        field[i] = np.sum(w * values[idx]) / np.sum(w)
    return field.reshape(gx.shape)


def main():
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly requis : pip install plotly")
        return

    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    print(f"Chargement du modèle {MODEL_PATH}...")
    model = SAC.load(MODEL_PATH, device="cpu")
    # IMPORTANT : random_start=False -> pose de départ FIXE au test (reproductible)
    env = UR7eReach360Env(render_mode=None, max_episode_len=MAX_STEPS,
                          success_threshold=0.005, random_start=False,
                          seed=SEED_CIBLES)

    print(f"Génération de {N_CIBLES} cibles atteignables sur 360° (IK)...")
    targets = reachable_targets(env, N_CIBLES, SEED_CIBLES)

    print(f"Évaluation sur {len(targets)} cibles (départ fixe)...")
    dists = np.array([run_episode(model, env, t) for t in targets])
    env.close()

    # Champ continu : distance finale moyenne locale (mm).
    dist_mm = dists * 1000.0

    print(f"Construction du volume continu ({GRID}^3 nœuds, σ={SIGMA} m)...")
    gx0, gx1 = targets[:, 0].min(), targets[:, 0].max()
    gy0, gy1 = targets[:, 1].min(), targets[:, 1].max()
    gz0, gz1 = targets[:, 2].min(), targets[:, 2].max()
    xs = np.linspace(gx0, gx1, GRID)
    ys = np.linspace(gy0, gy1, GRID)
    zs = np.linspace(gz0, gz1, GRID)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    field = gaussian_field(targets, dist_mm, gx, gy, gz, SIGMA)

    # Échelle de couleur calée sur le MIN et le MAX RÉELS des données
    # (du champ interpolé). Pas de percentile : on prend les extrêmes observés.
    valid = field[~np.isnan(field)]
    cmin = float(valid.min())
    cmax = float(valid.max())
    print(f"Échelle couleur (min/max réels du champ) : "
          f"{cmin:.1f} mm -> {cmax:.1f} mm")

    # --- Volume continu (translucide) ---
    fig = go.Figure()
    fig.add_trace(go.Volume(
        x=gx.ravel(), y=gy.ravel(), z=gz.ravel(),
        value=np.nan_to_num(field, nan=cmax).ravel(),
        isomin=cmin, isomax=cmax,
        opacity=0.12,
        surface_count=18,
        colorscale="RdYlGn_r",        # vert = précis (faible dist), rouge = imprécis
        colorbar=dict(title="Distance<br>moyenne<br>locale (mm)"),
        caps=dict(x_show=False, y_show=False, z_show=False),
        name="volume continu",
    ))

    # --- Nuage de points par-dessus (même échelle min/max réels) ---
    dmin = float(dist_mm.min())
    dmax = float(dist_mm.max())
    fig.add_trace(go.Scatter3d(
        x=targets[:, 0], y=targets[:, 1], z=targets[:, 2], mode="markers",
        name="cibles (points)",
        marker=dict(size=3.0, color=dist_mm, colorscale="RdYlGn_r",
                    cmin=dmin, cmax=dmax, opacity=0.85, line=dict(width=0)),
        text=[f"{d:.1f} mm" for d in dist_mm], hoverinfo="text",
    ))

    fig.update_layout(
        title="Density 3D continu — SAC étalon 360° (espace complet)<br>"
              "<sub>Volume interpolé (moyennation gaussienne) + nuage de cibles. "
              "Départ fixe. Échelle couleur = min/max réels. "
              "Tournez à la souris ; le slider balaie les iso-niveaux.</sub>",
        scene=dict(xaxis_title="x (m)", yaxis_title="y (m)",
                   zaxis_title="z (m) — hauteur",
                   aspectmode="data"),    # repère orthonormé : la forme 360° est fidèle
        margin=dict(l=0, r=0, t=90, b=0),
    )
    fig.write_html(OUT_HTML, include_plotlyjs="cdn")
    print(f"\nFigure sauvegardée : {OUT_HTML}")
    print("Ouvre-la au navigateur : tourne à la souris, slider pour l'intérieur.")
    print(f"\nRécap — succès 5mm: {(dists<0.005).mean()*100:.1f}% | "
          f"2cm: {(dists<0.02).mean()*100:.1f}% | 5cm: {(dists<0.05).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
