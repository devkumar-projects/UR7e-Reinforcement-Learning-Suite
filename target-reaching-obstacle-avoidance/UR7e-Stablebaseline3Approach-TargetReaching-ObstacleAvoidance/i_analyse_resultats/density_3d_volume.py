# ==============================================================================
# FICHIER : density_3d_volume.py
# RÔLE : "3D heatmap density plot" CONTINU du succès de SAC (250k), façon
#        Mathematica : un volume 3D rempli (rendu volumétrique translucide)
#        obtenu en MOYENNANT le voisinage des 500 cibles dispersées, AVEC
#        le nuage de points superposé.
#
# Méthode de moyennation :
#   - on construit une grille 3D régulière (GRID^3 nœuds) dans l'espace de travail ;
#   - en chaque nœud, le taux de succès est estimé par moyenne pondérée
#     gaussienne des cibles voisines (pondération exp(-d^2 / 2σ^2)).
#     => champ continu, lissé, mais fidèle aux données (pas d'extrapolation folle).
#
# Rendu interactif (HTML) :
#   - go.Volume : volume translucide, opacité traversante (on voit l'intérieur) ;
#   - go.Scatter3d : les 500 cibles, colorées, par-dessus ;
#   - sliders natifs de go.Volume pour balayer des iso-surfaces / coupes ;
#   - rotation/zoom souris.
#
# Réutilise le modèle 250k : AUCUN réentraînement.
# Usage : pip install plotly scipy ; python density_3d_volume.py
# ==============================================================================

import os
import numpy as np
from stable_baselines3 import SAC
from ur7e_wrapper import UR7eReachEnv

# ============================= PARAMÈTRES ====================================
MODEL_PATH = "sac_ur7e_reach"
N_CIBLES = 3000                # cibles atteignables visées
SEED_CIBLES = 99999
MAX_STEPS = 300
SEUIL_M = 0.05                 # succès défini à 5 cm (discriminant spatialement)
GRID = 32                      # résolution de la grille (32^3 ≈ 33k nœuds)
SIGMA = 0.06                   # rayon de moyennation gaussienne (m). ↑ = plus lisse
IK_TOL = 0.005                 # tolérance (m) pour juger une cible atteignable
OUT_HTML = os.path.join("resultats_xp", "figures_250k", "density_3d_volume_sac.html")
# bornes de la boîte d'échantillonnage des candidates (avant filtrage IK)
X0, X1 = 0.10, 0.85
Y0, Y1 = -0.80, 0.80
Z0, Z1 = 0.02, 1.00
# =============================================================================


def reachable_targets(env, n, seed, tol=IK_TOL):
    """
    Génère n cibles ATTEIGNABLES par le UR7e, via vérification IK exacte.

    Méthode (robuste) : PyBullet renvoie toujours une solution IK, même pour
    une cible hors d'atteinte. On vérifie donc explicitement :
      1) IK pour la cible candidate,
      2) on applique la config aux joints,
      3) on lit la position réelle de l'effecteur (forward kinematics),
      4) si l'écart à la cible < tol -> cible réellement atteignable.

    On tire des candidates dans la boîte et on garde celles qui passent, par
    lots, jusqu'à en obtenir n.
    """
    import pybullet as p
    rng = np.random.RandomState(seed)
    engine = env.engine
    robot = engine.robot                       # attribut réel : self.robot
    ee_index = engine.ee_index
    joints = engine.CONTROLLABLE_JOINTS        # constante de classe [2,3,4,5,6,7]

    accepted = []
    n_tested = 0
    while len(accepted) < n:
        cand = np.array([rng.uniform(X0, X1),
                         rng.uniform(Y0, Y1),
                         rng.uniform(Z0, Z1)])
        n_tested += 1
        # 1) IK
        sol = p.calculateInverseKinematics(robot, ee_index, cand.tolist())
        # 2) appliquer la config aux joints contrôlables
        for j, q in zip(joints, sol):
            p.resetJointState(robot, j, q)
        # 3) position réelle de l'effecteur
        ee = np.array(p.getLinkState(robot, ee_index)[0])
        # 4) test d'atteignabilité
        if np.linalg.norm(ee - cand) < tol:
            accepted.append(cand)
        if n_tested % 2000 == 0:
            print(f"  ... {len(accepted)}/{n} cibles atteignables "
                  f"({n_tested} testées, taux {100*len(accepted)/n_tested:.0f}%)")
    # remet le robot dans une pose neutre
    for j in joints:
        p.resetJointState(robot, j, 0.0)
    print(f"  -> {n} cibles atteignables retenues sur {n_tested} candidates "
          f"({100*n/n_tested:.0f}% d'acceptation).")
    return np.array(accepted)


def run_episode(model, env, target):
    obs, _ = env.reset()
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
    """
    Moyennation gaussienne : pour chaque nœud de grille, moyenne pondérée des
    'values' des points, poids = exp(-d^2 / 2σ^2). Champ continu et lissé.
    Calcul vectorisé par blocs pour rester léger en mémoire.
    """
    from scipy.spatial import cKDTree
    nodes = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    tree = cKDTree(points)
    # on ne considère que les points dans un rayon 3σ (au-delà, poids ~0)
    radius = 3 * sigma
    field = np.zeros(len(nodes))
    weight = np.zeros(len(nodes))
    # requête par voisinage
    neighbors = tree.query_ball_point(nodes, r=radius)
    for i, idx in enumerate(neighbors):
        if not idx:
            field[i] = np.nan
            continue
        d2 = np.sum((points[idx] - nodes[i])**2, axis=1)
        w = np.exp(-d2 / (2 * sigma**2))
        field[i] = np.sum(w * values[idx]) / np.sum(w)
        weight[i] = np.sum(w)
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
    env = UR7eReachEnv(render_mode=None, max_episode_len=MAX_STEPS,
                       success_threshold=0.005)

    print(f"Génération de {N_CIBLES} cibles atteignables (vérification IK)...")
    targets = reachable_targets(env, N_CIBLES, SEED_CIBLES)

    print(f"Évaluation sur {len(targets)} cibles...")
    dists = np.array([run_episode(model, env, t) for t in targets])
    env.close()

    # Champ continu : on interpole la DISTANCE FINALE moyenne locale (en mm),
    # plus riche en nuances que le taux de succès binaire (qui sature à ~98 %
    # au seuil 5 cm et donnerait un volume uniformément vert, peu informatif).
    # Couleur : vert = précis (faible distance), rouge = imprécis (forte distance).
    dist_mm = dists * 1000.0

    # --- Grille 3D bornée sur l'étendue réelle des cibles atteignables ---
    print(f"Construction du volume continu ({GRID}^3 nœuds, σ={SIGMA} m)...")
    gx0, gx1 = targets[:, 0].min(), targets[:, 0].max()
    gy0, gy1 = targets[:, 1].min(), targets[:, 1].max()
    gz0, gz1 = targets[:, 2].min(), targets[:, 2].max()
    xs = np.linspace(gx0, gx1, GRID)
    ys = np.linspace(gy0, gy1, GRID)
    zs = np.linspace(gz0, gz1, GRID)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    field = gaussian_field(targets, dist_mm, gx, gy, gz, SIGMA)   # distance moyenne locale (mm)

    # Bornes de couleur auto-ajustées pour le contraste (5e–95e percentile)
    valid = field[~np.isnan(field)]
    cmin = float(np.percentile(valid, 5))
    cmax = float(np.percentile(valid, 95))

    # --- Volume continu (translucide) ---
    # colorscale inversée : faible distance = vert, forte distance = rouge
    fig = go.Figure()
    fig.add_trace(go.Volume(
        x=gx.ravel(), y=gy.ravel(), z=gz.ravel(),
        value=np.nan_to_num(field, nan=cmax).ravel(),
        isomin=cmin, isomax=cmax,
        opacity=0.12,           # translucide : on voit à travers
        surface_count=18,       # nb de couches iso -> slider d'exploration
        colorscale="RdYlGn_r",
        colorbar=dict(title="Distance<br>moyenne<br>locale (mm)"),
        caps=dict(x_show=False, y_show=False, z_show=False),
        name="volume continu",
    ))

    # --- Nuage de points par-dessus ---
    val = np.clip(dists * 1000, 0, 100)
    fig.add_trace(go.Scatter3d(
        x=targets[:, 0], y=targets[:, 1], z=targets[:, 2], mode="markers",
        name="cibles (points)",
        marker=dict(size=3.5, color=val, colorscale="RdYlGn_r",
                    cmin=0, cmax=100, opacity=0.85, line=dict(width=0)),
        text=[f"{d*1000:.1f} mm" for d in dists], hoverinfo="text",
    ))

    fig.update_layout(
        title="Density 3D continu du succès de SAC (250k)<br>"
              "<sub>Volume interpolé (moyennation gaussienne) + nuage de cibles. "
              "Tournez à la souris ; le slider balaie les iso-niveaux.</sub>",
        scene=dict(xaxis_title="x (m) — portée", yaxis_title="y (m) — latéralité",
                   zaxis_title="z (m) — hauteur"),
        margin=dict(l=0, r=0, t=80, b=0),
    )
    fig.write_html(OUT_HTML, include_plotlyjs="cdn")
    print(f"\nFigure sauvegardée : {OUT_HTML}")
    print("Ouvre-la au navigateur : tourne à la souris, et utilise le slider")
    print("(go.Volume) pour balayer les niveaux de succès et voir l'intérieur.")
    print(f"\nRécap — succès 5mm: {(dists<0.005).mean()*100:.1f}% | "
          f"2cm: {(dists<0.02).mean()*100:.1f}% | 5cm: {(dists<0.05).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
