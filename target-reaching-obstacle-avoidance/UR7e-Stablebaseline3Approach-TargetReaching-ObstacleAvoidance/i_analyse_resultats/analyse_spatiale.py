# ==============================================================================
# FICHIER : analyse_spatiale.py
# RÔLE : Analyses spatiales complémentaires de la performance de SAC.
#        Produit, pour les seuils 5 mm et 5 cm :
#          1. Projections 2D multiples : plan (x,z) "vue de côté" ET (x,y) "vue de dessus"
#          2. Graphe taux de succès en fonction de la DISTANCE RADIALE à la base
#             (teste directement l'hypothèse "plus c'est loin, plus c'est dur")
#          3. Scatter 3D INTERACTIF (HTML rotatif) pour exploration — nécessite plotly
#
# Données : test_targets.npy (positions) + eval_sac_seed*.csv (distances finales),
# appariées par index de ligne. Taux de succès par cible = moyenne sur les graines.
#
# Usage : python analyse_spatiale.py
#   (pour le 3D : pip install plotly)
# ==============================================================================

import os
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = "resultats_xp"
FIG_DIR = os.path.join(OUT_DIR, "figures")
ALGO = "sac"

SEUILS_M = [0.005, 0.05]
SEUILS_LABELS = ["5 mm", "5 cm"]

# Position de la base du robot (origine) pour la distance radiale
BASE = np.array([0.0, 0.0, 0.0])


def load_targets():
    path = os.path.join(OUT_DIR, "test_targets.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} introuvable (généré par run_experiment.py).")
    return np.load(path)


def load_runs(algo, n_targets):
    paths = sorted(glob.glob(os.path.join(OUT_DIR, f"eval_{algo}_seed*.csv")))
    runs = []
    for p in paths:
        dists = []
        with open(p) as f:
            for row in csv.DictReader(f):
                dists.append(float(row["distance_m"]))
        if len(dists) == n_targets:
            runs.append(dists)
    return np.array(runs)


def success_per_target(runs, seuil_m):
    return (runs < seuil_m).astype(float).mean(axis=0)


# ----------------------------------------------------------------------------
# 1. Projections 2D multiples : (x,z) et (x,y)
# ----------------------------------------------------------------------------
def fig_projections(targets, runs):
    x, y, z = targets[:, 0], targets[:, 1], targets[:, 2]
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    for col, (seuil, label) in enumerate(zip(SEUILS_M, SEUILS_LABELS)):
        succ = success_per_target(runs, seuil) * 100

        # Vue de côté (x, z)
        ax = axes[0, col]
        sc = ax.scatter(x, z, c=succ, cmap="RdYlGn", vmin=0, vmax=100,
                        s=120, edgecolor="k", linewidth=0.5)
        ax.set_title(f"Vue de côté (x, z) — seuil {label}")
        ax.set_xlabel("x (m) — portée horizontale")
        ax.set_ylabel("z (m) — hauteur")
        fig.colorbar(sc, ax=ax).set_label("Taux de succès (%)")

        # Vue de dessus (x, y)
        ax2 = axes[1, col]
        sc2 = ax2.scatter(x, y, c=succ, cmap="RdYlGn", vmin=0, vmax=100,
                          s=120, edgecolor="k", linewidth=0.5)
        ax2.set_title(f"Vue de dessus (x, y) — seuil {label}")
        ax2.set_xlabel("x (m) — portée horizontale")
        ax2.set_ylabel("y (m) — latéralité")
        fig.colorbar(sc2, ax=ax2).set_label("Taux de succès (%)")

    plt.suptitle(f"Projections 2D de la performance de {ALGO.upper()}", fontsize=14)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "projections_2d_sac.png")
    plt.savefig(out, dpi=150)
    print("Figure sauvegardée :", out)


# ----------------------------------------------------------------------------
# 2. Taux de succès en fonction de la distance radiale à la base
# ----------------------------------------------------------------------------
def fig_distance_radiale(targets, runs):
    radial = np.linalg.norm(targets - BASE, axis=1)  # distance de chaque cible à la base

    plt.figure(figsize=(10, 6))
    for seuil, label, color in zip(SEUILS_M, SEUILS_LABELS, ["darkred", "darkgreen"]):
        succ = success_per_target(runs, seuil) * 100
        # nuage de points
        plt.scatter(radial, succ, alpha=0.5, color=color, s=50,
                    label=f"cibles (seuil {label})")
        # tendance : moyenne par tranche de distance
        bins = np.linspace(radial.min(), radial.max(), 7)
        idx = np.digitize(radial, bins)
        bx, by = [], []
        for b in range(1, len(bins)):
            m = idx == b
            if m.sum() > 0:
                bx.append(radial[m].mean())
                by.append(succ[m].mean())
        plt.plot(bx, by, "-o", color=color, linewidth=2,
                 label=f"tendance (seuil {label})")

    plt.xlabel("Distance radiale cible → base du robot (m)")
    plt.ylabel("Taux de succès (%)")
    plt.title(f"Performance de {ALGO.upper()} selon l'éloignement de la cible")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "succes_vs_distance_radiale.png")
    plt.savefig(out, dpi=150)
    print("Figure sauvegardée :", out)


# ----------------------------------------------------------------------------
# 3. Scatter 3D interactif (HTML) — nécessite plotly
# ----------------------------------------------------------------------------
def fig_3d_interactif(targets, runs):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[3D] plotly non installé -> scatter 3D ignoré. "
              "Installe-le avec : pip install plotly")
        return

    x, y, z = targets[:, 0], targets[:, 1], targets[:, 2]
    # On colore selon le seuil 5 cm (le plus discriminant spatialement)
    succ = success_per_target(runs, 0.05) * 100

    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z, mode="markers",
        marker=dict(size=6, color=succ, colorscale="RdYlGn", cmin=0, cmax=100,
                    colorbar=dict(title="Succès (%)"), line=dict(width=1, color="black")),
        text=[f"({xi:.2f},{yi:.2f},{zi:.2f}) — {s:.0f}%"
              for xi, yi, zi, s in zip(x, y, z, succ)],
    )])
    fig.update_layout(
        title=f"Scatter 3D interactif — performance {ALGO.upper()} (seuil 5 cm)",
        scene=dict(xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)"),
    )
    out = os.path.join(FIG_DIR, "scatter_3d_sac.html")
    fig.write_html(out)
    print("Scatter 3D interactif sauvegardé :", out, "(ouvrir dans un navigateur)")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    targets = load_targets()
    runs = load_runs(ALGO, len(targets))
    print(f"{ALGO.upper()} : {runs.shape[0]} graines x {len(targets)} cibles.")

    fig_projections(targets, runs)
    fig_distance_radiale(targets, runs)
    fig_3d_interactif(targets, runs)


if __name__ == "__main__":
    main()
