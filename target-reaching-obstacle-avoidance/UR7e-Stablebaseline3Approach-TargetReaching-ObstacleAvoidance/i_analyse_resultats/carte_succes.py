# ==============================================================================
# FICHIER : carte_succes.py
# RÔLE : Cartographie de la performance de SAC par ZONE de l'espace de travail.
#        Produit, pour les seuils 5 mm et 5 cm :
#          - une heatmap en grille (plan x-z) du taux de succès par case
#          - un scatter coloré des cibles individuelles (sans interpolation)
#
# Principe : on relie chaque distance finale (eval_sac_seed*.csv) à la position
# 3D de sa cible (test_targets.npy) via l'index de ligne (l'évaluation parcourt
# les cibles DANS L'ORDRE du fichier .npy). On agrège les 5 graines : pour chaque
# cible, taux de succès moyen sur les graines.
#
# Projection : plan (x, z) — vue "de côté" du bras, où x = portée horizontale
# et z = hauteur. C'est le plan le plus parlant pour la difficulté d'atteinte.
#
# Usage : python carte_succes.py
# ==============================================================================

import os
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = "resultats_xp"
FIG_DIR = os.path.join(OUT_DIR, "figures")
ALGO = "sac"

SEUILS_M = [0.005, 0.05]        # 5 mm et 5 cm
SEUILS_LABELS = ["5 mm", "5 cm"]
GRID_N = 5                       # grille GRID_N x GRID_N pour la heatmap


def load_targets():
    path = os.path.join(OUT_DIR, "test_targets.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} introuvable. Il est généré par run_experiment.py.")
    return np.load(path)   # forme (N, 3) : colonnes x, y, z


def load_distances_per_target(algo, n_targets):
    """
    Retourne un tableau (n_targets,) = distance finale MOYENNE par cible,
    moyennée sur toutes les graines. Les cibles étant communes à tous les runs,
    la ligne i de chaque eval CSV correspond à la cible i.
    """
    paths = sorted(glob.glob(os.path.join(OUT_DIR, f"eval_{algo}_seed*.csv")))
    all_runs = []
    for p in paths:
        dists = []
        with open(p) as f:
            for row in csv.DictReader(f):
                dists.append(float(row["distance_m"]))
        if len(dists) == n_targets:
            all_runs.append(dists)
    all_runs = np.array(all_runs)   # (n_seeds, n_targets)
    return all_runs                 # on garde le détail par graine


def success_per_target(all_runs, seuil_m):
    """Taux de succès par cible (moyenne sur les graines) pour un seuil donné."""
    succ = (all_runs < seuil_m).astype(float)   # (n_seeds, n_targets)
    return succ.mean(axis=0)                     # (n_targets,) dans [0,1]


def heatmap_grid(x, z, values, n=GRID_N):
    """
    Agrège les valeurs sur une grille n x n dans le plan (x, z).
    Retourne la matrice de taux moyen + les bords de cases.
    """
    x_edges = np.linspace(x.min(), x.max(), n + 1)
    z_edges = np.linspace(z.min(), z.max(), n + 1)
    grid = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            mask = ((x >= x_edges[i]) & (x <= x_edges[i+1]) &
                    (z >= z_edges[j]) & (z <= z_edges[j+1]))
            if mask.sum() > 0:
                grid[j, i] = values[mask].mean()   # j=ligne(z), i=col(x)
    return grid, x_edges, z_edges


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    targets = load_targets()
    n_targets = len(targets)
    x, y, z = targets[:, 0], targets[:, 1], targets[:, 2]

    all_runs = load_distances_per_target(ALGO, n_targets)
    print(f"{ALGO.upper()} : {all_runs.shape[0]} graines x {n_targets} cibles chargées.")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    for col, (seuil, label) in enumerate(zip(SEUILS_M, SEUILS_LABELS)):
        succ = success_per_target(all_runs, seuil)   # taux par cible

        # ---- Heatmap (ligne 0) ----
        grid, xe, ze = heatmap_grid(x, z, succ)
        ax = axes[0, col]
        im = ax.imshow(grid * 100, origin="lower", aspect="auto",
                       extent=[xe[0], xe[-1], ze[0], ze[-1]],
                       cmap="RdYlGn", vmin=0, vmax=100)
        ax.set_title(f"Heatmap taux de succès — seuil {label}")
        ax.set_xlabel("x (m) — portée horizontale")
        ax.set_ylabel("z (m) — hauteur")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Taux de succès (%)")

        # ---- Scatter coloré (ligne 1) ----
        ax2 = axes[1, col]
        sc = ax2.scatter(x, z, c=succ * 100, cmap="RdYlGn", vmin=0, vmax=100,
                         s=120, edgecolor="k", linewidth=0.5)
        ax2.set_title(f"Cibles individuelles — seuil {label}")
        ax2.set_xlabel("x (m) — portée horizontale")
        ax2.set_ylabel("z (m) — hauteur")
        cbar2 = fig.colorbar(sc, ax=ax2)
        cbar2.set_label("Taux de succès (%)")

    plt.suptitle(f"Cartographie de la performance de {ALGO.upper()} "
                 f"dans le plan (x, z)", fontsize=14)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "carte_succes_sac.png")
    plt.savefig(out, dpi=150)
    print("Figure sauvegardée :", out)


if __name__ == "__main__":
    main()
