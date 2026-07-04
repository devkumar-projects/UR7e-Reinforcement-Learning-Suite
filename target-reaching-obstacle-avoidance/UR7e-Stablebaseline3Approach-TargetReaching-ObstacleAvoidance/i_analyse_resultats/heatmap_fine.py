# ==============================================================================
# FICHIER : heatmap_fine.py
# RÔLE : Analyse spatiale FINE de SAC sur un grand nombre de cibles.
#
# Recharge les 5 modèles SAC déjà entraînés (PAS de réentraînement), génère
# N_CIBLES nouvelles cibles dans l'espace de travail, les évalue avec chaque
# modèle, et moyenne. Produit des heatmaps fines (grille bien remplie) + le
# graphe radial, désormais statistiquement significatifs.
#
# Pourquoi : 50 cibles -> ~2/case sur une grille 5x5 (bruité). Avec 500 cibles
# sur une grille 8x8 (64 cases) -> ~8 cibles/case x 5 modèles = ~40 mesures/case,
# suffisant pour un taux fiable (±8% environ).
#
# Usage : python heatmap_fine.py
# ==============================================================================

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from ur7e_wrapper import UR7eReachEnv

# ============================= PARAMÈTRES ====================================
N_CIBLES = 500
GRID_N = 8
SEED_CIBLES = 99999      # graine dédiée (différente du jeu de test à 12345)
MAX_STEPS = 300
SEUILS_M = [0.005, 0.05]
SEUILS_LABELS = ["5 mm", "5 cm"]
# Bornes de l'échelle de couleur adaptées à chaque seuil (vmin, vmax en %)
SEUILS_VLIM = [(0, 25), (50, 100)]   # 5 mm : 0-25% ; 5 cm : 50-100%
MODELS_DIR = "resultats_xp"          # où sont sac_seed0.zip ... sac_seed4.zip
FIG_DIR = os.path.join("resultats_xp", "figures")
BASE = np.array([0.0, 0.0, 0.0])
# =============================================================================


def generate_targets(n, seed):
    rng = np.random.RandomState(seed)
    t = np.empty((n, 3))
    t[:, 0] = rng.uniform(0.25, 0.65, n)   # x
    t[:, 1] = rng.uniform(-0.45, 0.45, n)  # y
    t[:, 2] = rng.uniform(0.15, 0.70, n)   # z
    return t


def find_sac_models():
    paths = sorted(glob.glob(os.path.join(MODELS_DIR, "sac_seed*.zip")))
    if not paths:
        raise FileNotFoundError(
            f"Aucun modèle sac_seed*.zip dans {MODELS_DIR}/")
    return paths


def evaluate_targets(model, env, targets):
    """Retourne le tableau des distances finales pour toutes les cibles."""
    dists = np.empty(len(targets))
    for i, tgt in enumerate(targets):
        obs, _ = env.reset()
        env.engine.target = np.array(tgt, dtype=np.float64)
        obs = env.engine._get_observation()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
        dists[i] = info["distance"]
    return dists


def heatmap_grid(x, z, values, n):
    x_edges = np.linspace(x.min(), x.max(), n + 1)
    z_edges = np.linspace(z.min(), z.max(), n + 1)
    grid = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            m = ((x >= x_edges[i]) & (x <= x_edges[i+1]) &
                 (z >= z_edges[j]) & (z <= z_edges[j+1]))
            if m.sum() > 0:
                grid[j, i] = values[m].mean()
    return grid, x_edges, z_edges


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    targets = generate_targets(N_CIBLES, SEED_CIBLES)
    x, y, z = targets[:, 0], targets[:, 1], targets[:, 2]

    model_paths = find_sac_models()
    print(f"{len(model_paths)} modèles SAC trouvés. Évaluation sur {N_CIBLES} cibles...")

    env = UR7eReachEnv(render_mode=None, max_episode_len=MAX_STEPS)

    # distances : (n_models, n_cibles)
    all_dists = []
    for k, mp in enumerate(model_paths):
        model = SAC.load(mp, device="cpu")
        d = evaluate_targets(model, env, targets)
        all_dists.append(d)
        print(f"  modèle {k+1}/{len(model_paths)} évalué "
              f"(médiane {np.median(d)*1000:.1f} mm)")
    env.close()
    all_dists = np.array(all_dists)   # (n_models, n_cibles)

    # ---- Heatmaps + scatter pour chaque seuil ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for col, (seuil, label, (vmin, vmax)) in enumerate(
            zip(SEUILS_M, SEUILS_LABELS, SEUILS_VLIM)):
        # taux de succès par cible = moyenne sur les modèles
        succ = (all_dists < seuil).astype(float).mean(axis=0) * 100  # (n_cibles,)

        grid, xe, ze = heatmap_grid(x, z, succ, GRID_N)
        ax = axes[0, col]
        im = ax.imshow(grid, origin="lower", aspect="auto",
                       extent=[xe[0], xe[-1], ze[0], ze[-1]],
                       cmap="RdYlGn", vmin=vmin, vmax=vmax)
        ax.set_title(f"Heatmap {GRID_N}x{GRID_N} — seuil {label} ({N_CIBLES} cibles)")
        ax.set_xlabel("x (m) — portée horizontale")
        ax.set_ylabel("z (m) — hauteur")
        fig.colorbar(im, ax=ax).set_label(f"Taux de succès (%) [échelle {vmin}-{vmax}]")

        ax2 = axes[1, col]
        sc = ax2.scatter(x, z, c=succ, cmap="RdYlGn", vmin=vmin, vmax=vmax,
                         s=18, edgecolor="none")
        ax2.set_title(f"Nuage des {N_CIBLES} cibles — seuil {label}")
        ax2.set_xlabel("x (m) — portée horizontale")
        ax2.set_ylabel("z (m) — hauteur")
        fig.colorbar(sc, ax=ax2).set_label(f"Taux de succès (%) [échelle {vmin}-{vmax}]")

    plt.suptitle(f"Cartographie fine de SAC ({N_CIBLES} cibles, "
                 f"{len(model_paths)} modèles moyennés)", fontsize=14)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "heatmap_fine_sac.png")
    plt.savefig(out, dpi=150)
    print("Figure sauvegardée :", out)

    # ---- Graphe radial ----
    radial = np.linalg.norm(targets - BASE, axis=1)
    plt.figure(figsize=(10, 6))
    for seuil, label, color in zip(SEUILS_M, SEUILS_LABELS, ["darkred", "darkgreen"]):
        succ = (all_dists < seuil).astype(float).mean(axis=0) * 100
        bins = np.linspace(radial.min(), radial.max(), 12)
        idx = np.digitize(radial, bins)
        bx, by = [], []
        for b in range(1, len(bins)):
            m = idx == b
            if m.sum() > 0:
                bx.append(radial[m].mean()); by.append(succ[m].mean())
        plt.plot(bx, by, "-o", color=color, linewidth=2, label=f"seuil {label}")
    plt.xlabel("Distance radiale cible → base du robot (m)")
    plt.ylabel("Taux de succès (%)")
    plt.title(f"Performance de SAC selon l'éloignement ({N_CIBLES} cibles)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out2 = os.path.join(FIG_DIR, "radial_fine_sac.png")
    plt.savefig(out2, dpi=150)
    print("Figure sauvegardée :", out2)


if __name__ == "__main__":
    main()
