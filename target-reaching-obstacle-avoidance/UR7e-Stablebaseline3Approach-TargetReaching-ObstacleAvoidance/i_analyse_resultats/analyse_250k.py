# ==============================================================================
# FICHIER : analyse_250k.py
# RÔLE : Suite d'analyses du modèle SAC entraîné à 250k étapes (sac_ur7e_reach.zip).
#        Étude de CAPACITÉ d'un agent unique (pas de moyenne multi-graines).
#
# Produit 4 analyses :
#   1. Cartographie spatiale fine (heatmaps x-z + nuage), échelle auto-ajustée
#      au min/max réel de chaque seuil pour un contraste maximal.
#   2. Taux de succès multi-seuils (5 mm / 2 cm / 5 cm).
#   3. Analyse des échecs : trajectoire distance→cible au cours du temps,
#      pour comprendre POURQUOI certains épisodes échouent (oscillation ? blocage ?).
#   4. Trajectoires 3D de l'effecteur sur quelques épisodes.
#
# Tout réutilise le modèle déjà entraîné : AUCUN réentraînement.
#
# Usage : python analyse_250k.py
# ==============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from stable_baselines3 import SAC
from ur7e_wrapper import UR7eReachEnv

# ============================= PARAMÈTRES ====================================
MODEL_PATH = "sac_ur7e_reach"       # le modèle 250k
N_CIBLES = 500                       # pour la cartographie
GRID_N = 8
SEED_CIBLES = 99999
MAX_STEPS = 300
SUCCESS_THRESHOLD = 0.005
SEUILS_M = [0.005, 0.02, 0.05]
SEUILS_LABELS = ["5 mm", "2 cm", "5 cm"]
N_TRAJ_3D = 6                        # nb d'épisodes pour les trajectoires 3D
BASE = np.array([0.0, 0.0, 0.0])
FIG_DIR = os.path.join("resultats_xp", "figures_250k")
# =============================================================================


def generate_targets(n, seed):
    rng = np.random.RandomState(seed)
    t = np.empty((n, 3))
    t[:, 0] = rng.uniform(0.25, 0.65, n)
    t[:, 1] = rng.uniform(-0.45, 0.45, n)
    t[:, 2] = rng.uniform(0.15, 0.70, n)
    return t


def run_episode(model, env, target, log_path=False):
    """
    Joue un épisode vers une cible. Retourne (distance_finale, n_pas, reached,
    + si log_path : liste des distances et liste des positions effecteur).
    """
    obs, _ = env.reset()
    env.engine.target = np.array(target, dtype=np.float64)
    obs = env.engine._get_observation()
    done = False
    dist_hist, pos_hist = [], []
    info = {}
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        done = term or trunc
        if log_path:
            dist_hist.append(info["distance"])
            pos_hist.append(env.engine.get_ee_position().copy())
    reached = info.get("done_reason") == "target_reached"
    if log_path:
        return info["distance"], len(dist_hist), reached, dist_hist, pos_hist
    return info["distance"], None, reached, None, None


# ----------------------------------------------------------------------------
# 1. Cartographie spatiale fine
# ----------------------------------------------------------------------------
def heatmap_grid(x, z, values, n):
    xe = np.linspace(x.min(), x.max(), n + 1)
    ze = np.linspace(z.min(), z.max(), n + 1)
    grid = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            m = ((x >= xe[i]) & (x <= xe[i+1]) & (z >= ze[j]) & (z <= ze[j+1]))
            if m.sum() > 0:
                grid[j, i] = values[m].mean()
    return grid, xe, ze


def analyse_cartographie(model, env, targets):
    x, z = targets[:, 0], targets[:, 2]
    dists = np.array([run_episode(model, env, t)[0] for t in targets])

    seuils = [0.005, 0.05]
    labels = ["5 mm", "5 cm"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for col, (seuil, label) in enumerate(zip(seuils, labels)):
        succ = (dists < seuil).astype(float) * 100
        grid, xe, ze = heatmap_grid(x, z, succ, GRID_N)

        # Échelle auto-ajustée au min/max réel de la grille (contraste maximal)
        vmin = np.nanmin(grid)
        vmax = np.nanmax(grid)
        if vmax - vmin < 1:   # évite une échelle dégénérée si tout est égal
            vmin, vmax = max(0, vmin - 5), min(100, vmax + 5)

        ax = axes[0, col]
        im = ax.imshow(grid, origin="lower", aspect="auto",
                       extent=[xe[0], xe[-1], ze[0], ze[-1]],
                       cmap="RdYlGn", vmin=vmin, vmax=vmax)
        ax.set_title(f"Heatmap {GRID_N}x{GRID_N} — seuil {label}")
        ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
        fig.colorbar(im, ax=ax).set_label(
            f"Succès (%) [échelle auto {vmin:.0f}-{vmax:.0f}]")

        ax2 = axes[1, col]
        sc = ax2.scatter(x, z, c=succ, cmap="RdYlGn", vmin=vmin, vmax=vmax,
                         s=18, edgecolor="none")
        ax2.set_title(f"Nuage {N_CIBLES} cibles — seuil {label}")
        ax2.set_xlabel("x (m)"); ax2.set_ylabel("z (m)")
        fig.colorbar(sc, ax=ax2).set_label("Succès (%)")

    plt.suptitle(f"Cartographie SAC 250k ({N_CIBLES} cibles, échelles auto-ajustées)",
                 fontsize=14)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "carto_250k.png")
    plt.savefig(out, dpi=150); plt.close()
    print("  ->", out)
    return dists


# ----------------------------------------------------------------------------
# 2. Multi-seuils
# ----------------------------------------------------------------------------
def analyse_multiseuils(dists):
    print("\n  Taux de succès multi-seuils (modèle 250k) :")
    rows = []
    for seuil, label in zip(SEUILS_M, SEUILS_LABELS):
        taux = (dists < seuil).mean() * 100
        print(f"    {label:<6} : {taux:.1f} %")
        rows.append((label, taux))

    plt.figure(figsize=(7, 5))
    plt.bar([r[0] for r in rows], [r[1] for r in rows],
            color=["#c0392b", "#e67e22", "#27ae60"], edgecolor="k", alpha=0.85)
    for i, (lab, t) in enumerate(rows):
        plt.text(i, t + 1.5, f"{t:.0f}%", ha="center", fontweight="bold")
    plt.ylabel("Taux de succès (%)")
    plt.ylim(0, 105)
    plt.title("Taux de succès du modèle SAC 250k selon le seuil")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "multiseuils_250k.png")
    plt.savefig(out, dpi=150); plt.close()
    print("  ->", out)


# ----------------------------------------------------------------------------
# 3. Analyse des échecs : distance vs temps
# ----------------------------------------------------------------------------
def analyse_echecs(model, env, targets, n_eval=50):
    """Rejoue des épisodes en loggant la distance, sépare succès / échecs."""
    succes_hist, echec_hist = [], []
    for t in targets[:n_eval]:
        d_final, n_pas, reached, dist_hist, _ = run_episode(
            model, env, t, log_path=True)
        if reached:
            succes_hist.append(dist_hist)
        else:
            echec_hist.append(dist_hist)

    plt.figure(figsize=(11, 6))
    for h in succes_hist:
        plt.plot(np.array(h) * 1000, color="green", alpha=0.25)
    for h in echec_hist:
        plt.plot(np.array(h) * 1000, color="red", alpha=0.5)
    # Légende manuelle
    plt.plot([], [], color="green", label=f"succès ({len(succes_hist)})")
    plt.plot([], [], color="red", label=f"échecs ({len(echec_hist)})")
    plt.axhline(5, color="black", linestyle="--", linewidth=1, label="seuil 5 mm")
    plt.xlabel("Pas de simulation")
    plt.ylabel("Distance effecteur → cible (mm)")
    plt.title("Évolution de la distance au cours des épisodes (SAC 250k)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)   # zoom sur l'approche finale
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "echecs_distance_temps_250k.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  -> {out}  ({len(succes_hist)} succès, {len(echec_hist)} échecs)")


# ----------------------------------------------------------------------------
# 4. Trajectoires 3D de l'effecteur
# ----------------------------------------------------------------------------
def analyse_trajectoires_3d(model, env, targets):
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    for t in targets[:N_TRAJ_3D]:
        _, _, reached, _, pos_hist = run_episode(model, env, t, log_path=True)
        pos = np.array(pos_hist)
        color = "green" if reached else "red"
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=color, alpha=0.7)
        ax.scatter(*pos[0], color="blue", s=40)            # départ
        ax.scatter(*t, color=color, marker="*", s=200, edgecolor="k")  # cible

    ax.scatter([], [], [], color="blue", label="départ effecteur")
    ax.scatter([], [], [], color="green", marker="*", label="cible atteinte")
    ax.scatter([], [], [], color="red", marker="*", label="cible manquée")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.set_title(f"Trajectoires 3D de l'effecteur ({N_TRAJ_3D} épisodes, SAC 250k)")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "trajectoires_3d_250k.png")
    plt.savefig(out, dpi=150); plt.close()
    print("  ->", out)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print(f"Chargement du modèle {MODEL_PATH} (250k)...")
    model = SAC.load(MODEL_PATH, device="cpu")
    env = UR7eReachEnv(render_mode=None, max_episode_len=MAX_STEPS,
                       success_threshold=SUCCESS_THRESHOLD)

    targets = generate_targets(N_CIBLES, SEED_CIBLES)

    print("\n[1/4] Cartographie spatiale fine...")
    dists = analyse_cartographie(model, env, targets)

    print("\n[2/4] Multi-seuils...")
    analyse_multiseuils(dists)

    print("\n[3/4] Analyse des échecs (distance vs temps)...")
    analyse_echecs(model, env, targets)

    print("\n[4/4] Trajectoires 3D...")
    analyse_trajectoires_3d(model, env, targets)

    env.close()
    print(f"\nTerminé. Figures dans {FIG_DIR}/")


if __name__ == "__main__":
    main()
