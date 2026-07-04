# ==============================================================================
# FICHIER : plot_results.py
# RÔLE : Génération des figures de la comparaison SAC vs PPO pour le rapport.
#
# Produit :
#   1. Courbes d'apprentissage agrégées (récompense moyenne vs étapes) avec
#      BANDE D'ÉCART-TYPE sur les N graines -> prouve la rigueur (tendance + variabilité).
#   2. Box plots des distances finales (précision et dispersion).
#   3. Box plots du taux de succès et du nombre de pas par graine.
#   4. Scatter temps d'entraînement vs performance (compromis coût/perf).
#
# Usage : python plot_results.py
# ==============================================================================

import os
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = "resultats_xp"
FIG_DIR = os.path.join(OUT_DIR, "figures")

SAC_COLOR = "steelblue"
PPO_COLOR = "darkorange"


def load_monitor(path):
    """Charge un monitor.csv de SB3 (2 lignes d'en-tête, puis r,l,t)."""
    steps, rewards = [], []
    with open(path) as f:
        lines = f.readlines()
    # ligne 0 = commentaire JSON, ligne 1 = en-tête r,l,t
    cum = 0
    for line in lines[2:]:
        parts = line.strip().split(",")
        if len(parts) < 3:
            continue
        r, l = float(parts[0]), float(parts[1])
        cum += l
        steps.append(cum)
        rewards.append(r)
    return np.array(steps), np.array(rewards)


def interp_grid(all_steps, all_rewards, n_points=200):
    """Interpole toutes les courbes sur une grille commune pour les agréger."""
    max_step = min(s[-1] for s in all_steps if len(s) > 0)
    grid = np.linspace(0, max_step, n_points)
    stacked = []
    for s, r in zip(all_steps, all_rewards):
        if len(s) > 1:
            stacked.append(np.interp(grid, s, r))
    return grid, np.array(stacked)


def load_learning_curves(algo):
    """Charge les courbes d'apprentissage de tous les seeds d'un algo."""
    paths = sorted(glob.glob(os.path.join(OUT_DIR, f"{algo}_seed*", "monitor.csv")))
    all_s, all_r = [], []
    for p in paths:
        s, r = load_monitor(p)
        all_s.append(s)
        all_r.append(r)
    return all_s, all_r


def load_eval_distances(algo):
    """Charge les distances finales (toutes cibles, tous seeds) d'un algo."""
    paths = sorted(glob.glob(os.path.join(OUT_DIR, f"eval_{algo}_seed*.csv")))
    dists = []
    for p in paths:
        with open(p) as f:
            for row in csv.DictReader(f):
                dists.append(float(row["distance_m"]) * 1000)  # mm
    return np.array(dists)


def load_summary_by_algo(key):
    path = os.path.join(OUT_DIR, "summary.csv")
    out = {"sac": [], "ppo": []}
    with open(path) as f:
        for r in csv.DictReader(f):
            out[r["algo"]].append(float(r[key]))
    return out


def fig_learning_curves():
    plt.figure(figsize=(10, 6))
    for algo, color in (("sac", SAC_COLOR), ("ppo", PPO_COLOR)):
        all_s, all_r = load_learning_curves(algo)
        if not all_s:
            continue
        grid, stacked = interp_grid(all_s, all_r)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        plt.plot(grid, mean, color=color, linewidth=2, label=f"{algo.upper()} (moyenne)")
        plt.fill_between(grid, mean - std, mean + std, color=color, alpha=0.2,
                         label=f"{algo.upper()} (± écart-type)")
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("Étapes d'entraînement")
    plt.ylabel("Récompense moyenne par épisode")
    plt.title("Courbes d'apprentissage agrégées sur les graines (SAC vs PPO)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "courbes_apprentissage.png"), dpi=150)
    print("  courbes_apprentissage.png")


def fig_boxplot_distances():
    sac_d = load_eval_distances("sac")
    ppo_d = load_eval_distances("ppo")
    plt.figure(figsize=(8, 6))
    bp = plt.boxplot([sac_d, ppo_d], tick_labels=["SAC", "PPO"], patch_artist=True,
                     showfliers=True)
    for patch, color in zip(bp["boxes"], [SAC_COLOR, PPO_COLOR]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    plt.axhline(5, color="red", linestyle="--", linewidth=1,
                label="seuil succès (5 mm)")
    plt.ylabel("Distance finale à la cible (mm)")
    plt.title("Distribution des distances finales (toutes cibles, toutes graines)")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "boxplot_distances.png"), dpi=150)
    print("  boxplot_distances.png")


def fig_boxplot_success_steps():
    succ = load_summary_by_algo("success_rate")
    steps = load_summary_by_algo("steps_mean")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    bp1 = axes[0].boxplot([np.array(succ["sac"])*100, np.array(succ["ppo"])*100],
                          tick_labels=["SAC", "PPO"], patch_artist=True)
    for patch, color in zip(bp1["boxes"], [SAC_COLOR, PPO_COLOR]):
        patch.set_facecolor(color); patch.set_alpha(0.5)
    axes[0].set_ylabel("Taux de succès (%)")
    axes[0].set_title("Taux de succès par graine")
    axes[0].grid(True, alpha=0.3, axis="y")

    bp2 = axes[1].boxplot([steps["sac"], steps["ppo"]],
                          tick_labels=["SAC", "PPO"], patch_artist=True)
    for patch, color in zip(bp2["boxes"], [SAC_COLOR, PPO_COLOR]):
        patch.set_facecolor(color); patch.set_alpha(0.5)
    axes[1].set_ylabel("Pas moyens pour atteindre")
    axes[1].set_title("Efficacité de trajectoire par graine")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "boxplot_succes_pas.png"), dpi=150)
    print("  boxplot_succes_pas.png")


def fig_scatter_cost():
    tt = load_summary_by_algo("train_time_s")
    sr = load_summary_by_algo("success_rate")
    plt.figure(figsize=(8, 6))
    plt.scatter(tt["sac"], np.array(sr["sac"])*100, color=SAC_COLOR, s=80,
                label="SAC", edgecolor="k", alpha=0.7)
    plt.scatter(tt["ppo"], np.array(sr["ppo"])*100, color=PPO_COLOR, s=80,
                label="PPO", edgecolor="k", alpha=0.7)
    plt.xlabel("Temps d'entraînement (s)")
    plt.ylabel("Taux de succès (%)")
    plt.title("Compromis coût de calcul / performance (chaque point = une graine)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "scatter_cout_perf.png"), dpi=150)
    print("  scatter_cout_perf.png")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print("Génération des figures dans", FIG_DIR, ":")
    fig_learning_curves()
    fig_boxplot_distances()
    fig_boxplot_success_steps()
    fig_scatter_cost()
    print("Terminé.")


if __name__ == "__main__":
    main()
