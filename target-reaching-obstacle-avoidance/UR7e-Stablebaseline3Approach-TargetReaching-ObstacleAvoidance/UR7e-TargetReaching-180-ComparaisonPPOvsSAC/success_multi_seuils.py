# ==============================================================================
# FICHIER : success_multi_seuils.py
# RÔLE : Recalcule le taux de succès SAC vs PPO à PLUSIEURS seuils de tolérance,
#        à partir des distances finales déjà enregistrées (resultats_xp/eval_*.csv).
#        AUCUN réentraînement : on relit simplement les données de l'expérience.
#
# Idée : un seul seuil (5 mm) ne dit pas tout. Reporter le succès à 5 mm / 2 cm /
# 5 cm montre que l'agent est presque toujours "dans la zone" à 5 cm mais peine
# à la précision millimétrique — analyse bien plus riche qu'un chiffre unique.
#
# Produit : un tableau console + une figure (barres groupées) + un CSV.
#
# Usage : python success_multi_seuils.py
# ==============================================================================

import os
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = "resultats_xp"
FIG_DIR = os.path.join(OUT_DIR, "figures")

# Seuils de tolérance à évaluer (en mètres)
SEUILS_M = [0.005, 0.02, 0.05]          # 5 mm, 2 cm, 5 cm
SEUILS_LABELS = ["5 mm", "2 cm", "5 cm"]

SAC_COLOR = "steelblue"
PPO_COLOR = "darkorange"


def load_distances(algo):
    """
    Charge toutes les distances finales (en mètres) d'un algo,
    regroupées par graine. Retourne une liste de tableaux (un par seed).
    """
    paths = sorted(glob.glob(os.path.join(OUT_DIR, f"eval_{algo}_seed*.csv")))
    per_seed = []
    for p in paths:
        dists = []
        with open(p) as f:
            for row in csv.DictReader(f):
                dists.append(float(row["distance_m"]))
        per_seed.append(np.array(dists))
    return per_seed


def success_rates(per_seed, seuil_m):
    """
    Pour un seuil donné, calcule le taux de succès de chaque graine,
    puis renvoie (moyenne, écart-type) sur les graines.
    """
    rates = [(d < seuil_m).mean() for d in per_seed]
    rates = np.array(rates) * 100  # en %
    return rates.mean(), rates.std()


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    sac = load_distances("sac")
    ppo = load_distances("ppo")
    if not sac or not ppo:
        print("Données introuvables dans", OUT_DIR)
        return

    # ---- Tableau console + collecte pour figure/CSV ----
    print("=" * 60)
    print("TAUX DE SUCCÈS À PLUSIEURS SEUILS (moyenne ± σ sur les graines)")
    print("=" * 60)
    print(f"{'Seuil':<8} | {'SAC':<18} | {'PPO':<18}")
    print("-" * 60)

    rows = []
    sac_means, sac_stds, ppo_means, ppo_stds = [], [], [], []
    for seuil_m, label in zip(SEUILS_M, SEUILS_LABELS):
        sm, ss = success_rates(sac, seuil_m)
        pm, ps = success_rates(ppo, seuil_m)
        sac_means.append(sm); sac_stds.append(ss)
        ppo_means.append(pm); ppo_stds.append(ps)
        print(f"{label:<8} | {sm:5.1f} ± {ss:4.1f} %     | {pm:5.1f} ± {ps:4.1f} %")
        rows.append({"seuil": label,
                     "sac_succes_moy_%": round(sm, 1), "sac_std": round(ss, 1),
                     "ppo_succes_moy_%": round(pm, 1), "ppo_std": round(ps, 1)})
    print("=" * 60)

    # ---- Export CSV ----
    out_csv = os.path.join(OUT_DIR, "succes_multi_seuils.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Tableau exporté :", out_csv)

    # ---- Figure : barres groupées ----
    x = np.arange(len(SEUILS_LABELS))
    width = 0.35
    plt.figure(figsize=(9, 6))
    plt.bar(x - width/2, sac_means, width, yerr=sac_stds, capsize=5,
            color=SAC_COLOR, alpha=0.8, label="SAC", edgecolor="k")
    plt.bar(x + width/2, ppo_means, width, yerr=ppo_stds, capsize=5,
            color=PPO_COLOR, alpha=0.8, label="PPO", edgecolor="k")
    plt.xticks(x, SEUILS_LABELS)
    plt.xlabel("Seuil de tolérance (distance à la cible)")
    plt.ylabel("Taux de succès (%)")
    plt.title("Taux de succès selon le niveau d'exigence (SAC vs PPO)")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "succes_multi_seuils.png")
    plt.savefig(fig_path, dpi=150)
    print("Figure sauvegardée :", fig_path)


if __name__ == "__main__":
    main()
