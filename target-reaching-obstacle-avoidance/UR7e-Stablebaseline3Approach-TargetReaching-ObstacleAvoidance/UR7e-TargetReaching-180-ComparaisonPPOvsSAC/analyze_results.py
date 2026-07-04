# ==============================================================================
# FICHIER : analyze_results.py
# RÔLE : Analyse statistique de la comparaison SAC vs PPO.
#
# Lit les données brutes produites par run_experiment.py et produit :
#   - moyenne ± écart-type de chaque métrique, par algorithme, sur les N graines
#   - un test de significativité de Mann-Whitney U sur les métriques clés
#   - un tableau récapitulatif exporté en CSV
#
# CHOIX DU TEST — Mann-Whitney U (non paramétrique) :
#   Avec seulement N graines (petit échantillon), l'hypothèse de normalité requise
#   par le t-test ne peut être vérifiée de façon fiable. Le test de Mann-Whitney U
#   ne suppose aucune distribution : il compare les rangs des valeurs. Il indique
#   si un algo tend à être systématiquement supérieur à l'autre.
#
#   Interprétation de la p-value :
#     p < 0.05  -> différence statistiquement significative
#     p >= 0.05 -> pas de différence significative démontrable
#   NUANCE : avec peu de graines, la puissance est limitée ; un résultat non
#   significatif ne prouve PAS l'égalité, il indique seulement qu'on ne peut pas
#   conclure à une différence avec ce nombre d'échantillons.
#
# Usage : python analyze_results.py
# ==============================================================================

import os
import csv
import numpy as np

try:
    from scipy.stats import mannwhitneyu
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

OUT_DIR = "resultats_xp"

# Métriques sur lesquelles on fait le test (nom CSV, libellé, sens "mieux")
METRICS = [
    ("success_rate", "Taux de succès", "haut"),
    ("dist_median_mm", "Distance médiane (mm)", "bas"),
    ("steps_mean", "Pas pour atteindre", "bas"),
    ("train_time_s", "Temps d'entraînement (s)", "bas"),
    ("energy_mean", "Énergie trajectoire", "bas"),
]


def load_summary():
    path = os.path.join(OUT_DIR, "summary.csv")
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def split_by_algo(rows):
    data = {"sac": {}, "ppo": {}}
    for key, _, _ in METRICS:
        for algo in ("sac", "ppo"):
            data[algo][key] = np.array(
                [float(r[key]) for r in rows if r["algo"] == algo])
    return data


def main():
    rows = load_summary()
    data = split_by_algo(rows)
    n_sac = sum(1 for r in rows if r["algo"] == "sac")
    n_ppo = sum(1 for r in rows if r["algo"] == "ppo")

    print("=" * 78)
    print(f"COMPARAISON STATISTIQUE SAC vs PPO  (SAC: {n_sac} graines, PPO: {n_ppo} graines)")
    print("=" * 78)
    print(f"{'Métrique':<28} | {'SAC (moy ± σ)':<20} | {'PPO (moy ± σ)':<20} | p-value")
    print("-" * 78)

    table_rows = []
    for key, label, sens in METRICS:
        sac_v = data["sac"][key]
        ppo_v = data["ppo"][key]
        sac_str = f"{sac_v.mean():.3g} ± {sac_v.std():.2g}"
        ppo_str = f"{ppo_v.mean():.3g} ± {ppo_v.std():.2g}"

        if HAS_SCIPY:
            try:
                _, p = mannwhitneyu(sac_v, ppo_v, alternative="two-sided")
                p_str = f"{p:.3f}" + (" *" if p < 0.05 else "")
            except ValueError:
                p_str = "n/a"
        else:
            p_str = "scipy absent"

        print(f"{label:<28} | {sac_str:<20} | {ppo_str:<20} | {p_str}")
        table_rows.append({
            "metrique": label, "sac_moy": sac_v.mean(), "sac_std": sac_v.std(),
            "ppo_moy": ppo_v.mean(), "ppo_std": ppo_v.std(),
            "p_value": p_str, "mieux_si": sens,
        })

    print("-" * 78)
    print("* = différence significative (p < 0.05)")
    if not HAS_SCIPY:
        print("\n[!] scipy non installé : pip install scipy  (pour les p-values)")

    # Export du tableau
    out_csv = os.path.join(OUT_DIR, "tableau_comparaison.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
        w.writeheader()
        w.writerows(table_rows)
    print(f"\nTableau exporté : {out_csv}")
    print("Lance maintenant : python plot_results.py")


if __name__ == "__main__":
    main()
