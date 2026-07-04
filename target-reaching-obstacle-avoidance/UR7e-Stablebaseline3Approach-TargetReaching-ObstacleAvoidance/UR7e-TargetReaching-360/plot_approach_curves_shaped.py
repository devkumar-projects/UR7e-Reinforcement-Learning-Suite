# ==============================================================================
# FICHIER : plot_approach_curves_shaped.py
# RÔLE : Tracer l'ÉVOLUTION DE LA DISTANCE effecteur<->cible en fonction du pas,
#        pour le modèle "shaped" (98-99% à 5 cm), sur plusieurs épisodes —
#        en distinguant les RÉUSSITES et au moins un ÉCHEC.
#
# Objectif : visualiser le PROFIL D'APPROCHE. Sur une réussite, la distance
# décroît puis se stabilise sous le seuil (freinage propre). Sur un échec,
# elle se bloque typiquement au-dessus du seuil sans le franchir.
#
# Sortie : resultats_xp/figures_360/approach_curves_shaped.png
# Usage : python plot_approach_curves_shaped.py
# ==============================================================================

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from ur7e_wrapper_360_shaped import UR7eReach360ShapedEnv

MODEL_PATH = "sac_ur7e_360_shaped_reach"
EVAL_THRESHOLD = 0.05          # 5 cm
MAX_STEPS = 300
SEED = 11                      # graine où la cible 13 échouait
N_SCAN = 60                    # nb de cibles à jouer pour récolter réussites+échec(s)
N_SUCCESS_PLOT = 6             # nb de réussites à tracer
OUT = os.path.join("resultats_xp", "figures_360", "approach_curves_shaped.png")


def run_episode_record(model, env):
    """Joue un épisode et renvoie (liste des distances par pas, succès booléen)."""
    obs, _ = env.reset()
    dists = []
    done = False
    info = {}
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        dists.append(info.get("distance", np.nan))
        done = terminated or truncated
    reached = info.get("done_reason") == "target_reached"
    return np.array(dists), reached


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    print(f"Chargement du modèle '{MODEL_PATH}'...")
    env = UR7eReach360ShapedEnv(render_mode=None, max_episode_len=MAX_STEPS,
                                success_threshold=EVAL_THRESHOLD,
                                random_start=False, seed=SEED)
    model = SAC.load(MODEL_PATH, device="cpu")

    print(f"Collecte des profils d'approche sur {N_SCAN} cibles...")
    successes, failures = [], []
    for i in range(N_SCAN):
        dists, reached = run_episode_record(model, env)
        (successes if reached else failures).append((i + 1, dists))
        print(f"  cible {i+1:2d} : {'réussite' if reached else 'ÉCHEC'} "
              f"({len(dists)} pas, dist finale {dists[-1]*100:.1f} cm)")
    env.close()

    print(f"  -> {len(successes)} réussites, {len(failures)} échec(s).")

    # sélection : quelques réussites variées (durées différentes) + tous les échecs
    successes_sorted = sorted(successes, key=lambda t: len(t[1]))
    if len(successes_sorted) > N_SUCCESS_PLOT:
        # échantillonne uniformément du plus court au plus long
        idx = np.linspace(0, len(successes_sorted) - 1, N_SUCCESS_PLOT).astype(int)
        to_plot_succ = [successes_sorted[k] for k in idx]
    else:
        to_plot_succ = successes_sorted

    # --- tracé ---
    fig, ax = plt.subplots(figsize=(10, 6))

    for cid, dists in to_plot_succ:
        ax.plot(np.arange(1, len(dists) + 1), dists * 100,
                color="#2a7", alpha=0.8, lw=1.8,
                label=f"réussite (cible {cid})")

    for cid, dists in failures:
        ax.plot(np.arange(1, len(dists) + 1), dists * 100,
                color="#d33", alpha=0.9, lw=2.2,
                label=f"ÉCHEC (cible {cid})")

    ax.axhline(EVAL_THRESHOLD * 100, color="k", ls=":", lw=1.2,
               label=f"seuil succès ({EVAL_THRESHOLD*100:.0f} cm)")

    ax.set_xlabel("pas de temps")
    ax.set_ylabel("distance effecteur ↔ cible (cm)")
    ax.set_title("Profil d'approche — modèle 360° shaped (98-99 % à 5 cm)\n"
                 "zoom phase finale : freinage sous le seuil (vert) vs blocage (rouge)")
    ax.grid(alpha=0.25)
    ax.set_ylim(4, 20)        # zoom sur la phase d'approche finale
    # légende compacte (évite doublons)
    handles, labels = ax.get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    ax.legend(seen.values(), seen.keys(), fontsize=8, loc="upper right", ncol=2)

    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print(f"\nFigure sauvegardée : {OUT}")


if __name__ == "__main__":
    main()
