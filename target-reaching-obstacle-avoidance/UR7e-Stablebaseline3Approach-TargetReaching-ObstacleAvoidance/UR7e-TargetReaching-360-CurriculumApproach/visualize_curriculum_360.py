# ==============================================================================
# FICHIER : visualize_curriculum_360.py
# RÔLE : VISUALISER dans PyBullet (fenêtre GUI) le modèle curriculum entraîné,
#        sur ~50 cibles atteignables enchaînées. Permet d'observer concrètement
#        le comportement de l'agent — notamment le "freinage fin" en approche
#        finale qui limite la précision sous ~10 cm.
#
# Pour chaque cible : départ aléatoire non singulier (hérité du reset), marqueur
# vert de la cible, déroulé de l'épisode en temps réel ralenti, et affichage de
# la distance finale + succès/échec au seuil d'évaluation choisi.
#
# Usage : python visualize_curriculum_360.py
#   (ouvre une fenêtre PyBullet ; ferme-la ou Ctrl+C pour arrêter)
# ==============================================================================

import time
import numpy as np
from stable_baselines3 import SAC
from ur7e_wrapper_360 import UR7eReach360Env

MODEL_PATH = "sac_ur7e_curriculum_360_reach"
N_TARGETS = 50                 # nombre de cibles à enchaîner
EVAL_THRESHOLD = 0.05          # seuil d'évaluation du succès (5 cm)
MAX_STEPS = 300
SLOWDOWN = 1.0 / 120.0         # pause entre pas (s) pour rendre le mouvement visible
PAUSE_BETWEEN = 0.6            # pause (s) entre deux cibles
SEED = 7


def main():
    print(f"Chargement du modèle '{MODEL_PATH}'...")
    # render_mode="human" -> ouvre la fenêtre GUI PyBullet
    env = UR7eReach360Env(render_mode="human", max_episode_len=MAX_STEPS,
                          success_threshold=EVAL_THRESHOLD,
                          random_start=True, seed=SEED)
    model = SAC.load(MODEL_PATH, device="cpu")

    print(f"Visualisation sur {N_TARGETS} cibles (seuil succès = "
          f"{EVAL_THRESHOLD*100:.0f} cm). Ferme la fenêtre ou Ctrl+C pour arrêter.\n")

    distances = []
    successes = 0
    try:
        for i in range(N_TARGETS):
            obs, _ = env.reset()
            done = False
            info = {}
            steps = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1
                time.sleep(SLOWDOWN)        # ralentit pour rendre visible

            dist = info.get("distance", float("nan"))
            distances.append(dist)
            reached = info.get("done_reason") == "target_reached"
            successes += int(reached)
            tag = "✓ ATTEINTE" if reached else "✗ ratée"
            print(f"  Cible {i+1:2d}/{N_TARGETS} : {tag}  |  "
                  f"distance finale {dist*100:5.1f} cm  |  {steps} pas")
            time.sleep(PAUSE_BETWEEN)        # petite pause avant la cible suivante
    except KeyboardInterrupt:
        print("\nInterrompu par l'utilisateur.")
    finally:
        d = np.array(distances) if distances else np.array([np.nan])
        n = len(distances)
        print("\n--- Récapitulatif ---")
        if n > 0:
            print(f"  Cibles évaluées      : {n}")
            print(f"  Succès (< {EVAL_THRESHOLD*100:.0f} cm)      : "
                  f"{successes}/{n}  ({100*successes/n:.0f}%)")
            print(f"  Distance finale moy. : {np.nanmean(d)*100:.1f} cm")
            print(f"  Distance médiane     : {np.nanmedian(d)*100:.1f} cm")
            print(f"  Min / Max            : {np.nanmin(d)*100:.1f} / "
                  f"{np.nanmax(d)*100:.1f} cm")
        env.close()


if __name__ == "__main__":
    main()
