# ==============================================================================
# FICHIER : visualize_360_shaped.py
# RÔLE : VISUALISER dans PyBullet (fenêtre GUI) le modèle SAC "shaped" (99% à
#        5 cm sur l'espace 360° complet, départ depuis la pose de repos fixe),
#        sur ~50 cibles atteignables enchaînées.
#
# Objectif : OBSERVER le freinage terminal propre que le reward shaping
# (bonus proximité <3cm + pénalité vitesse <5cm) a permis d'apprendre —
# par contraste avec l'oscillation/dépassement du modèle curriculum.
#
# Chaque cible : départ depuis la pose home FIXE, marqueur vert de la cible,
# déroulé en temps réel ralenti, distance finale + succès/échec à 5 cm.
#
# Usage : python visualize_360_shaped.py
# ==============================================================================

import time
import numpy as np
from stable_baselines3 import SAC
from ur7e_wrapper_360_shaped import UR7eReach360ShapedEnv

MODEL_PATH = "sac_ur7e_360_shaped_reach"
N_TARGETS = 50                 # nombre de cibles à enchaîner
EVAL_THRESHOLD = 0.05          # seuil d'évaluation du succès (5 cm)
MAX_STEPS = 300
SLOWDOWN = 1.0 / 80.0          # pause entre pas (s) — augmente pour ralentir
PAUSE_BETWEEN = 0.6            # pause (s) entre deux cibles
SEED = 11                      # graine de test (cibles différentes de l'entraînement)


def main():
    print(f"Chargement du modèle '{MODEL_PATH}'...")
    # départ home FIXE (random_start=False, défaut du wrapper shaped)
    env = UR7eReach360ShapedEnv(render_mode="human", max_episode_len=MAX_STEPS,
                                success_threshold=EVAL_THRESHOLD,
                                random_start=False, seed=SEED)
    model = SAC.load(MODEL_PATH, device="cpu")

    print(f"Visualisation sur {N_TARGETS} cibles (seuil succès = "
          f"{EVAL_THRESHOLD*100:.0f} cm). Observe le FREINAGE en approche finale.")
    print("Ferme la fenêtre ou Ctrl+C pour arrêter.\n")

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
                time.sleep(SLOWDOWN)
            dist = info.get("distance", float("nan"))
            distances.append(dist)
            reached = info.get("done_reason") == "target_reached"
            successes += int(reached)
            tag = "✓ ATTEINTE" if reached else "✗ ratée"
            print(f"  Cible {i+1:2d}/{N_TARGETS} : {tag}  |  "
                  f"distance finale {dist*100:5.1f} cm  |  {steps} pas")
            time.sleep(PAUSE_BETWEEN)
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
