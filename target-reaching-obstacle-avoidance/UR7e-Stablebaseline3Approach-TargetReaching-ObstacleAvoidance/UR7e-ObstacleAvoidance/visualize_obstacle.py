# ==============================================================================
# FICHIER : visualize_obstacle.py
# RÔLE : VISUALISER dans PyBullet le modèle d'ÉVITEMENT D'OBSTACLE, sur ~50
#        cibles. La cible est une sphère verte, l'obstacle une sphère ROUGE
#        placée aléatoirement sur le chemin. On observe le CONTOURNEMENT —
#        et, pour les ~28% d'échecs, comment/où la collision survient.
#
# Chaque épisode classé en : ✓ ATTEINTE / ✗ COLLISION / ⌛ TIMEOUT.
# Récapitulatif final avec les trois taux.
#
# Usage : python visualize_obstacle.py
# ==============================================================================

import time
import numpy as np
from stable_baselines3 import SAC
from ur7e_wrapper_obstacle import UR7eReachObstacleEnv

MODEL_PATH = "sac_ur7e_obstacle_reach"
N_TARGETS = 50
EVAL_THRESHOLD = 0.05
OBSTACLE_RADIUS = 0.10            # doit matcher l'entraînement
MAX_STEPS = 300
SLOWDOWN = 1.0 / 80.0             # augmente pour ralentir
PAUSE_BETWEEN = 0.7
SEED = 13                         # cibles/obstacles de test (différents de l'entr.)


def main():
    print(f"Chargement du modèle '{MODEL_PATH}'...")
    env = UR7eReachObstacleEnv(render_mode="human", max_episode_len=MAX_STEPS,
                               success_threshold=EVAL_THRESHOLD,
                               random_start=False,
                               obstacle_radius=OBSTACLE_RADIUS, seed=SEED)
    model = SAC.load(MODEL_PATH, device="cpu")

    print(f"Visualisation sur {N_TARGETS} cibles. Vert = cible, ROUGE = obstacle.")
    print("Observe le CONTOURNEMENT de la sphère rouge. Ctrl+C pour arrêter.\n")

    n_success = n_collision = n_timeout = 0
    clearances = []
    try:
        for i in range(N_TARGETS):
            obs, _ = env.reset()
            done = False
            info = {}
            steps = 0
            min_clear = np.inf
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1
                c = info.get("obstacle_clearance")
                if c is not None:
                    min_clear = min(min_clear, c)
                time.sleep(SLOWDOWN)

            reason = info.get("done_reason", "?")
            dist = info.get("distance", float("nan"))
            if reason == "target_reached":
                n_success += 1; tag = "✓ ATTEINTE"
            elif reason == "collision":
                n_collision += 1; tag = "✗ COLLISION"
            else:
                n_timeout += 1; tag = "⌛ TIMEOUT"
            clearances.append(min_clear)
            print(f"  Cible {i+1:2d}/{N_TARGETS} : {tag:12s} |  "
                  f"dist {dist*100:5.1f} cm  |  clearance min {min_clear*100:5.1f} cm"
                  f"  |  {steps} pas")
            time.sleep(PAUSE_BETWEEN)
    except KeyboardInterrupt:
        print("\nInterrompu.")
    finally:
        n = n_success + n_collision + n_timeout
        print("\n--- Récapitulatif ---")
        if n > 0:
            print(f"  Épisodes            : {n}")
            print(f"  ✓ Succès            : {n_success}/{n}  ({100*n_success/n:.0f}%)")
            print(f"  ✗ Collisions        : {n_collision}/{n}  ({100*n_collision/n:.0f}%)")
            print(f"  ⌛ Timeouts          : {n_timeout}/{n}  ({100*n_timeout/n:.0f}%)")
            cl = np.array([c for c in clearances if np.isfinite(c)])
            if len(cl):
                print(f"  Clearance min moy.  : {np.mean(cl)*100:.1f} cm "
                      f"(négatif = pénétration obstacle)")
        env.close()


if __name__ == "__main__":
    main()
