# ==============================================================================
# FICHIER : evaluate_ur7e.py
# RÔLE : Démonstration visuelle + métriques de l'agent UR7e entraîné.
#
# Améliorations issues de l'analyse du PPT du groupe :
#   - On rapporte la DISTANCE MÉDIANE (plus la min/max), pas seulement le taux,
#     pour une comparaison honnête avec la partie cinématique (seuil mm).
#
# Usage :
#   python evaluate_ur7e.py sac
#   python evaluate_ur7e.py ppo
# ==============================================================================

import sys
import time
import numpy as np
from stable_baselines3 import SAC, PPO
from ur7e_wrapper import UR7eReachEnv

N_EPISODES = 20


def main():
    algo = sys.argv[1].lower() if len(sys.argv) > 1 else "sac"
    if algo == "sac":
        model = SAC.load("sac_ur7e_reach")
        print("Modèle SAC chargé.")
    elif algo == "ppo":
        model = PPO.load("ppo_ur7e_reach")
        print("Modèle PPO chargé.")
    else:
        print(f"Algo inconnu : {algo}. Utilise 'sac' ou 'ppo'.")
        return

    env = UR7eReachEnv(render_mode="human", max_episode_len=300)

    successes = 0
    distances_finales = []
    longueurs = []

    for ep in range(N_EPISODES):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        dist = info["distance"]
        distances_finales.append(dist)
        longueurs.append(steps)
        reached = info.get("done_reason") == "target_reached"
        successes += int(reached)

        statut = "ATTEINTE" if reached else "échec (timeout)"
        print(f"Épisode {ep+1:2d} | {statut:16s} | "
              f"distance = {dist*1000:.1f} mm | {steps} pas")
        time.sleep(0.3)

    d = np.array(distances_finales) * 1000  # en mm
    print("\n" + "=" * 55)
    print(f"Algo : {algo.upper()}  |  {N_EPISODES} cibles")
    print(f"Taux de succès (<5mm) : {successes}/{N_EPISODES} "
          f"({100*successes/N_EPISODES:.0f}%)")
    print(f"Distance médiane : {np.median(d):.1f} mm")
    print(f"Distance min     : {np.min(d):.1f} mm")
    print(f"Distance max     : {np.max(d):.1f} mm")
    print(f"Longueur moyenne : {np.mean(longueurs):.1f} pas")
    print("=" * 55)

    env.close()


if __name__ == "__main__":
    main()
