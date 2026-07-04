# ==============================================================================
# FICHIER : continue_train_her_360.py
# RÔLE : REPRENDRE l'entraînement du modèle SAC+HER 360° déjà entraîné, sans
#        repartir de zéro. Recharge les réseaux appris et continue.
#
# Objectif : prolonger de +250k pas pour viser un success_rate > 53%.
#
# Points techniques importants :
#   - SAC.load(..., env=env) : l'env est REQUIS pour HER (compute_reward).
#   - reset_num_timesteps=False : le compteur de pas CONTINUE (251k, 252k, ...)
#     au lieu de repartir à 0 -> logs et scheduling cohérents.
#   - Le replay buffer du run initial n'a pas été sauvegardé : on redémarre donc
#     avec un buffer vide (la POLITIQUE est conservée ; petit transitoire au
#     début, le temps que le buffer se re-remplisse). C'est normal et sans danger.
#   - CETTE FOIS on sauvegarde aussi le buffer (save_replay_buffer) pour que les
#     reprises FUTURES soient sans couture.
#
# Usage : caffeinate -i python continue_train_her_360.py
# ==============================================================================

import os
import random
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from ur7e_wrapper_her import UR7eReachHEREnv

ADDITIONAL_TIMESTEPS = 250_000          # pas SUPPLÉMENTAIRES
SEED = 0
LOG_DIR = "logs_sac_her_360"
MODEL_IN = "sac_ur7e_her_360_reach"     # modèle existant à reprendre
MODEL_OUT = "sac_ur7e_her_360_reach"    # on écrase (le modèle prolongé remplace)
BUFFER_OUT = "sac_ur7e_her_360_buffer"  # buffer sauvegardé pour reprises futures


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    if not os.path.exists(MODEL_IN + ".zip"):
        print(f"ERREUR : {MODEL_IN}.zip introuvable. Lance d'abord l'entraînement initial.")
        return

    print("1. Recréation de l'environnement goal-conditioned (HER)...")
    env = UR7eReachHEREnv(render_mode=None, max_episode_len=300,
                          random_start=True, seed=SEED)
    env = Monitor(env, LOG_DIR, info_keywords=("is_success",))

    print(f"2. Rechargement du modèle existant '{MODEL_IN}'...")
    # env=env est OBLIGATOIRE pour HER (le buffer a besoin de compute_reward).
    model = SAC.load(MODEL_IN, env=env, device="cpu")
    print(f"   Modèle rechargé. Pas déjà effectués : {model.num_timesteps:,}")

    # Si un buffer avait été sauvegardé précédemment, on le rechargerait ici :
    if os.path.exists(BUFFER_OUT + ".pkl"):
        print(f"   Buffer trouvé ({BUFFER_OUT}.pkl) -> rechargement (reprise sans couture).")
        model.load_replay_buffer(BUFFER_OUT)
    else:
        print("   Pas de buffer sauvegardé : redémarrage avec buffer vide.")
        # CORRECTIF : avec un buffer vide et reset_num_timesteps=False, le compteur
        # est déjà à 250k, donc la condition "num_timesteps > learning_starts" est
        # immédiatement vraie -> SB3 tente d'apprendre AVANT qu'un épisode complet
        # soit dans le buffer (HER exige des épisodes entiers) -> crash.
        # On repousse learning_starts APRÈS le compteur actuel pour forcer la
        # collecte de quelques épisodes complets avant le premier apprentissage.
        model.learning_starts = model.num_timesteps + 1000   # > longueur d'épisode (300)
        print(f"   learning_starts repoussé à {model.learning_starts:,} "
              f"(collecte d'épisodes complets avant d'apprendre).")

    print(f"3. Poursuite de l'apprentissage (+{ADDITIONAL_TIMESTEPS:,} pas)...")
    print("   SURVEILLE success_rate : objectif > 0.53. Au tout début, un léger")
    print("   creux est possible (buffer vide qui se re-remplit) puis ça repart.")
    model.learn(
        total_timesteps=ADDITIONAL_TIMESTEPS,
        reset_num_timesteps=False,          # CONTINUE le compteur (ne repart pas à 0)
        progress_bar=True,
    )

    print("4. Sauvegarde du modèle prolongé...")
    model.save(MODEL_OUT)
    # Cette fois on sauvegarde aussi le buffer -> reprises futures sans couture.
    print("   Sauvegarde du replay buffer (pour reprises futures sans couture)...")
    model.save_replay_buffer(BUFFER_OUT)

    env.close()
    print(f"Terminé. Modèle '{MODEL_OUT}.zip' mis à jour "
          f"({model.num_timesteps:,} pas au total).")
    print(f"Buffer sauvegardé : '{BUFFER_OUT}.pkl'.")


if __name__ == "__main__":
    main()
