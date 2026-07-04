# ==============================================================================
# FICHIER : train_sac_her_360.py
# RÔLE : Entraînement SAC + HER sur le UR7e 360°, objectif "zone 5 cm".
#        HER (Hindsight Experience Replay) réécrit les épisodes ratés comme des
#        succès pour un autre but -> apprentissage beaucoup plus rapide sur une
#        tâche d'atteinte à récompense rare.
#
# Configuration (choix validés) :
#   - observation goal-conditioned : angles + vitesses (12D) + achieved/desired
#   - récompense SPARSE 0 / -1 (seuil 5 cm)
#   - stratégie de réécriture : future, n_sampled_goal=4
#   - moteur 360° : cibles validées IK, départs aléatoires non singuliers
#
# Sauvegarde sous un nom DISTINCT : n'écrase aucun modèle existant.
# Usage : caffeinate -i python train_sac_her_360.py
# ==============================================================================

import os
import random
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from ur7e_wrapper_her import UR7eReachHEREnv

TOTAL_TIMESTEPS = 250_000
SEED = 0
LOG_DIR = "logs_sac_her_360"
MODEL_NAME = "sac_ur7e_her_360_reach"      # nom distinct


def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    set_global_seeds(SEED)

    print("1. Création de l'environnement goal-conditioned (HER, zone 5 cm)...")
    env = UR7eReachHEREnv(render_mode=None, max_episode_len=300,
                          random_start=True, seed=SEED)
    env = Monitor(env, LOG_DIR, info_keywords=("is_success",))

    print("2. Création de l'agent SAC + HerReplayBuffer...")
    model = SAC(
        "MultiInputPolicy",                 # requis pour une observation dict
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,               # 4 buts réécrits par transition réelle
            goal_selection_strategy="future",
        ),
        verbose=1, tensorboard_log=LOG_DIR,
        learning_rate=3e-4, buffer_size=300_000, batch_size=256,
        gamma=0.99, tau=0.005, train_freq=1, gradient_steps=1,
        learning_starts=1000, device="cpu", seed=SEED,
        ent_coef=0.2,   # FIXE (pas "auto") : empêche l'exploration de s'éteindre
                        # trop tôt. Correctif validé par diagnostic : avec "auto",
                        # l'entropie chutait à ~0.009 en 15k pas, gelant l'agent
                        # avant qu'il découvre comment atteindre le but en sparse.
    )

    print(f"3. Apprentissage SAC+HER ({TOTAL_TIMESTEPS:,} étapes)...")
    print("   HER réécrit les échecs en succès rétrospectifs : ep_len_mean")
    print("   devrait chuter nettement plus vite qu'en SAC simple.")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)

    print("4. Sauvegarde du modèle...")
    model.save(MODEL_NAME)
    env.close()
    print(f"Modèle '{MODEL_NAME}.zip' créé avec succès !")
    print("NB : pour recharger, fournir aussi l'env (HER a besoin de compute_reward).")


if __name__ == "__main__":
    main()
