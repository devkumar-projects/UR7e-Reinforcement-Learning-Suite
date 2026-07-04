# ==============================================================================
# FICHIER : train_sac_ur7e_360.py
# RÔLE : Entraînement SAC sur l'espace de travail COMPLET du UR7e (360°),
#        avec poses de départ aléatoires. Produit le modèle étalon 360°.
#
# Différences avec train_sac_ur7e.py :
#   - utilise l'environnement UR7eReach360Env (cibles 360° validées IK,
#     départs aléatoires non singuliers) ;
#   - 250 000 étapes (étude de précision) ;
#   - graine fixée pour la reproductibilité ;
#   - sauvegarde sous un nom distinct pour ne pas écraser le modèle existant.
# ==============================================================================

import os
import numpy as np
import random
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from ur7e_wrapper_360 import UR7eReach360Env

TOTAL_TIMESTEPS = 250_000
SEED = 0
LOG_DIR = "logs_sac_ur7e_360"
MODEL_NAME = "sac_ur7e_360_reach"      # nom distinct : n'écrase pas sac_ur7e_reach


def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    set_global_seeds(SEED)

    print("1. Création de l'environnement UR7e espace complet (360°)...")
    env = UR7eReach360Env(render_mode=None, max_episode_len=300,
                          success_threshold=0.005,
                          random_start=True,        # départs aléatoires
                          seed=SEED)
    env = Monitor(env, LOG_DIR)

    print("2. Création de l'agent SAC...")
    model = SAC(
        "MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR,
        learning_rate=3e-4, buffer_size=300_000, batch_size=256,
        gamma=0.99, tau=0.005, train_freq=1, gradient_steps=1,
        learning_starts=1000, device="cpu", seed=SEED,
    )

    print(f"3. Apprentissage SAC ({TOTAL_TIMESTEPS:,} étapes, espace complet)...")
    print("   NB : départs aléatoires + cibles 360° validées IK -> resets plus")
    print("   coûteux qu'en version avant. Comptez plus de temps qu'à l'identique.")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)

    print("4. Sauvegarde du modèle...")
    model.save(MODEL_NAME)
    env.close()
    print(f"Modèle '{MODEL_NAME}.zip' créé avec succès !")


if __name__ == "__main__":
    main()
