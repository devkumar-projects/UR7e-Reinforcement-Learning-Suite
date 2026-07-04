# ==============================================================================
# FICHIER : continue_train_crossing.py
# RÔLE : REPRENDRE l'entraînement de franchissement (modèle + buffer).
# Usage : caffeinate -i python continue_train_crossing.py
# ==============================================================================

import os
import random
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from train_sac_crossing import (
    make_env, StatsCallback, LOG_DIR, MODEL_OUT, BUFFER_OUT, SEED,
)

ADDITIONAL_TIMESTEPS = 250_000


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    if not os.path.exists(MODEL_OUT + ".zip"):
        print(f"ERREUR : {MODEL_OUT}.zip introuvable. Lance d'abord l'entraînement initial.")
        return

    print("1. Recréation de l'environnement...")
    env = make_env(SEED)
    env = Monitor(env, LOG_DIR, info_keywords=("is_success", "distance", "collision"))

    print(f"2. Rechargement du modèle '{MODEL_OUT}'...")
    model = SAC.load(MODEL_OUT, env=env, device="cpu")
    print(f"   Pas déjà effectués : {model.num_timesteps:,}")

    if os.path.exists(BUFFER_OUT + ".pkl"):
        print(f"   Rechargement du buffer '{BUFFER_OUT}.pkl'...")
        model.load_replay_buffer(BUFFER_OUT)
        print(f"   Buffer rechargé ({model.replay_buffer.size():,} transitions).")
    else:
        print("   Pas de buffer : redémarrage buffer vide.")
        model.learning_starts = model.num_timesteps + 1000

    print(f"3. Poursuite (+{ADDITIONAL_TIMESTEPS:,} pas)...")
    model.learn(
        total_timesteps=ADDITIONAL_TIMESTEPS,
        callback=StatsCallback(),
        reset_num_timesteps=False,
        progress_bar=True,
    )

    print("4. Sauvegarde...")
    model.save(MODEL_OUT)
    model.save_replay_buffer(BUFFER_OUT)
    env.close()
    print(f"Terminé. {model.num_timesteps:,} pas au total.")


if __name__ == "__main__":
    main()
