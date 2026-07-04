# ==============================================================================
# FICHIER : train_sac_ur7e.py
# RÔLE : Entraînement de l'agent par SAC (off-policy) sur le UR7e.
# ==============================================================================

import os
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from ur7e_wrapper import UR7eReachEnv

TOTAL_TIMESTEPS = 250_000
LOG_DIR = "logs_sac_ur7e"
MODEL_NAME = "sac_ur7e_reach"


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    print("1. Création de l'environnement UR7e (sans GUI)...")
    env = UR7eReachEnv(render_mode=None, max_episode_len=300)
    env = Monitor(env, LOG_DIR)

    print("2. Création de l'agent SAC...")
    model = SAC(
        "MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR,
        learning_rate=3e-4, buffer_size=200_000, batch_size=256,
        gamma=0.99, tau=0.005, train_freq=1, gradient_steps=1,
        learning_starts=1000, device="cpu",
    )

    print(f"3. Apprentissage SAC ({TOTAL_TIMESTEPS:,} étapes)...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)

    print("4. Sauvegarde du modèle...")
    model.save(MODEL_NAME)
    env.close()
    print(f"Modèle '{MODEL_NAME}.zip' créé avec succès !")


if __name__ == "__main__":
    main()
