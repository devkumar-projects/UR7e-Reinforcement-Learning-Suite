# ==============================================================================
# FICHIER : train_ppo_ur7e.py
# RÔLE : Entraînement de l'agent par PPO (on-policy) sur le UR7e.
# ==============================================================================

import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from ur7e_wrapper import UR7eReachEnv

TOTAL_TIMESTEPS = 500_000
LOG_DIR = "logs_ppo_ur7e"
MODEL_NAME = "ppo_ur7e_reach"


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    print("1. Création de l'environnement UR7e (sans GUI)...")
    env = UR7eReachEnv(render_mode=None, max_episode_len=300)
    env = Monitor(env, LOG_DIR)

    print("2. Création de l'agent PPO...")
    model = PPO(
        "MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR,
        learning_rate=3e-4, n_steps=2048, batch_size=64,
        gamma=0.99, gae_lambda=0.95, device="cpu",
    )

    print(f"3. Apprentissage PPO ({TOTAL_TIMESTEPS:,} étapes)...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)

    print("4. Sauvegarde du modèle...")
    model.save(MODEL_NAME)
    env.close()
    print(f"Modèle '{MODEL_NAME}.zip' créé avec succès !")


if __name__ == "__main__":
    main()
