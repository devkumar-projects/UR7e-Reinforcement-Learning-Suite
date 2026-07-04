# ==============================================================================
# FICHIER : train_sac_obstacle.py
# RÔLE : Entraîner SAC (DE ZÉRO) au reaching 360° AVEC évitement d'un obstacle
#        sphérique placé aléatoirement sur le chemin effecteur->cible.
#
# Base : la recette shaped (départ home fixe, shaping de freinage, seuil 5 cm),
# à laquelle on ajoute l'obstacle (observation 28D) + pénalités de collision et
# de proximité. Réentraînement complet (l'obs change de taille, pas de reprise
# du modèle 99% sans obstacle).
#
# Panneau de stats : success_rate, collision_rate, mean_distance_cm.
# Reprise possible : sauvegarde modèle + buffer (continue_train_obstacle.py).
#
# Usage : caffeinate -i python train_sac_obstacle.py
# ==============================================================================

import os
import random
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from ur7e_wrapper_obstacle import UR7eReachObstacleEnv

TOTAL_TIMESTEPS = 500_000         # évitement = plus dur que reaching nu
SEED = 0
SUCCESS_THRESHOLD = 0.05
OBSTACLE_RADIUS = 0.10            # 10 cm de rayon : obstacle significatif
LOG_DIR = "logs_sac_obstacle"
MODEL_OUT = "sac_ur7e_obstacle_reach"
BUFFER_OUT = "sac_ur7e_obstacle_buffer"


class StatsCallback(BaseCallback):
    """Ajoute mean_distance_cm et collision_rate au panneau de stats SB3."""

    def _on_step(self):
        if self.num_timesteps % 1000 != 0:
            return True
        buf = self.model.ep_info_buffer
        if buf:
            recent = list(buf)[-100:]
            dists = [ep.get("distance") for ep in recent
                     if ep.get("distance") is not None]
            cols = [ep.get("collision", 0.0) for ep in recent]
            if dists:
                self.logger.record("rollout/mean_distance_cm",
                                   float(np.mean(dists)) * 100.0)
            if cols:
                self.logger.record("rollout/collision_rate", float(np.mean(cols)))
        return True


def make_env(seed):
    env = UR7eReachObstacleEnv(
        render_mode=None, max_episode_len=300,
        success_threshold=SUCCESS_THRESHOLD,
        random_start=False,             # départ home fixe (recette gagnante)
        obstacle_radius=OBSTACLE_RADIUS,
        seed=seed,
    )
    _orig_step = env.step

    def step_with_flags(action):
        obs, reward, terminated, truncated, info = _orig_step(action)
        info = dict(info)
        info["is_success"] = 1.0 if info.get("done_reason") == "target_reached" else 0.0
        info["collision"] = float(info.get("collision", 0.0))
        return obs, reward, terminated, truncated, info

    env.step = step_with_flags
    return env


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    print("1. Environnement reaching + évitement obstacle (obs 28D, "
          f"obstacle r={OBSTACLE_RADIUS*100:.0f}cm)...")
    env = make_env(SEED)
    env = Monitor(env, LOG_DIR, info_keywords=("is_success", "distance", "collision"))

    print("2. Agent SAC (MlpPolicy, ent_coef auto) — réentraînement de zéro...")
    model = SAC(
        "MlpPolicy", env,
        verbose=1, tensorboard_log=LOG_DIR,
        learning_rate=3e-4, buffer_size=500_000, batch_size=256,
        gamma=0.99, tau=0.005, train_freq=1, gradient_steps=1,
        learning_starts=1000, device="cpu", seed=SEED,
    )

    print(f"3. Apprentissage ({TOTAL_TIMESTEPS:,} pas)...")
    print("   SURVEILLE : success_rate (monter), collision_rate (DESCENDRE vers 0),")
    print("   mean_distance_cm. L'agent doit apprendre à CONTOURNER l'obstacle.")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=StatsCallback(),
                progress_bar=True)

    print("4. Sauvegarde (modèle + buffer)...")
    model.save(MODEL_OUT)
    model.save_replay_buffer(BUFFER_OUT)
    env.close()
    print(f"Terminé. Modèle '{MODEL_OUT}.zip' et buffer '{BUFFER_OUT}.pkl' sauvés.")


if __name__ == "__main__":
    main()
