# ==============================================================================
# FICHIER : train_sac_crossing.py
# RÔLE : Entraîner SAC (DE ZÉRO) au FRANCHISSEMENT d'un obstacle cylindrique :
#        partir de la zone "départ/collision" (collé à l'obstacle) et rejoindre
#        la zone d'évitement de l'autre côté, sans collision, marge >= 5 cm.
#
# MÉTRIQUES MONITEUR :
#   - collision_rate   : fraction d'épisodes finis en collision
#   - success_rate     : franchissement réussi (zone évitement + marge, sans collision)
#   - mean_distance_cm : "distance au franchissement" finale moyenne
#
# Reprise possible : modèle + buffer (continue_train_crossing.py).
# Usage : caffeinate -i python train_sac_crossing.py
# ==============================================================================

import os
import random
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from ur7e_wrapper_crossing import UR7eCrossingEnv

TOTAL_TIMESTEPS = 500_000
SEED = 0
CYL_RADIUS = 0.10
CYL_HEIGHT = 0.50
LOG_DIR = "logs_sac_crossing"
MODEL_OUT = "sac_ur7e_crossing_reach"
BUFFER_OUT = "sac_ur7e_crossing_buffer"


class StatsCallback(BaseCallback):
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
    env = UR7eCrossingEnv(
        render_mode=None, max_episode_len=300,
        cyl_radius=CYL_RADIUS, cyl_height=CYL_HEIGHT, seed=seed,
    )
    _orig_step = env.step

    def step_with_flags(action):
        obs, reward, terminated, truncated, info = _orig_step(action)
        info = dict(info)
        info["is_success"] = 1.0 if info.get("done_reason") == "crossed" else 0.0
        info["collision"] = float(info.get("collision", 0.0))
        return obs, reward, terminated, truncated, info

    env.step = step_with_flags
    return env


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    print("1. Environnement FRANCHISSEMENT (cylindre "
          f"r={CYL_RADIUS*100:.0f}cm x {CYL_HEIGHT*100:.0f}cm)...")
    env = make_env(SEED)
    env = Monitor(env, LOG_DIR, info_keywords=("is_success", "distance", "collision"))

    print("2. Agent SAC (MlpPolicy, ent_coef auto) — de zéro...")
    model = SAC(
        "MlpPolicy", env,
        verbose=1, tensorboard_log=LOG_DIR,
        learning_rate=3e-4, buffer_size=500_000, batch_size=256,
        gamma=0.99, tau=0.005, train_freq=1, gradient_steps=1,
        learning_starts=1000, device="cpu", seed=SEED,
    )

    print(f"3. Apprentissage ({TOTAL_TIMESTEPS:,} pas)...")
    print("   SURVEILLE : success_rate (monter), collision_rate (DESCENDRE),")
    print("   mean_distance_cm (distance au franchissement, doit descendre).")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=StatsCallback(),
                progress_bar=True)

    print("4. Sauvegarde (modèle + buffer)...")
    model.save(MODEL_OUT)
    model.save_replay_buffer(BUFFER_OUT)
    env.close()
    print(f"Terminé. '{MODEL_OUT}.zip' + '{BUFFER_OUT}.pkl' sauvés.")


if __name__ == "__main__":
    main()
