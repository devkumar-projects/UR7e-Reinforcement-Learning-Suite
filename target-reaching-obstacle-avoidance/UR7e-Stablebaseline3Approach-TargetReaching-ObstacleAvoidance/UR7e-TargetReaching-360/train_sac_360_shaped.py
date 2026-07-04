# ==============================================================================
# FICHIER : train_sac_360_shaped.py
# RÔLE : Entraîner SAC sur l'espace 360° COMPLET avec la recette exacte du
#        modèle 100% : reward shaping (bonus proximité <3cm + pénalité vitesse
#        <5cm) ET départ depuis la pose de repos FIXE. Seuil 5 cm.
#
# DÉMARCHE (isolation de variable) : c'est la config qui obtenait 100% sur
# l'espace frontal compact, à laquelle on ne change QUE l'espace de cibles
# (frontal -> coquille 360° complète). On mesure ainsi l'effet pur de
# l'élargissement spatial, toutes choses égales par ailleurs.
#
# Panneau de stats enrichi : success_rate, mean_distance_cm.
# Reprise possible : sauvegarde modèle + buffer.
#
# Usage : caffeinate -i python train_sac_360_shaped.py
# ==============================================================================

import os
import random
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from ur7e_wrapper_360_shaped import UR7eReach360ShapedEnv

TOTAL_TIMESTEPS = 1_000_000
SEED = 0
SUCCESS_THRESHOLD = 0.05          # 5 cm (comparaison directe avec le modèle 100%)
LOG_DIR = "logs_sac_360_shaped"
MODEL_OUT = "sac_ur7e_360_shaped_reach"
BUFFER_OUT = "sac_ur7e_360_shaped_buffer"


class StatsCallback(BaseCallback):
    """Ajoute mean_distance_cm au panneau de stats SB3 (lu via ep_info_buffer)."""

    def _on_step(self):
        if self.num_timesteps % 1000 != 0:
            return True
        buf = self.model.ep_info_buffer
        if buf:
            recent = list(buf)[-100:]
            dists = [ep.get("distance") for ep in recent
                     if ep.get("distance") is not None]
            if dists:
                self.logger.record("rollout/mean_distance_cm",
                                   float(np.mean(dists)) * 100.0)
        return True


def make_env(seed):
    env = UR7eReach360ShapedEnv(
        render_mode=None, max_episode_len=300,
        success_threshold=SUCCESS_THRESHOLD,
        random_start=False,            # POSE HOME FIXE (recette 100%)
        seed=seed,
    )
    _orig_step = env.step

    def step_with_success(action):
        obs, reward, terminated, truncated, info = _orig_step(action)
        info = dict(info)
        info["is_success"] = 1.0 if info.get("done_reason") == "target_reached" else 0.0
        return obs, reward, terminated, truncated, info

    env.step = step_with_success
    return env


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    print("1. Environnement 360° + shaping 100% + départ home fixe (seuil 5 cm)...")
    env = make_env(SEED)
    env = Monitor(env, LOG_DIR, info_keywords=("is_success", "distance"))

    print("2. Agent SAC (MlpPolicy, ent_coef auto)...")
    model = SAC(
        "MlpPolicy", env,
        verbose=1, tensorboard_log=LOG_DIR,
        learning_rate=3e-4, buffer_size=300_000, batch_size=256,
        gamma=0.99, tau=0.005, train_freq=1, gradient_steps=1,
        learning_starts=1000, device="cpu", seed=SEED,
    )

    print(f"3. Apprentissage ({TOTAL_TIMESTEPS:,} pas)...")
    print("   SURVEILLE success_rate (objectif >0.8) et mean_distance_cm (doit")
    print("   descendre sous 5 cm). Comparaison directe avec le modèle 100%.")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=StatsCallback(),
                progress_bar=True)

    print("4. Sauvegarde (modèle + buffer)...")
    model.save(MODEL_OUT)
    model.save_replay_buffer(BUFFER_OUT)
    env.close()
    print(f"Terminé. Modèle '{MODEL_OUT}.zip' et buffer '{BUFFER_OUT}.pkl' sauvés.")


if __name__ == "__main__":
    main()
