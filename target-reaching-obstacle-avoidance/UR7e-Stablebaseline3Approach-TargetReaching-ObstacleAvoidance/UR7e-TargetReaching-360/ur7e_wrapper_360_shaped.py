# ==============================================================================
# FICHIER : ur7e_wrapper_360_shaped.py
# RÔLE : Pont Gymnasium pour BulletUR7e360Shaped — moteur espace 360° complet
#        AVEC le reward shaping du modèle 100% (bonus proximité + pénalité
#        vitesse en approche), et DÉPART DEPUIS LA POSE DE REPOS FIXE.
#
# Démarche d'isolation de variable : on reprend la recette exacte qui obtenait
# 100% sur l'espace frontal compact (reward shaping + départ home fixe), et on
# ne change QU'UNE chose : l'espace de cibles passe du secteur frontal à la
# coquille 360° atteignable complète.
# ==============================================================================

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from bullet_ur7e_360_shaped import BulletUR7e360  # moteur avec shaping greffé


class UR7eReach360ShapedEnv(gym.Env):
    """UR7e, espace 360° complet, reward shaping 100%, départ home fixe."""

    metadata = {"render_modes": [None, "human"]}

    def __init__(self, render_mode=None, max_episode_len=300,
                 success_threshold=0.05,        # 5 cm (identique au modèle 100%)
                 random_start=False,            # POSE HOME FIXE (recette 100%)
                 seed=None):
        super().__init__()
        gui = render_mode == "human"
        self.engine = BulletUR7e360(
            gui=gui, max_episode_len=max_episode_len,
            success_threshold=success_threshold,
            random_start=random_start, seed=seed,
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,),
                                       dtype=np.float32)
        high = np.full(21, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high,
                                            dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.engine.reset()
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.engine.step(action)
        terminated = info.get("done_reason") == "target_reached"
        truncated = done and not terminated
        return obs, reward, terminated, truncated, info

    def close(self):
        self.engine.close()
