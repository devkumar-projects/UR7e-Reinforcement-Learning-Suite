# ==============================================================================
# FICHIER : ur7e_wrapper_hybrid.py
# RÔLE : Pont Gymnasium pour BulletUR7eHybrid — phase RL d'évitement d'un
#        cylindre dans le pipeline IK->RL->IK. Observation 28D.
#        L'agent démarre au point d'ENTRÉE et doit rejoindre le point de SORTIE
#        (seuil 5cm) en contournant le cylindre, de préférence côté intérieur.
# ==============================================================================

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from bullet_ur7e_hybrid import BulletUR7eHybrid


class UR7eHybridEnv(gym.Env):
    """Phase RL d'évitement de cylindre (pipeline IK->RL->IK), obs 28D."""

    metadata = {"render_modes": [None, "human"]}
    OBS_DIM = 28      # 12 + 3 + 3 + 3 + 7(cylindre)

    def __init__(self, render_mode=None, max_episode_len=300,
                 success_threshold=0.05, cyl_radius=0.25, cyl_height=1.0,
                 seed=None):
        super().__init__()
        gui = render_mode == "human"
        self.engine = BulletUR7eHybrid(
            gui=gui, max_episode_len=max_episode_len,
            success_threshold=success_threshold,
            cyl_radius=cyl_radius, cyl_height=cyl_height, seed=seed,
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,),
                                       dtype=np.float32)
        high = np.full(self.OBS_DIM, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high,
                                            dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.engine.reset()
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.engine.step(action)
        terminated = info.get("done_reason") in ("exit_reached", "collision")
        truncated = done and not terminated
        return obs, reward, terminated, truncated, info

    def close(self):
        self.engine.close()
