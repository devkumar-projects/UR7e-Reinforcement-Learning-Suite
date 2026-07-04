# ==============================================================================
# FICHIER : ur7e_wrapper_360.py
# RÔLE : Pont Gymnasium pour le moteur étalon espace complet (BulletUR7e360).
#        Équivalent de ur7e_wrapper.py, branché sur bullet_ur7e_360.
# ==============================================================================

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from bullet_ur7e_360 import BulletUR7e360


class UR7eReach360Env(gym.Env):
    """Environnement Gymnasium pour le UR7e sur l'espace de travail complet."""

    metadata = {"render_modes": [None]}

    def __init__(self, render_mode=None, max_episode_len=300,
                 success_threshold=0.005, random_start=True, seed=None):
        super().__init__()
        gui = render_mode == "human"
        self.engine = BulletUR7e360(
            gui=gui, max_episode_len=max_episode_len,
            success_threshold=success_threshold,
            random_start=random_start, seed=seed,
        )
        # Action : 6 vitesses articulaires normalisées dans [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,),
                                       dtype=np.float32)
        # Observation : dimension 21
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
