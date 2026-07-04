# ==============================================================================
# FICHIER : ur7e_wrapper_crossing.py
# RÔLE : Pont Gymnasium pour BulletUR7eCrossing — franchissement d'obstacle
#        cylindrique (zone départ -> zone d'évitement). Observation 24D.
# ==============================================================================

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from bullet_ur7e_crossing import BulletUR7eCrossing


class UR7eCrossingEnv(gym.Env):
    """Franchissement d'un cylindre : zone départ -> zone d'évitement (obs 24D)."""

    metadata = {"render_modes": [None, "human"]}
    OBS_DIM = 49      # 25 base + 18 répulsifs + 6 attractifs

    def __init__(self, render_mode=None, max_episode_len=300,
                 cyl_radius=0.10, cyl_height=0.50, seed=None):
        super().__init__()
        gui = render_mode == "human"
        self.engine = BulletUR7eCrossing(
            gui=gui, max_episode_len=max_episode_len,
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
        terminated = info.get("done_reason") in (
            "crossed", "collision", "joint_limits", "singularity")
        truncated = done and not terminated
        return obs, reward, terminated, truncated, info

    def close(self):
        self.engine.close()
