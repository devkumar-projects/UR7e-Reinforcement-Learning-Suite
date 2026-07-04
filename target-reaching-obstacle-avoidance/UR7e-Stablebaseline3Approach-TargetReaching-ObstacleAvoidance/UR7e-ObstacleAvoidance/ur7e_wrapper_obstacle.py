# ==============================================================================
# FICHIER : ur7e_wrapper_obstacle.py
# RÔLE : Pont Gymnasium pour BulletUR7eObstacle — reaching 360° AVEC évitement
#        d'un obstacle sphérique placé aléatoirement sur le segment
#        effecteur_initial -> cible. Observation étendue à 28D (ajout obstacle).
# ==============================================================================

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from bullet_ur7e_obstacle import BulletUR7e360 as BulletUR7eObstacle


class UR7eReachObstacleEnv(gym.Env):
    """UR7e reaching 360° avec évitement d'obstacle (obs 28D)."""

    metadata = {"render_modes": [None, "human"]}

    OBS_DIM = 28      # 12 + 3 + 3 + 3 + 3 + 1 + 3

    def __init__(self, render_mode=None, max_episode_len=300,
                 success_threshold=0.05, random_start=False,
                 obstacle_radius=0.10, seed=None):
        super().__init__()
        gui = render_mode == "human"
        self.engine = BulletUR7eObstacle(
            gui=gui, max_episode_len=max_episode_len,
            success_threshold=success_threshold,
            random_start=random_start, obstacle_radius=obstacle_radius,
            seed=seed,
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
        terminated = info.get("done_reason") in ("target_reached", "collision")
        truncated = done and not terminated
        return obs, reward, terminated, truncated, info

    def close(self):
        self.engine.close()
