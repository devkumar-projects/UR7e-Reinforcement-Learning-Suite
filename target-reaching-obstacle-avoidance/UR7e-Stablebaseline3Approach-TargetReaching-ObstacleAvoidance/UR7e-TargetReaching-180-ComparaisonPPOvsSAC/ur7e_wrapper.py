# ==============================================================================
# FICHIER : ur7e_wrapper.py
# RÔLE : L'Interface Standardisée Gymnasium (Le "Pont" du projet).
#
# Enveloppe le moteur BulletUR7e pour le rendre compatible avec stable-baselines3.
#   - Action : Box(6,) dans [-1, 1] (vitesses articulaires normalisées).
#   - Observation : Box(21,) (angles + vitesses + poses + cible).
#   - Traduit le retour gym (4 valeurs) vers Gymnasium (5 valeurs).
#
# terminated : fin liée à la TÂCHE (cible atteinte) -> succès.
# truncated  : fin par limite de pas -> pas un échec de la tâche.
# ==============================================================================

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from bullet_ur7e import BulletUR7e


class UR7eReachEnv(gym.Env):
    """Environnement Gymnasium : faire atteindre une cible 3D au UR7e."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, max_episode_len=300,
                 success_threshold=0.005):
        super().__init__()

        gui = (render_mode == "human")
        self.engine = BulletUR7e(
            gui=gui,
            max_episode_len=max_episode_len,
            success_threshold=success_threshold,
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

        obs = self.engine.reset()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs.shape[0],), dtype=np.float32
        )

    def step(self, action):
        obs, reward, done, info = self.engine.step(np.asarray(action, dtype=np.float32))
        reason = info.get("done_reason", "")
        terminated = done and (reason == "target_reached")
        truncated = done and (reason == "max_steps")
        return obs, float(reward), terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.engine.reset()
        return obs, {}

    def render(self):
        pass

    def close(self):
        self.engine.close()
