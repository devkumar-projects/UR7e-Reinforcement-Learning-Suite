# ==============================================================================
# FICHIER : ur7e_wrapper_her.py
# RÔLE : Wrapper GOAL-CONDITIONED pour entraîner SAC + HER sur le UR7e 360°.
#        Objectif : amener l'effecteur dans une ZONE de 5 cm autour de la cible
#        (zone suffisamment proche pour qu'un asservissement IK prenne ensuite
#        le relais — version hybride ultérieure).
#
# Format imposé par HER (Hindsight Experience Replay) : l'observation est un
# DICTIONNAIRE à 3 clés :
#   - 'observation'  : état propre du robot = angles (6) + vitesses (6) = 12D
#   - 'achieved_goal': position 3D actuelle de l'effecteur (tool0)
#   - 'desired_goal' : position 3D de la cible
#
# Récompense SPARSE : 0 si distance(achieved, desired) < 5 cm, sinon -1.
# compute_reward est VECTORISÉE (HER l'appelle sur des lots de transitions
# réécrites a posteriori).
#
# S'appuie sur le moteur étalon 360° (bullet_ur7e_360) SANS le modifier :
#   cibles 360° validées IK, départs aléatoires non singuliers, contrôle vitesse.
# ==============================================================================

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from bullet_ur7e_360 import BulletUR7e360

SUCCESS_RADIUS = 0.05      # 5 cm : rayon de la zone-cible (déclenchera l'IK plus tard)


class UR7eReachHEREnv(gym.Env):
    """Environnement goal-conditioned (compatible HER) pour le UR7e 360°."""

    metadata = {"render_modes": [None, "human"]}

    def __init__(self, render_mode=None, max_episode_len=300,
                 random_start=True, seed=None):
        super().__init__()
        gui = render_mode == "human"
        # success_threshold du moteur réglé à 5 cm : l'épisode se conclut dès
        # que l'effecteur entre dans la zone de 5 cm.
        self.engine = BulletUR7e360(
            gui=gui, max_episode_len=max_episode_len,
            success_threshold=SUCCESS_RADIUS,
            random_start=random_start, seed=seed,
        )

        # Action : 6 vitesses articulaires normalisées dans [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,),
                                       dtype=np.float32)

        # Observation : dictionnaire goal-conditioned
        inf = np.inf
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(-inf, inf, shape=(12,), dtype=np.float32),
            "achieved_goal": spaces.Box(-inf, inf, shape=(3,), dtype=np.float32),
            "desired_goal": spaces.Box(-inf, inf, shape=(3,), dtype=np.float32),
        })

    # ----------------------------------------------------------------------
    def _make_obs(self):
        """Construit le dictionnaire goal-conditioned depuis l'état du moteur."""
        angles, velocities = self.engine._get_joint_states()
        ee = self.engine.get_ee_position()
        return {
            "observation": np.concatenate([angles, velocities]).astype(np.float32),
            "achieved_goal": ee.astype(np.float32),
            "desired_goal": np.asarray(self.engine.target, dtype=np.float32),
        }

    # ----------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.engine.reset()                 # gère départ aléatoire + cible 360° IK
        return self._make_obs(), {}

    def step(self, action):
        # On délègue la physique au moteur, mais on RECALCULE la récompense en
        # sparse goal-conditioned (cohérent avec compute_reward, requis par HER).
        _, _, done_engine, info = self.engine.step(action)

        obs = self._make_obs()
        achieved = obs["achieved_goal"]
        desired = obs["desired_goal"]

        reward = float(self.compute_reward(achieved, desired, info))
        # succès = être entré dans la zone de 5 cm
        is_success = reward == 0.0
        terminated = bool(is_success)
        # troncature = fin d'épisode du moteur sans succès (max_steps atteint)
        truncated = bool(done_engine and not is_success)

        info = dict(info)
        info["is_success"] = is_success
        return obs, reward, terminated, truncated, info

    # ----------------------------------------------------------------------
    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Récompense SPARSE, VECTORISÉE (HER la rappelle sur des lots) :
          0  si distance < SUCCESS_RADIUS (dans la zone de 5 cm)
          -1 sinon

        achieved_goal / desired_goal peuvent être de forme (3,) ou (N, 3).
        """
        achieved_goal = np.asarray(achieved_goal, dtype=np.float32)
        desired_goal = np.asarray(desired_goal, dtype=np.float32)
        dist = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return np.where(dist < SUCCESS_RADIUS, 0.0, -1.0).astype(np.float32)

    def close(self):
        self.engine.close()
