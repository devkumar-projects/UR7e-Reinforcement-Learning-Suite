# ==============================================================================
# FICHIER : test_her_facile.py   (DIAGNOSTIC — jetable)
# RÔLE : Vérifier si HER décolle dans le cas CANONIQUE facile, pour isoler la
#        cause de la non-convergence observée en 360° + départs aléatoires.
#
# Configuration "facile" (une seule différence à la fois retirée) :
#   - cibles AVANT uniquement (x>0), pas 360°
#   - départ FIXE (random_start=False), pas aléatoire
#   - seuil 5 cm, récompense sparse, HER future n_sampled_goal=4 (inchangés)
#
# Interprétation :
#   - si success_rate MONTE  -> HER + plomberie sains ; le coupable est la
#     difficulté (360°/départs aléatoires) -> il faudra un curriculum.
#   - si success_rate reste ~0 -> problème de fond (plomberie/récompense) à
#     corriger avant tout.
#
# Usage : caffeinate -i python test_her_facile.py
# ==============================================================================

import os
import random
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from bullet_ur7e_360 import BulletUR7e360

TOTAL_TIMESTEPS = 50_000
SEED = 0
LOG_DIR = "logs_her_entfix"
SUCCESS_RADIUS = 0.05


# --- Moteur de diagnostic : cibles AVANT uniquement -------------------------
class BulletUR7eFacile(BulletUR7e360):
    """Identique au moteur 360°, mais cibles restreintes à l'avant (x>0)."""

    def sample_target(self, max_tries=500):
        import pybullet as p
        q_current, _ = self._get_joint_states()
        for _ in range(max_tries):
            cand = np.array([
                self.rng.uniform(0.20, self.REACH_MAX),   # x AVANT uniquement
                self.rng.uniform(-0.45, 0.45),            # y restreint
                self.rng.uniform(0.15, 0.70),             # z restreint
            ])
            r = np.linalg.norm(cand - self.SHOULDER_CENTER)
            if not (self.REACH_MIN <= r <= self.REACH_MAX):
                continue
            sol = p.calculateInverseKinematics(self.robot, self.ee_index,
                                               cand.tolist())
            self._set_config(np.array(sol))
            ee = self.get_ee_position()
            if np.linalg.norm(ee - cand) < self.ik_tol:
                self._set_config(q_current)
                return cand
        self._set_config(q_current)
        return np.array([0.4, 0.0, 0.4])


# --- Wrapper HER facile (départ fixe) ---------------------------------------
class UR7eHERFacileEnv(gym.Env):
    metadata = {"render_modes": [None]}

    def __init__(self, seed=None):
        super().__init__()
        # random_start=False -> départ FIXE (config facile)
        self.engine = BulletUR7eFacile(
            gui=False, max_episode_len=300,
            success_threshold=SUCCESS_RADIUS,
            random_start=False, seed=seed,
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        inf = np.inf
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(-inf, inf, shape=(12,), dtype=np.float32),
            "achieved_goal": spaces.Box(-inf, inf, shape=(3,), dtype=np.float32),
            "desired_goal": spaces.Box(-inf, inf, shape=(3,), dtype=np.float32),
        })

    def _make_obs(self):
        angles, velocities = self.engine._get_joint_states()
        ee = self.engine.get_ee_position()
        return {
            "observation": np.concatenate([angles, velocities]).astype(np.float32),
            "achieved_goal": ee.astype(np.float32),
            "desired_goal": np.asarray(self.engine.target, dtype=np.float32),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.engine.reset()
        return self._make_obs(), {}

    def step(self, action):
        _, _, done_engine, info = self.engine.step(action)
        obs = self._make_obs()
        reward = float(self.compute_reward(obs["achieved_goal"],
                                           obs["desired_goal"], info))
        is_success = reward == 0.0
        terminated = bool(is_success)
        truncated = bool(done_engine and not is_success)
        info = dict(info)
        info["is_success"] = is_success
        return obs, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        achieved_goal = np.asarray(achieved_goal, dtype=np.float32)
        desired_goal = np.asarray(desired_goal, dtype=np.float32)
        dist = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return np.where(dist < SUCCESS_RADIUS, 0.0, -1.0).astype(np.float32)

    def close(self):
        self.engine.close()


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    print("TEST C — HER + ent_coef FIXE 0.2 (exploration forcée), config facile")
    env = UR7eHERFacileEnv(seed=SEED)
    env = Monitor(env, LOG_DIR, info_keywords=("is_success",))

    model = SAC(
        "MultiInputPolicy", env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
        verbose=1, learning_rate=3e-4, buffer_size=200_000, batch_size=256,
        gamma=0.99, tau=0.005, train_freq=1, gradient_steps=1,
        learning_starts=1000, device="cpu", seed=SEED,
        ent_coef=0.2,  # FIXE et élevé : force l exploration (au lieu de auto)
    )
    print(f"Apprentissage {TOTAL_TIMESTEPS:,} pas, ent_coef FIXE 0.2. SURVEILLE success_rate :")
    print("  - s'il MONTE -> l'exploration qui s'éteignait était le coupable.")
    print("  - s'il reste bas -> le problème n'est pas l'exploration ; on arrête HER.")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
    env.close()
    print("Test A terminé.")


if __name__ == "__main__":
    main()
