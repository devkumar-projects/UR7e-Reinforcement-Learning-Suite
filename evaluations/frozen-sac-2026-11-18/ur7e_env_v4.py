"""
Environnement RL UR7e v4 — Optimisé pour couverture maximale
──────────────────────────────────────────────────────────────
Différences vs v3 :
  - Départ TOUJOURS depuis HOME fixe (pas de bruit)
  - Curriculum sur la précision (reward progressif)
  - max_steps augmenté à 500
  - Espace de travail mieux borné
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import ikpy.chain
from pathlib import Path

URDF_PATH = str(Path(__file__).resolve().parent / 'ur7e_generated.urdf')
ACTIVE_MASK = [False, False, True, True, True, True, True, True, False]

JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]

JOINT_LIMITS = [
    (-2*np.pi, 2*np.pi), (-2*np.pi, 2*np.pi), (-np.pi, np.pi),
    (-2*np.pi, 2*np.pi), (-2*np.pi, 2*np.pi), (-2*np.pi, 2*np.pi),
]

# Position HOME — départ UNIQUE
HOME_CONFIG = np.array([0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0],
                        dtype=np.float32)

# Espace de travail réaliste (sphère creuse)
R_MIN = 0.20
R_MAX = 0.85

_CHAIN = None

def get_chain():
    global _CHAIN
    if _CHAIN is None:
        _CHAIN = ikpy.chain.Chain.from_urdf_file(
            URDF_PATH, active_links_mask=ACTIVE_MASK
        )
    return _CHAIN

def angles_to_ikpy(q6):
    return [0, 0] + list(q6) + [0]

def fk(q6):
    T = get_chain().forward_kinematics(angles_to_ikpy(q6))
    return T[:3, 3].astype(np.float32)

def sample_reachable_target():
    """Échantillonne une cible dans l'espace de travail réaliste."""
    for _ in range(200):
        # Échantillonnage sphérique uniforme
        r     = np.random.uniform(R_MIN, R_MAX)
        theta = np.random.uniform(0, 2*np.pi)
        phi   = np.random.uniform(0, np.pi)
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi) + 0.2   # décalage base
        if 0.05 <= z <= 0.9:
            return np.array([x, y, z], dtype=np.float32)
    return np.array([0.4, 0.0, 0.4], dtype=np.float32)


class UR7eEnvV4(gym.Env):
    """
    Environnement v4 — départ HOME fixe + curriculum précision.
    """

    def __init__(self, max_steps=500, max_delta=0.05):
        super().__init__()
        self.max_steps = max_steps
        self.max_delta = max_delta
        get_chain()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

        self._angles     = HOME_CONFIG.copy()
        self._target     = np.array([0.4, 0.0, 0.4], dtype=np.float32)
        self._step_count = 0
        self._best_dist  = np.inf   # pour reward shaping

    def _get_obs(self):
        angles_norm = np.array([
            self._angles[i] / JOINT_LIMITS[i][1] for i in range(6)
        ], dtype=np.float32)
        pos_eff = fk(self._angles)
        erreur  = (self._target - pos_eff).astype(np.float32)
        return np.concatenate([angles_norm, pos_eff, self._target, erreur])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0

        # ── DÉPART TOUJOURS DEPUIS HOME (pas de bruit) ──
        self._angles = HOME_CONFIG.copy().astype(np.float32)

        # Cible dans l'espace réaliste
        self._target = sample_reachable_target()

        self._best_dist = np.linalg.norm(fk(self._angles) - self._target)

        return self._get_obs(), {}

    def step(self, action):
        self._step_count += 1
        delta     = np.clip(action, -1, 1) * self.max_delta
        new_a     = self._angles + delta
        hit_limit = False

        for i in range(6):
            lo, hi = JOINT_LIMITS[i]
            if new_a[i] < lo or new_a[i] > hi:
                hit_limit = True
            new_a[i] = np.clip(new_a[i], lo, hi)
        self._angles = new_a.astype(np.float32)

        pos_eff = fk(self._angles)
        dist    = float(np.linalg.norm(pos_eff - self._target))

        # ── Reward shaping : récompenser le rapprochement ──
        reward = -dist * 10
        # Bonus si on bat le meilleur (encourage progression)
        if dist < self._best_dist:
            reward += (self._best_dist - dist) * 50
            self._best_dist = dist
        reward -= 0.01 * float(np.sum(np.abs(delta)))

        # ── Curriculum précision : paliers de récompense ──
        terminated = False
        if dist < 0.010:           # 10 mm
            reward += 30.0
        if dist < 0.005:           # 5 mm
            reward += 80.0
        if dist < 0.002:           # 2 mm — objectif
            reward += 300.0
            terminated = True
        if hit_limit:
            reward    -= 50.0
            terminated = True

        truncated = self._step_count >= self.max_steps
        info      = {"distance_mm": dist * 1000, "step": self._step_count}

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
