"""
Environnement RL UR7e — Suivi de trajectoire 2D (avec phase d'approche)
─────────────────────────────────────────────────────────────────────────
L'agent part de HOME, REJOINT le début de la ligne, PUIS la suit.

Deux phases :
  1. APPROCHE : rejoindre le 1er point (tolérante, pas de pénalité éloignement)
  2. SUIVI    : suivre la ligne avec contrainte stricte d'erreur latérale

Observation (taille FIXE = 6 + 3 + 1 + 1 + 3 + K*3) :
  - angles q1..q6 normalisés        (6)
  - position effecteur (x,y,z)      (3)
  - progression (% parcouru)        (1)
  - phase (0=approche, 1=suivi)     (1)
  - point cible courant             (3)
  - K prochains points (lookahead)  (K*3)

Action (6) : Δθ pour chaque joint
"""
import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

UR7E_WS = os.path.expanduser('~/ur7e_ws')
sys.path.insert(0, UR7E_WS)

from ur7e_env_v4 import JOINT_LIMITS, HOME_CONFIG, fk, get_chain
from trajectory_generator import generate_trajectory, trajectory_length

# ─────────────────────────────────────────────
K_LOOKAHEAD   = 5
N_TRAJ_POINTS = 60
SEUIL_POINT   = 0.025    # 25 mm pour valider un point cible (plus tolérant)
SEUIL_ERREUR  = 0.005    # 5 mm — erreur latérale "bonne"
ERREUR_MAX    = 0.06     # 6 cm — sortie de ligne (phase SUIVI uniquement)
SEUIL_ACCROCHE = 0.020   # 20 mm — distance pour "accrocher" la ligne

JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]


class UR7eTrajEnv(gym.Env):
    metadata = {"render_modes": [None]}

    def __init__(self, max_steps=800, max_delta=0.05,
                 k_lookahead=K_LOOKAHEAD, fixed_trajectory=None):
        super().__init__()
        self.max_steps  = max_steps
        self.max_delta  = max_delta
        self.k          = k_lookahead
        self.fixed_traj = fixed_trajectory
        get_chain()

        # Obs : 6 + 3 + 1 + 1 + 3 + K*3
        obs_dim = 6 + 3 + 1 + 1 + 3 + self.k * 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

        self._angles     = HOME_CONFIG.copy().astype(np.float32)
        self._trajectory = None
        self._cursor     = 0
        self._step_count = 0
        self._prev_delta = np.zeros(6, dtype=np.float32)
        self._errors     = []
        self._phase      = 0      # 0 = approche, 1 = suivi
        self._stagnation = 0
        self._max_cursor = 0

    # ─────────────────────────────────────────
    def _closest_point_on_traj(self, pos):
        # Recherche fenêtrée autour du curseur (rapide + évite retour arrière)
        lo = max(0, self._cursor - 3)
        hi = min(len(self._trajectory), self._cursor + 12)
        window = self._trajectory[lo:hi]
        dists  = np.linalg.norm(window - pos, axis=1)
        idx    = int(np.argmin(dists))
        return lo + idx, float(dists[idx])

    def _get_lookahead(self):
        pts = []
        for i in range(self.k):
            idx = min(self._cursor + i, len(self._trajectory) - 1)
            pts.append(self._trajectory[idx])
        return np.concatenate(pts).astype(np.float32)

    def _get_obs(self):
        angles_norm = np.array([
            self._angles[i] / JOINT_LIMITS[i][1] for i in range(6)
        ], dtype=np.float32)
        pos_eff     = fk(self._angles)
        progression = np.array([self._cursor / len(self._trajectory)],
                                dtype=np.float32)
        phase       = np.array([float(self._phase)], dtype=np.float32)
        target      = self._trajectory[self._cursor]
        lookahead   = self._get_lookahead()

        return np.concatenate([
            angles_norm, pos_eff, progression, phase, target, lookahead
        ]).astype(np.float32)

    # ─────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._cursor     = 0
        self._errors     = []
        self._prev_delta = np.zeros(6, dtype=np.float32)
        self._phase      = 0     # commence en APPROCHE
        self._stagnation = 0     # compteur de steps sans progression
        self._max_cursor = 0     # plus loin atteint

        self._angles = HOME_CONFIG.copy().astype(np.float32)

        if self.fixed_traj is not None:
            self._trajectory = self.fixed_traj.astype(np.float32)
        else:
            self._trajectory, _ = generate_trajectory(n_points=N_TRAJ_POINTS)

        return self._get_obs(), {}

    # ─────────────────────────────────────────
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

        pos_eff    = fk(self._angles)
        target     = self._trajectory[self._cursor]
        dist_cible = float(np.linalg.norm(pos_eff - target))
        _, err_lat = self._closest_point_on_traj(pos_eff)

        terminated = False
        truncated  = False

        # ══════════════════════════════════════
        # PHASE 0 — APPROCHE : rejoindre le 1er point
        # ══════════════════════════════════════
        if self._phase == 0:
            dist_debut = float(np.linalg.norm(pos_eff - self._trajectory[0]))

            # Récompense : se rapprocher du début de la ligne
            reward = -dist_debut * 30
            reward -= 0.01 * float(np.sum(np.abs(delta)))
            if hit_limit:
                reward -= 30.0

            # Transition vers la phase SUIVI quand on accroche la ligne
            if dist_debut < SEUIL_ACCROCHE:
                self._phase = 1
                reward += 50.0   # bonus d'accrochage
                self._errors = []  # reset erreurs (on commence le suivi propre)

        # ══════════════════════════════════════
        # PHASE 1 — SUIVI : suivre la ligne précisément
        # ══════════════════════════════════════
        else:
            self._errors.append(err_lat)

            # Avancement curseur
            progressed = False
            if dist_cible < SEUIL_POINT and self._cursor < len(self._trajectory)-1:
                self._cursor += 1
                progressed = True

            # 1. Précision (erreur latérale)
            reward = -err_lat * 150
            # 2. Bonus précision fine
            if err_lat < SEUIL_ERREUR:
                reward += 5.0
            # 3. Progression — FORTEMENT récompensée (anti-immobilité)
            if progressed:
                reward += 40.0
                self._stagnation = 0
            else:
                self._stagnation += 1
            # 4. Bonus de nouveau record de progression
            if self._cursor > self._max_cursor:
                reward += 20.0
                self._max_cursor = self._cursor
            # 5. Guidage vers le point cible (pousse à avancer)
            reward -= dist_cible * 30
            # 6. Pénalité de STAGNATION (anti-blocage)
            if self._stagnation > 30:
                reward -= 5.0
            # 7. Limites
            if hit_limit:
                reward -= 30.0
            # 8. Fluidité (à-coups)
            jerk = np.sum(np.abs(delta - self._prev_delta))
            reward -= 0.5 * float(jerk)

            # Succès : ligne complétée
            if self._cursor >= len(self._trajectory)-1 and dist_cible < SEUIL_POINT:
                rms = np.sqrt(np.mean(np.square(self._errors)))
                reward += 200.0 + max(0, (SEUIL_ERREUR - rms) * 10000)
                terminated = True

            # Échec : sortie de ligne (uniquement en phase SUIVI)
            if err_lat > ERREUR_MAX:
                reward -= 50.0
                terminated = True
            # Échec : bloqué trop longtemps sans progresser
            if self._stagnation > 150:
                reward -= 30.0
                terminated = True

        self._prev_delta = delta.copy()

        if self._step_count >= self.max_steps:
            truncated = True

        rms_courant = (np.sqrt(np.mean(np.square(self._errors)))
                       if self._errors else 0)
        info = {
            "phase"           : self._phase,
            "err_laterale_mm" : err_lat * 1000,
            "err_rms_mm"      : rms_courant * 1000,
            "progression"     : self._cursor / len(self._trajectory),
            "completed"       : terminated and err_lat < ERREUR_MAX and self._phase == 1,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass


# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("Test environnement suivi (avec phase approche)\n")
    env = UR7eTrajEnv()
    print(f"Observation : {env.observation_space.shape}  (28 → 29 avec phase)")
    print(f"Action      : {env.action_space.shape}")

    obs, _ = env.reset()
    print(f"\nObs shape    : {obs.shape}")
    print(f"Trajectoire  : {len(env._trajectory)} points")
    print(f"Phase init   : {env._phase} (0=approche)")

    print("\n20 steps aléatoires :")
    for i in range(20):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        ph = "APPROCHE" if info['phase'] == 0 else "SUIVI"
        print(f"  step {i+1:2d} | {ph:8s} | reward {reward:7.1f} | "
              f"err_lat {info['err_laterale_mm']:6.1f} mm | "
              f"prog {info['progression']*100:4.0f}%")
        if term or trunc:
            print("  → épisode terminé")
            break
    print("\nTest terminé !")
