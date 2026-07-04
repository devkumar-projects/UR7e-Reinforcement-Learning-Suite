"""
Visualisation 3D du suivi de trajectoire — UR7e SAC
─────────────────────────────────────────────────────
Fait suivre plusieurs trajectoires à l'agent et trace en 3D :
  - la trajectoire CIBLE (vert)
  - le chemin RÉELLEMENT suivi par l'effecteur (bleu)
  - les points de départ HOME et d'accroche

Produit une figure claire pour analyse / rapport.
Pas besoin de ROS — tout en interne via ikpy.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3 import SAC

UR7E_WS  = os.path.expanduser('~/ur7e_ws')
TRAJ_DIR = os.path.join(UR7E_WS, 'trajectoire')
sys.path.insert(0, UR7E_WS)
sys.path.insert(0, TRAJ_DIR)

from ur7e_env_v4 import JOINT_LIMITS, HOME_CONFIG, fk, get_chain
from trajectory_generator import generate_trajectory, trajectory_length
from ur7e_traj_env import (
    K_LOOKAHEAD, SEUIL_POINT, SEUIL_ACCROCHE, ERREUR_MAX, N_TRAJ_POINTS
)

# ═════════════════════════════════════════════
MODEL_PATH = os.path.join(TRAJ_DIR, 'models_traj', 'best_model')
N_TRAJECTOIRES = 4        # nombre de trajectoires à tester/afficher
MAX_STEPS      = 800
SEED_BASE      = 100      # graines pour reproductibilité

get_chain()
print(f"Chargement : {MODEL_PATH}")
model = SAC.load(MODEL_PATH)
print("Modèle chargé ✓\n")


# ─────────────────────────────────────────────
def closest_dist(trajectory, cursor, pos):
    lo = max(0, cursor - 3)
    hi = min(len(trajectory), cursor + 12)
    window = trajectory[lo:hi]
    dists  = np.linalg.norm(window - pos, axis=1)
    return float(dists.min())

def get_lookahead(trajectory, cursor, k):
    pts = []
    for i in range(k):
        idx = min(cursor + i, len(trajectory) - 1)
        pts.append(trajectory[idx])
    return np.concatenate(pts).astype(np.float32)

def run_trajectory(trajectory):
    """Fait suivre une trajectoire à l'agent. Retourne (trace, rms, completed)."""
    angles = HOME_CONFIG.copy().astype(np.float32)
    cursor = 0
    phase  = 0
    trace  = []
    errors = []

    for _ in range(MAX_STEPS):
        pos_eff = fk(angles)
        trace.append(pos_eff.copy())

        # Construire l'observation (identique à l'env)
        angles_norm = np.array([angles[i]/JOINT_LIMITS[i][1]
                                 for i in range(6)], dtype=np.float32)
        progression = np.array([cursor/len(trajectory)], dtype=np.float32)
        phase_arr   = np.array([float(phase)], dtype=np.float32)
        target      = trajectory[cursor]
        lookahead   = get_lookahead(trajectory, cursor, K_LOOKAHEAD)
        obs = np.concatenate([angles_norm, pos_eff, progression,
                              phase_arr, target, lookahead]).astype(np.float32)

        action, _ = model.predict(obs, deterministic=True)
        delta = np.clip(action, -1, 1) * 0.05
        new_a = angles + delta
        for i in range(6):
            lo, hi = JOINT_LIMITS[i]
            new_a[i] = np.clip(new_a[i], lo, hi)
        angles = new_a.astype(np.float32)

        pos_eff    = fk(angles)
        dist_cible = float(np.linalg.norm(pos_eff - target))
        err_lat    = closest_dist(trajectory, cursor, pos_eff)

        if phase == 0:
            dist_debut = float(np.linalg.norm(pos_eff - trajectory[0]))
            if dist_debut < SEUIL_ACCROCHE:
                phase = 1
                errors = []
        else:
            errors.append(err_lat)
            if dist_cible < SEUIL_POINT and cursor < len(trajectory)-1:
                cursor += 1
            if cursor >= len(trajectory)-1 and dist_cible < SEUIL_POINT:
                rms = np.sqrt(np.mean(np.square(errors)))*1000
                return np.array(trace), rms, True
            if err_lat > ERREUR_MAX:
                rms = np.sqrt(np.mean(np.square(errors)))*1000 if errors else 999
                return np.array(trace), rms, False

    rms = np.sqrt(np.mean(np.square(errors)))*1000 if errors else 999
    return np.array(trace), rms, False


# ─────────────────────────────────────────────
# Tester N trajectoires
# ─────────────────────────────────────────────
print(f"Test de {N_TRAJECTOIRES} trajectoires...\n")

fig = plt.figure(figsize=(16, 12))
home_pos = fk(HOME_CONFIG)

for idx in range(N_TRAJECTOIRES):
    trajectory, name = generate_trajectory(n_points=N_TRAJ_POINTS,
                                            seed=SEED_BASE + idx)
    trace, rms, completed = run_trajectory(trajectory)

    status = "✓ complétée" if completed else "✗ interrompue"
    print(f"  Traj {idx+1} ({name:8s}) : RMS {rms:5.1f} mm | {status}")

    ax = fig.add_subplot(2, 2, idx+1, projection='3d')

    # Trajectoire cible (vert)
    ax.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2],
            'g-', linewidth=3, alpha=0.7, label='Cible')
    ax.scatter(trajectory[:,0], trajectory[:,1], trajectory[:,2],
               c='green', s=15, alpha=0.4)

    # Trace réelle de l'effecteur (bleu)
    ax.plot(trace[:,0], trace[:,1], trace[:,2],
            'b-', linewidth=1.5, alpha=0.8, label='Suivi robot')

    # Départ HOME (étoile)
    ax.scatter([home_pos[0]], [home_pos[1]], [home_pos[2]],
               c='orange', s=150, marker='*', zorder=5, label='HOME')

    # Début de la ligne
    ax.scatter([trajectory[0,0]], [trajectory[0,1]], [trajectory[0,2]],
               c='red', s=80, marker='o', zorder=5, label='Début ligne')

    ax.set_xlabel('X (m)', fontsize=9)
    ax.set_ylabel('Y (m)', fontsize=9)
    ax.set_zlabel('Z (m)', fontsize=9)
    ax.set_title(f'{name} — RMS {rms:.1f} mm — {status}',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.view_init(elev=20, azim=-60)

plt.suptitle('Suivi de trajectoire UR7e — SAC',
             fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()

png = os.path.join(TRAJ_DIR, 'visu_suivi_3d.png')
plt.savefig(png, dpi=150, bbox_inches='tight')
print(f"\nFigure sauvegardée : {png}")
plt.show()
print("Terminé !")
