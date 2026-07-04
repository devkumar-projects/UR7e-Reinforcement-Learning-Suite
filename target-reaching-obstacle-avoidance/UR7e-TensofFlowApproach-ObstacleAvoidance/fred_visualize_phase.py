# ==============================================================================
# FICHIER : fred_visualize_phase.py
# RÔLE : Visualiser une politique entraînée (phase 1/2/3) dans PyBullet, avec
#        LISSAGE B-SPLINE fidèle à Fred + détection de collision de la spline
#        (Option A) pour les phases avec obstacle.
#
# Usage : python fred_visualize_phase.py 1   (ou 2, ou 3)
# ==============================================================================

import os, sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import time
import numpy as np
import tensorflow as tf
import pybullet as p
from scipy.interpolate import splprep, splev
from tf_agents.environments import tf_py_environment

from fred_phase_base import M_TO_CM

PHASE = sys.argv[1] if len(sys.argv) > 1 else "1"
POLICY_DIR = f"fred_policy_phase{PHASE}_final"
N_TARGETS = 20
SEED = 7
SMOOTH_SAMPLES = 80
SLOWDOWN = 1.0 / 60.0
PAUSE_BETWEEN = 0.8


def make_env(phase, seed):
    if phase == "1":
        from fred_phase1_env import FredPhase1Env
        return FredPhase1Env(use_gui=True, seed=seed)
    elif phase == "2":
        from fred_phase2_env import FredPhase2Env
        return FredPhase2Env(use_gui=True, seed=seed)
    else:
        from fred_phase3_env import FredPhase3Env
        return FredPhase3Env(use_gui=True, seed=seed)


def fit_bspline(pts_cm):
    pts = np.asarray(pts_cm)
    n = len(pts)
    if n < 3:
        return pts
    k = min(n - 1, 5)
    s = n + np.sqrt(2 * n)
    try:
        tck, _ = splprep([pts[:, 0], pts[:, 1], pts[:, 2]], k=k, s=s)
        fine = np.linspace(0, 1, SMOOTH_SAMPLES)
        xs, ys, zs = splev(fine, tck)
        return np.stack([xs, ys, zs], axis=1)
    except Exception:
        return pts


def main():
    if not os.path.isdir(POLICY_DIR):
        print(f"ERREUR : '{POLICY_DIR}' introuvable. Lance l'entraînement de "
              f"la phase {PHASE} d'abord (ou ajuste le suffixe _final/_interrupted).")
        return

    print(f"=== Visualisation PHASE {PHASE} ===")
    policy = tf.saved_model.load(POLICY_DIR)
    py_env = make_env(PHASE, SEED)
    env = tf_py_environment.TFPyEnvironment(py_env)
    has_obstacle = PHASE in ("2", "3")

    successes = 0
    spline_collisions = 0
    try:
        for i in range(N_TARGETS):
            t = env.reset()
            keys = [py_env._get_ee_pos_cm().copy()]
            ep_return = 0.0
            while not t.is_last():
                a = policy.action(t)
                t = env.step(a.action)
                ep_return += float(t.reward.numpy()[0])
                keys.append(py_env._get_ee_pos_cm().copy())
            reached = ep_return >= 50.0
            successes += int(reached)

            smooth = fit_bspline(np.array(keys))
            py_env._set_config(py_env._ik(smooth[0] / M_TO_CM))
            n_coll = 0
            for pt in smooth:
                py_env._set_config(py_env._ik(pt / M_TO_CM))
                p.stepSimulation(physicsClientId=py_env._physics_client)
                if has_obstacle and py_env._in_collision():
                    n_coll += 1
                time.sleep(SLOWDOWN)
            if has_obstacle and n_coll > 0:
                spline_collisions += 1

            status = "OK" if reached else "X"
            extra = (f"  [spline coll: {n_coll} pts]" if (has_obstacle and n_coll)
                     else "")
            print(f"  Cible {i+1:2d}/{N_TARGETS}: {status}  retour {ep_return:6.1f}"
                  f"  {len(keys)} clés->{len(smooth)} pts{extra}")
            time.sleep(PAUSE_BETWEEN)
    except KeyboardInterrupt:
        print("\nInterrompu.")
    finally:
        print(f"\n--- Réussites: {successes}/{N_TARGETS} "
              f"({100*successes/max(1,N_TARGETS):.0f}%) ---")
        if has_obstacle:
            print(f"--- Spline en collision: {spline_collisions}/{N_TARGETS} "
                  f"épisodes ---")
        py_env.close()


if __name__ == "__main__":
    main()
