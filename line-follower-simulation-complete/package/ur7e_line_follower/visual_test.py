"""Test visuel humain — robot bouge lentement, logs clairs, pause entre steps.

Modes :
  Sans --model  : contrôleur P géométrique (expert)
  Avec --model  : politique SAC chargée depuis le checkpoint

Options clés :
  --episodes N   : nombre d'épisodes (chaque épisode = nouveau dessin)
  --steps N      : steps max par épisode
  --pause S      : pause en secondes entre chaque step (défaut 0.6)
  --model PATH   : checkpoint SAC (.zip) à évaluer visuellement
"""
from __future__ import annotations
import argparse
import math
import time
import numpy as np
import rclpy

from .env import UR7eLineFollowerEnv
from .control import MAX_WALL_SPEED_M_S
from .target_line import closest_point_on_polyline
from .kinematics import laser_wall_dot

BANNER = "=" * 62


def _bar(v: float, width: int = 20) -> str:
    v = float(np.clip(v, 0.0, 1.0))
    filled = int(round(v * width))
    return "[" + "█" * filled + "░" * (width - filled) + f"] {v*100:5.1f}%"


def _expert_action(env, kp: float) -> np.ndarray:
    dot = laser_wall_dot(env.node.joint_pos)
    target_idx = min(env._wp_idx, env._n_wp - 1)
    target = np.asarray(env.waypoints[target_idx], dtype=float)
    if dot is None:
        return np.zeros(2, dtype=np.float32)
    err = target - np.asarray(dot, dtype=float)
    desired = kp * err
    speed = float(np.linalg.norm(desired))
    if speed > MAX_WALL_SPEED_M_S:
        desired *= MAX_WALL_SPEED_M_S / speed
    return np.clip(desired / MAX_WALL_SPEED_M_S, -1.0, 1.0).astype(np.float32)


def run(episodes: int = 3, steps: int = 60, pause: float = 0.6,
        kp: float = 2.2, model_path: str | None = None) -> None:

    mode_label = f"SAC [{model_path}]" if model_path else f"Expert P (kp={kp})"
    print(BANNER)
    print("  TEST VISUEL UR7e — vérification humaine en Gazebo")
    print(f"  Mode     : {mode_label}")
    print(f"  Épisodes : {episodes}  |  Steps/épisode : {steps}  |  Pause : {pause}s")
    print(BANNER)

    # Charge le modèle SAC si demandé
    sac_model = None
    if model_path:
        from stable_baselines3 import SAC
        from .train import _load_model_checked

    env = UR7eLineFollowerEnv(
        random_trajectories=True,
        sensor_noise=False,
        observation_mode='real',
        update_dot_visual=False,  # set_pose conflicts avec create service en Gazebo
        trials_per_trajectory=1,   # nouveau dessin à chaque épisode
        curriculum=False,
    )

    try:
        env.node.start_laser_dot_thread(rate_hz=3.0)  # dot rouge suit l'effecteur à 3 Hz

        if model_path:
            sac_model = _load_model_checked(model_path, env)
            print(f"  Checkpoint chargé OK\n")

        for ep in range(episodes):
            print(f"\n{'─'*62}")
            print(f"  ÉPISODE {ep+1}/{episodes}")
            print(f"{'─'*62}")

            obs, _ = env.reset()
            wp = env.waypoints
            print(f"  Dessin : {env._n_wp} pts | longueur={env._path_length:.3f} m")
            print(f"  Départ : y={wp[0,0]:.3f} m  z={wp[0,1]:.3f} m")
            print(f"  Arrivée: y={wp[-1,0]:.3f} m  z={wp[-1,1]:.3f} m")
            print(f"  Pause {pause:.1f}s — observe le dessin dans Gazebo...\n")
            time.sleep(pause * 2)

            for k in range(steps):
                if sac_model is not None:
                    action, _ = sac_model.predict(obs, deterministic=True)
                    action = np.asarray(action, dtype=np.float32).reshape(2)
                    movement = f"SAC vy={action[0]:+.3f}  vz={action[1]:+.3f}"
                else:
                    action = _expert_action(env, kp)
                    movement = f"Expert vy={action[0]:+.3f}  vz={action[1]:+.3f}"

                obs, reward, terminated, truncated, info = env.step(action)
                env.node.update_laser_dot_visual()  # met à jour la position pour le thread 3Hz

                cam      = np.asarray(obs[15:22], dtype=float)
                guidance = np.asarray(obs[12:15], dtype=float)
                dot_fk   = laser_wall_dot(env.node.joint_pos)
                dot_ekf  = env.node.ekf.position
                ekf_err_mm = (float(np.linalg.norm(dot_ekf - np.asarray(dot_fk))) * 1000
                              if dot_fk is not None else math.nan)
                dist_cm = math.nan
                if dot_fk is not None:
                    c = closest_point_on_polyline(dot_fk, wp,
                                                  start_idx=max(env._wp_idx - 1, 0),
                                                  window=5)
                    dist_cm = c['distance'] * 100.0

                line_ok  = cam[0] > 0.5
                laser_ok = cam[6] > 0.5
                klt_conf = cam[2]
                progress = float(info.get('progress', 0.0))
                cond     = float(info.get('cond_wall', math.nan))

                ligne_sym = "✓ LIGNE" if line_ok  else "✗ LIGNE"
                laser_sym = "✓ LASER" if laser_ok else "✗ LASER"

                print(f"[ep{ep+1} step {k+1:03d}/{steps}]  "
                      f"wp={env._wp_idx:02d}/{env._n_wp}  {ligne_sym}  {laser_sym}")
                print(f"  Progression   {_bar(progress)}")
                print(f"  KLT confiance {_bar(klt_conf)}")
                print(f"  Distance dessin : {dist_cm:6.2f} cm   EKF : {ekf_err_mm:.2f} mm")
                print(f"  Commande : {movement}   reward={reward:+.3f}")
                print()

                if terminated:
                    print(f"  >>> SUCCÈS — trajectoire complétée !\n")
                    break
                if truncated:
                    reason = info.get('stagnation_timeout', False)
                    print(f"  >>> Tronqué ({'stagnation' if reason else 'max steps'})\n")
                    break

                time.sleep(pause)

            if ep < episodes - 1:
                print(f"  Pause {pause*2:.1f}s avant le prochain dessin...\n")
                time.sleep(pause * 2)

        print(BANNER)
        print("  FIN DU TEST VISUEL")
        print(BANNER)

    finally:
        env.node.stop_laser_dot_thread()
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Test visuel humain UR7e")
    parser.add_argument('--episodes', type=int, default=3,
                        help='nombre d épisodes / dessins (défaut 3)')
    parser.add_argument('--steps', type=int, default=60,
                        help='steps max par épisode (défaut 60)')
    parser.add_argument('--pause', type=float, default=0.4,
                        help='pause en secondes entre chaque step (défaut 0.6)')
    parser.add_argument('--kp', type=float, default=2.2,
                        help='gain expert si pas de modèle SAC (défaut 2.2)')
    parser.add_argument('--model', type=str, default=None,
                        help='chemin vers un checkpoint SAC .zip')
    args = parser.parse_args()

    if not rclpy.ok():
        rclpy.init()
    try:
        run(episodes=args.episodes, steps=args.steps, pause=args.pause,
            kp=args.kp, model_path=args.model)
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
