#!/usr/bin/env python3
"""
Demo visuelle Gazebo — UR7e Line Follower.
4 épisodes A/B/C/D en séquence. Vitesse augmentée. Transition rapide.

Usage :
    ros2 run ur7e_line_follower demo_gazebo
    ros2 run ur7e_line_follower demo_gazebo --speed 2.0 --pause 1.0
"""
from __future__ import annotations
import argparse
import time
import sys

import numpy as np

MODEL_PATH = str(
    __import__("pathlib").Path.home()
    / ".ros/ur7e_line_follower/offline_runs/basile_level2/offline_sac_final.zip"
)

ALL_EPISODES = {
    "A": (-1, 42,  "Droite fixe"),
    "B": ( 1, 45,  "Courbe modérée"),
    "C": ( 2, 47,  "Ligne aléatoire — succès"),
    "D": ( 2, 50,  "Ligne aléatoire — difficile"),
}


def banner(text: str):
    print(f"\n{'═'*60}")
    print(f"  {text}")
    print(f"{'═'*60}")


def run_episode(model, env, level: int, seed: int, label: str,
                max_steps: int = 150, speed_factor: float = 1.0):
    """Exécute un épisode. Vitesse = speed_factor × MAX_WALL_SPEED nominal."""
    from ur7e_line_follower.target_line import (
        straight_line_from_start, curriculum_line_from_start,
        arc_length, waypoint_abscissae,
    )
    from ur7e_line_follower.kinematics import laser_wall_intersection_unbounded

    # Forcer niveau et seed AVANT reset
    env._curriculum_level_value = max(0, level)
    env._rng = np.random.default_rng(seed)

    obs, _ = env.reset()

    # Override trajectoire droite fixe APRÈS reset (sinon écrasé)
    if level == -1:
        HOME_Q = np.array([-0.133, -1.5708, 1.5708, 0.0, 1.5708, 0.0])
        start = laser_wall_intersection_unbounded(HOME_Q)
        if start is not None:
            env.waypoints   = straight_line_from_start(start, length=0.30)
            env._n_wp       = len(env.waypoints)
            env._path_length = arc_length(env.waypoints)
            env._waypoint_s  = waypoint_abscissae(env.waypoints)
            # Redessiner la trajectoire dans Gazebo
            env.node.update_trajectory_visual(env.waypoints)

    # Mettre à jour le dot laser immédiatement
    env.node.update_laser_dot_visual()

    banner(f"Épisode {label} — {ALL_EPISODES[label][2]}")
    print(f"  Trajectoire : {env._n_wp} pts | longueur = {env._path_length:.2f} m "
          f"| vitesse ×{speed_factor:.1f}")
    print(f"\n  {'Step':>5}  {'Prog':>6}  {'RMSE(cm)':>8}  {'Reward':>8}  {'Statut'}")
    print(f"  {'-'*55}")

    done, step, total_reward = False, 0, 0.0

    while not done and step < max_steps:
        # ── Action SAC avec vitesse augmentée ──
        raw_action, _ = model.predict(obs, deterministic=True)
        action = np.clip(raw_action * speed_factor, -1.0, 1.0).astype(np.float32)

        obs, reward, terminated, truncated, info = env.step(action)

        # Mise à jour laser dot Gazebo
        env.node.update_laser_dot_visual()

        done = terminated or truncated
        total_reward += reward
        step += 1

        prog    = info.get("progress", 0.0)
        rmse_cm = info.get("recent_rmse", 0.0) * 100
        success = info.get("is_success", False)

        if step % 5 == 0 or done:
            if success:
                statut = "✓ SUCCÈS"
            elif terminated and not success:
                statut = "✗ ÉCHEC"
            else:
                statut = f"▶ {prog:.0%}"
            print(f"  {step:>5}  {prog:>5.1%}  {rmse_cm:>7.1f}cm  {reward:>8.3f}  {statut}")

    prog_final = info.get("progress", 0.0)
    success    = info.get("is_success", False)
    rmse_cm    = info.get("recent_rmse", 0.0) * 100

    print(f"\n  ► {'SUCCÈS ✓' if success else 'ÉCHEC ✗'}  |  "
          f"Progression : {prog_final:.1%}  |  RMSE : {rmse_cm:.1f} cm  |  "
          f"Steps : {step}")

    return success, prog_final, rmse_cm


def main():
    parser = argparse.ArgumentParser(description="Demo Gazebo UR7e — 4 épisodes SAC")
    parser.add_argument("--episodes",  nargs="+", default=["A", "B", "C", "D"],
                        choices=["A", "B", "C", "D"])
    parser.add_argument("--model",     default=MODEL_PATH)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--speed",     type=float, default=2.0,
                        help="Facteur de vitesse sur les actions SAC (défaut 2.0 = 2× plus rapide)")
    parser.add_argument("--pause",     type=float, default=1.5,
                        help="Pause entre épisodes en secondes (défaut 1.5s)")
    args = parser.parse_args()

    banner("UR7e Demo Gazebo — 4 épisodes")
    print(f"  Modèle       : {args.model}")
    print(f"  Épisodes     : {' → '.join(args.episodes)}")
    print(f"  Vitesse ×    : {args.speed}")
    print(f"  Pause        : {args.pause}s")
    print(f"\n⚠  Gazebo doit être lancé :")
    print("   ros2 launch ur7e_line_follower simulation.launch.py\n")
    time.sleep(2.0)

    from ur7e_line_follower.env import UR7eLineFollowerEnv
    from ur7e_line_follower.train import _load_model_checked
    import ur7e_line_follower.control as _ctrl

    # ── Monkeypatch vitesse ──────────────────────────────────────────────────
    # On augmente la vitesse max du mur → robot plus rapide
    _original_speed = _ctrl.MAX_WALL_SPEED_M_S
    _ctrl.MAX_WALL_SPEED_M_S = min(_original_speed * args.speed, 0.35)
    print(f"[1/3] Vitesse mur : {_original_speed:.3f} → {_ctrl.MAX_WALL_SPEED_M_S:.3f} m/s")

    # ── Env persistante ──────────────────────────────────────────────────────
    print("[2/3] Connexion ROS2 / Gazebo...")
    env = UR7eLineFollowerEnv(
        random_trajectories=True,
        observation_mode="privileged_debug",
        curriculum=True,
        sensor_noise=False,
        update_dot_visual=True,
        trials_per_trajectory=1,
        max_steps=args.max_steps,
        reward_profile="normalized_huber",
        guided_reset=True,
    )
    print("  ✓ Env prête")

    # ── Thread laser dot ─────────────────────────────────────────────────────
    env.node.start_laser_dot_thread(rate_hz=15.0)
    print("  ✓ Laser dot thread 15 Hz")

    # ── Modèle ───────────────────────────────────────────────────────────────
    print("[3/3] Chargement modèle SAC...")
    model = _load_model_checked(args.model, env)
    print("  ✓ Modèle chargé\n")

    # ── Épisodes ─────────────────────────────────────────────────────────────
    results = {}
    for i, ep_key in enumerate(args.episodes):
        level, seed, _ = ALL_EPISODES[ep_key]
        success, prog, rmse = run_episode(
            model, env,
            level=level, seed=seed, label=ep_key,
            max_steps=args.max_steps,
            speed_factor=args.speed,
        )
        results[ep_key] = (success, prog, rmse)

        if i < len(args.episodes) - 1:
            print(f"\n  ── Prochain épisode dans {args.pause:.1f}s... ──")
            time.sleep(args.pause)

    # ── Résumé ───────────────────────────────────────────────────────────────
    banner("Résumé")
    print(f"  {'Ep':<4} {'Description':<32} {'Résultat':<10} {'Prog':>6}  {'RMSE':>8}")
    print(f"  {'-'*68}")
    for ep_key in args.episodes:
        _, _, desc = ALL_EPISODES[ep_key]
        success, prog, rmse = results[ep_key]
        print(f"  {ep_key:<4} {desc:<32} {'✓ SUCCÈS' if success else '✗ ÉCHEC':<10} "
              f"{prog:>5.1%}  {rmse:>6.1f} cm")
    print()

    env.node.stop_laser_dot_thread()
    _ctrl.MAX_WALL_SPEED_M_S = _original_speed  # restore
    env.close()


if __name__ == "__main__":
    main()
