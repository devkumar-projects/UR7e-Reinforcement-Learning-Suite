#!/usr/bin/env python3
"""Preflight for UR7e V3.3 offline training.

This script deliberately uses no ROS, rclpy, Gazebo, ros2_control, RViz or display.
Run it from the package root with PYTHONPATH=$PWD.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(f"[PRECHECK FAIL] {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="~/ur7e_models/offline_sac_level0.zip",
        help="Optional SB3 checkpoint to validate",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    root = Path.cwd()
    required = [
        root / "ur7e_line_follower" / "__init__.py",
        root / "offline_train_ur7e_curriculum_monitored.py",
        root / "offline_watch_ur7e.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        fail("Run from the package root. Missing: " + ", ".join(missing))

    print("[PRECHECK] Python:", sys.version.split()[0])
    print("[PRECHECK] Package root:", root)

    modules = [
        "numpy",
        "torch",
        "gymnasium",
        "stable_baselines3",
        "matplotlib",
        "ur7e_line_follower.control",
        "ur7e_line_follower.ekf",
        "ur7e_line_follower.kinematics",
        "ur7e_line_follower.reward",
        "ur7e_line_follower.singularity",
        "ur7e_line_follower.target_line",
        "offline_train_ur7e_curriculum_monitored",
    ]
    loaded = {}
    for name in modules:
        try:
            loaded[name] = importlib.import_module(name)
            version = getattr(loaded[name], "__version__", "n/a")
            print(f"[PRECHECK] import {name}: OK ({version})")
        except Exception as exc:
            fail(f"cannot import {name}: {exc}")

    torch = loaded["torch"]
    cuda_ok = bool(torch.cuda.is_available())
    print("[PRECHECK] CUDA available:", cuda_ok)
    if cuda_ok:
        print("[PRECHECK] GPU:", torch.cuda.get_device_name(0))
    if args.device.startswith("cuda") and not cuda_ok:
        fail("CUDA requested but unavailable inside Docker. Start the container with GPU access.")

    trainer = loaded["offline_train_ur7e_curriculum_monitored"]
    env = trainer.OfflineUR7eEnv(
        reward_profile="normalized_huber",
        random_lines=False,
        seed=2,
        line_level=1,
    )
    obs, info = env.reset(seed=2)
    if tuple(obs.shape) != (33,):
        fail(f"unexpected observation shape: {obs.shape}, expected (33,)")
    if not bool((obs == obs).all()):
        fail("observation contains NaN")
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, step_info = env.step(action)
    if tuple(next_obs.shape) != (33,):
        fail("environment step returned invalid observation shape")
    print("[PRECHECK] Offline environment: OK, obs=(33,), action=(2,)")

    checkpoint = Path(os.path.expanduser(args.checkpoint)).resolve()
    if checkpoint.exists():
        SAC = loaded["stable_baselines3"].SAC
        try:
            model = SAC.load(str(checkpoint), env=env, device=args.device)
        except Exception as exc:
            fail(f"checkpoint cannot be loaded: {exc}")
        print("[PRECHECK] Checkpoint: OK")
        print("[PRECHECK] SB3 device:", model.device)
    else:
        print(f"[PRECHECK] Checkpoint not found yet: {checkpoint}")
        print("[PRECHECK] Copy it before training; code/dependencies are otherwise valid.")

    forbidden = ["rclpy", "cv_bridge", "sensor_msgs", "controller_manager"]
    imported_forbidden = [name for name in sys.modules if name.split(".")[0] in forbidden]
    if imported_forbidden:
        fail("unexpected ROS modules imported: " + ", ".join(sorted(imported_forbidden)))

    print("[PRECHECK PASS] No ROS/Gazebo runtime is required for this offline training.")


if __name__ == "__main__":
    main()
