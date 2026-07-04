"""Phase 1 — transition repeatability gate.

From the same guided reset, repeat the same action and measure spot displacement,
physical pulse duration, joint-state message count and camera freshness.  This
turns the former non-Markov timing hypothesis into a quantitative GO/NO-GO test.
"""
from __future__ import annotations

import argparse
import numpy as np
import rclpy

from .env import UR7eLineFollowerEnv
from .reward import DEFAULT_REWARD_PROFILE, REWARD_PROFILES


def _coefficient_of_variation(values: np.ndarray) -> float:
    if values.size == 0:
        return float("inf")
    mean = float(np.mean(np.abs(values)))
    return float(np.std(values) / mean) if mean > 1e-9 else float("inf")


def run(
    action=(0.0, 1.0),
    n: int = 20,
    deterministic_pulse: bool = True,
    reward_profile: str = DEFAULT_REWARD_PROFILE,
) -> dict:
    a = np.asarray(action, dtype=np.float32).reshape(2)
    env = UR7eLineFollowerEnv(
        training_profile="minimal_straight_line_debug",
        deterministic_pulse=deterministic_pulse,
        update_dot_visual=False,
        reward_profile=reward_profile,
    )
    displacements: list[np.ndarray] = []
    durations: list[float] = []
    messages: list[int] = []
    fresh_frames: list[float] = []
    try:
        env.node.ensure_control_ready(timeout=5.0)
        env.reset(seed=0)
        env.step(np.zeros(2, dtype=np.float32))
        for trial in range(int(n)):
            env.reset(seed=0)
            dot0 = env.node.get_laser_dot()
            step0 = int(env.node.step_count)
            _, _, _, _, info = env.step(a)
            dot1 = env.node.get_laser_dot()
            step1 = int(env.node.step_count)
            if dot0 is None or dot1 is None:
                print(f"[repeat] essai {trial:02d}: intersection laser indisponible")
                continue
            delta = np.asarray(dot1, dtype=float) - np.asarray(dot0, dtype=float)
            displacements.append(delta)
            durations.append(float(info.get("pulse_duration_s", np.nan)))
            messages.append(step1 - step0)
            fresh_frames.append(float(info.get("fresh_camera_frame", True)))
            print(
                f"[repeat] {trial:02d} dy={delta[0]*1000:+7.2f}mm "
                f"dz={delta[1]*1000:+7.2f}mm "
                f"dt={durations[-1]*1000:6.1f}ms msgs={messages[-1]}"
            )

        disp = np.asarray(displacements, dtype=float)
        norms = np.linalg.norm(disp, axis=1) if disp.size else np.array([])
        duration_arr = np.asarray(durations, dtype=float)
        msg_arr = np.asarray(messages, dtype=float)
        result = {
            "n": int(norms.size),
            "mean_displacement_mm": float(np.mean(norms) * 1000) if norms.size else 0.0,
            "std_displacement_mm": float(np.std(norms) * 1000) if norms.size else float("inf"),
            "displacement_cv": _coefficient_of_variation(norms),
            "duration_mean_ms": float(np.nanmean(duration_arr) * 1000) if duration_arr.size else float("nan"),
            "duration_cv": _coefficient_of_variation(duration_arr[np.isfinite(duration_arr)]),
            "messages_mean": float(np.mean(msg_arr)) if msg_arr.size else float("nan"),
            "messages_std": float(np.std(msg_arr)) if msg_arr.size else float("nan"),
            "fresh_frame_rate": float(np.mean(fresh_frames)) if fresh_frames else float("nan"),
        }
        result["gate_pass"] = bool(
            result["n"] >= max(5, int(n) // 2)
            and result["displacement_cv"] <= 0.20
            and (not np.isfinite(result["duration_cv"]) or result["duration_cv"] <= 0.20)
            and result["messages_std"] <= 1.5
        )

        print("\n========== REPETABILITE ==========")
        print(f"essais valides       : {result['n']}")
        print(f"déplacement moyen    : {result['mean_displacement_mm']:.2f} mm")
        print(f"CV déplacement       : {100*result['displacement_cv']:.1f} %")
        print(f"durée moyenne         : {result['duration_mean_ms']:.1f} ms")
        print(f"CV durée              : {100*result['duration_cv']:.1f} %")
        print(f"messages/action       : {result['messages_mean']:.1f} ± {result['messages_std']:.2f}")
        print(f"GATE répétabilité     : {'PASS' if result['gate_pass'] else 'FAIL'}")
        if not result["gate_pass"]:
            print("Ne pas entraîner : vérifier stop()+settle, charge système ou lockstep Gazebo.")
        return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type=float, nargs=2, default=[0.0, 1.0])
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--no-deterministic", action="store_true")
    parser.add_argument("--reward-profile", choices=REWARD_PROFILES,
                        default=DEFAULT_REWARD_PROFILE)
    args = parser.parse_args()
    if not rclpy.ok():
        rclpy.init()
    try:
        run(
            tuple(args.action),
            args.n,
            deterministic_pulse=not args.no_deterministic,
            reward_profile=args.reward_profile,
        )
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
