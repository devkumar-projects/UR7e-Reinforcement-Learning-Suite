"""Phase 1 — four-action sign and axis gate."""
from __future__ import annotations

import argparse
import numpy as np
import rclpy

from .env import UR7eLineFollowerEnv
from .reward import DEFAULT_REWARD_PROFILE, REWARD_PROFILES

DIRECTIONS = {
    "+y": np.array([1.0, 0.0], dtype=np.float32),
    "-y": np.array([-1.0, 0.0], dtype=np.float32),
    "+z": np.array([0.0, 1.0], dtype=np.float32),
    "-z": np.array([0.0, -1.0], dtype=np.float32),
}
EXPECTED = {"+y": (0, +1), "-y": (0, -1), "+z": (1, +1), "-z": (1, -1)}


def run(steps: int = 6, reward_profile: str = DEFAULT_REWARD_PROFILE) -> dict:
    env = UR7eLineFollowerEnv(
        training_profile="minimal_straight_line_debug",
        deterministic_pulse=True,
        update_dot_visual=False,
        reward_profile=reward_profile,
    )
    table: dict[str, dict] = {}
    try:
        # Warm up ROS discovery/controller once.  Without this, the first +y
        # command may be published before the controller subscription is matched.
        env.node.ensure_control_ready(timeout=5.0)
        env.reset(seed=0)
        env.step(np.zeros(2, dtype=np.float32))
        env.reset(seed=0)
        for index, (name, action) in enumerate(DIRECTIONS.items()):
            env.reset(seed=0)
            dot0 = env.node.get_laser_dot()
            cam0 = np.asarray(getattr(env.node, "cam_detection", np.zeros(7)), dtype=float)
            for _ in range(int(steps)):
                _, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    print(f"[dirs] {name}: arrêt anticipé { {k:v for k,v in info.items() if k.endswith('_timeout') and v} }")
                    break
            dot1 = env.node.get_laser_dot()
            cam1 = np.asarray(getattr(env.node, "cam_detection", np.zeros(7)), dtype=float)
            if dot0 is None or dot1 is None:
                print(f"[dirs] {name}: intersection laser indisponible")
                continue
            delta = np.asarray(dot1, dtype=float) - np.asarray(dot0, dtype=float)
            table[name] = {
                "dy": float(delta[0]),
                "dz": float(delta[1]),
                "camera_offset_delta": float(cam1[1] - cam0[1]),
            }
            print(
                f"[dirs] {name:2s} -> dy={delta[0]*1000:+7.2f}mm "
                f"dz={delta[1]*1000:+7.2f}mm "
                f"d(offset_img)={table[name]['camera_offset_delta']:+.3f}"
            )

        gate = True
        print("\n========== COHERENCE DES AXES ==========")
        for name, (axis, sign) in EXPECTED.items():
            if name not in table:
                gate = False
                print(f"{name}: MANQUANT")
                continue
            principal = table[name]["dy" if axis == 0 else "dz"]
            cross = table[name]["dz" if axis == 0 else "dy"]
            sign_ok = principal * sign > 0.0
            dominance_ok = abs(principal) >= 0.5 * abs(cross)
            ok = sign_ok and dominance_ok
            gate = gate and ok
            print(
                f"{name}: principal={principal*1000:+.2f}mm "
                f"cross={cross*1000:+.2f}mm -> {'PASS' if ok else 'FAIL'}"
            )
        print(f"GATE directions : {'PASS' if gate else 'FAIL'}")
        if not gate:
            print("Corriger les signes/axes de la Jacobienne ou du contrôle avant entraînement.")
        return {"table": table, "gate_pass": bool(gate)}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--reward-profile", choices=REWARD_PROFILES,
                        default=DEFAULT_REWARD_PROFILE)
    args = parser.parse_args()
    if not rclpy.ok():
        rclpy.init()
    try:
        run(args.steps, reward_profile=args.reward_profile)
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
