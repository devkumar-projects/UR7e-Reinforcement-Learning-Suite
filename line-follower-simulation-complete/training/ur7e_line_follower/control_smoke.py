"""Deterministic action-direction and repeatability test for the UR7e controller.

Run with Gazebo already active::

    ros2 run ur7e_line_follower control_smoke --repeats 5 --pulses 4

Each action starts from HOME.  The test checks the sign of the measured laser
motion and reports dispersion across identical state/action repetitions.
"""
from __future__ import annotations

import argparse
import numpy as np
import rclpy

from .bridge import LineFollowerBridge
from .control import wall_action_to_joint_velocity, MAX_WALL_SPEED_M_S
from .trajectory_store import load_current_trajectory


def _single_trial(node: LineFollowerBridge, action: np.ndarray, pulses: int) -> np.ndarray:
    node.reset_world()
    node.wait_for_n_steps(25, timeout=2.0)
    start = node.get_laser_dot()
    if start is None:
        raise RuntimeError('laser hors du plan du mur au reset')
    previous = np.zeros(2, dtype=np.float64)
    for _ in range(max(1, int(pulses))):
        qdot, previous = wall_action_to_joint_velocity(
            node.joint_pos.copy(), action,
            previous_wall_velocity=previous,
            max_speed=0.45 * MAX_WALL_SPEED_M_S,
        )
        node.publish_velocity(qdot)
        if not node.wait_for_n_steps(25, timeout=1.0):
            raise RuntimeError('timeout /joint_states pendant le pulse')
        node.stop()
        node.wait_for_n_steps(5, timeout=0.5)
    end = node.get_laser_dot()
    if end is None:
        raise RuntimeError('laser hors du plan du mur après le pulse')
    return np.asarray(end, dtype=float) - np.asarray(start, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--pulses', type=int, default=4)
    parser.add_argument('--min-motion-mm', type=float, default=1.0)
    parser.add_argument('--max-cv', type=float, default=0.35,
                        help='coefficient de variation maximal sur l axe commandé')
    args = parser.parse_args()

    if not rclpy.ok():
        rclpy.init()
    node = LineFollowerBridge(visual_enabled=False)
    runtime_line = load_current_trajectory()
    if runtime_line is not None:
        print(f'[smoke] dessin runtime déjà présent ({len(runtime_line)} points)')
    else:
        print('[smoke] aucun dessin runtime enregistré; test du contrôle seul')
    cases = [
        ('+Y', np.array([+0.65, 0.00]), 0, +1.0),
        ('-Y', np.array([-0.65, 0.00]), 0, -1.0),
        ('+Z', np.array([0.00, +0.65]), 1, +1.0),
        ('-Z', np.array([0.00, -0.65]), 1, -1.0),
    ]
    failures = []
    try:
        for label, action, axis, sign in cases:
            deltas = np.stack([
                _single_trial(node, action, args.pulses)
                for _ in range(max(1, args.repeats))
            ])
            mean = deltas.mean(axis=0)
            std = deltas.std(axis=0)
            commanded = sign * mean[axis]
            cv = std[axis] / max(abs(mean[axis]), 1e-9)
            print(
                f'[smoke] {label}: mean Δ[y,z]={1000*mean} mm | '
                f'std={1000*std} mm | CV axe={cv:.3f}')
            if commanded * 1000.0 < args.min_motion_mm:
                failures.append(
                    f'{label}: signe/amplitude invalide ({1000*mean[axis]:+.2f} mm)')
            if cv > args.max_cv:
                failures.append(f'{label}: répétabilité insuffisante (CV={cv:.3f})')
        if failures:
            raise RuntimeError('ÉCHEC control_smoke:\n  - ' + '\n  - '.join(failures))
        print('[smoke] PASS: axes, signes et répétabilité cohérents.')
    finally:
        node.stop()
        try:
            node._spin_executor.shutdown(wait_for_completion=False)
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
