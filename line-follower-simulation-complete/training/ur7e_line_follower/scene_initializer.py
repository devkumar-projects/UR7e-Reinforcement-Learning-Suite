"""Show flags and one anchored random drawing automatically at Gazebo startup."""
from __future__ import annotations
import argparse
import numpy as np
import rclpy
from .bridge import LineFollowerBridge
from .target_line import curriculum_line_from_start, DEFAULT_HOME_DOT, arc_length
from .trajectory_store import save_current_trajectory

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--level', type=int, default=0, choices=[0, 1, 2])
    args, _ = parser.parse_known_args()
    if not rclpy.ok():
        rclpy.init()
    node = LineFollowerBridge(visual_enabled=True, node_name='ur7e_scene_initializer')
    try:
        rng = np.random.default_rng(args.seed)
        wp = curriculum_line_from_start(rng, DEFAULT_HOME_DOT, level=args.level)
        ok = node.show_trajectory_with_retry(wp, attempts=20, delay_s=0.5)
        if not ok:
            raise RuntimeError('impossible d afficher drapeaux + dessin via set_pose_vector')
        path = save_current_trajectory(wp)
        print(f'[scene] drapeaux + dessin aléatoire affichés ensemble | longueur={arc_length(wp):.2f} m')
        print(f'[scene] trajectoire partagée: {path}')
    finally:
        try: node._spin_executor.shutdown(wait_for_completion=False)
        except Exception: pass
        try: node.destroy_node()
        except Exception: pass
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__': main()
