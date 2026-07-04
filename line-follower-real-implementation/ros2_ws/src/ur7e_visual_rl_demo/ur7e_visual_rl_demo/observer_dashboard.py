"""Compact terminal dashboard for the visual observer chain."""
from __future__ import annotations

import os
import time
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class ObserverDashboard(Node):
    def __init__(self) -> None:
        super().__init__('ur7e_observer_dashboard')
        self.det = np.zeros(7)
        self.guide = np.zeros(3)
        self.wall = np.zeros(11)
        self.last = {'det': 0.0, 'guide': 0.0, 'wall': 0.0}
        self.create_subscription(Float32MultiArray, '/line_detection', self.det_cb, 20)
        self.create_subscription(Float32MultiArray, '/line_guidance', self.guide_cb, 20)
        self.create_subscription(Float32MultiArray, '/camera_wall_measurement', self.wall_cb, 20)
        self.create_timer(0.25, self.render)

    def det_cb(self, msg):
        if len(msg.data) >= 7:
            self.det[:] = msg.data[:7]; self.last['det'] = time.monotonic()

    def guide_cb(self, msg):
        if len(msg.data) >= 3:
            self.guide[:] = msg.data[:3]; self.last['guide'] = time.monotonic()

    def wall_cb(self, msg):
        if len(msg.data) >= 11:
            self.wall[:] = msg.data[:11]; self.last['wall'] = time.monotonic()

    def render(self):
        now = time.monotonic()
        os.system('clear')
        print('UR7e CAMERA + LASER OBSERVER DASHBOARD')
        print('--------------------------------------')
        print(f"line={int(self.det[0]>0.5)} laser={int(self.det[6]>0.5)} KLT={self.det[2]:.3f} coverage={self.det[5]:.3f}")
        print(f"offset_norm={self.det[1]:+.4f} tangent_img=({self.det[3]:+.3f},{self.det[4]:+.3f})")
        print(f"lookahead_img=({self.guide[0]:+.3f},{self.guide[1]:+.3f}) progress={100*self.guide[2]:.1f}%")
        if self.wall[0] > 0.5:
            print(f"laser_wall=({self.wall[1]:+.4f},{self.wall[2]:+.4f}) m")
            print(f"line_wall =({self.wall[3]:+.4f},{self.wall[4]:+.4f}) m")
            print(f"tangent_wall=({self.wall[5]:+.3f},{self.wall[6]:+.3f})")
            print(f"cross_wall=({self.wall[8]*1000:+.1f},{self.wall[9]*1000:+.1f}) mm |norm|={np.linalg.norm(self.wall[8:10])*1000:.1f} mm")
        else:
            print('metric homography measurement: INVALID')
        print('ages:', ', '.join(f"{k}={now-v:.2f}s" if v else f"{k}=never" for k,v in self.last.items()))
        print('\nOpen the debug image in another terminal:')
        print('  ros2 run rqt_image_view rqt_image_view /line_debug')


def main(args=None):
    rclpy.init(args=args)
    node = ObserverDashboard()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node(); rclpy.shutdown()
