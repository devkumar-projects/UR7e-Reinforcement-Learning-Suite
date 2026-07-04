#!/usr/bin/env bash
set +e
pkill -INT -f camera_laser_calibrator
pkill -INT -f visual_policy_runner
pkill -INT -f rqt_image_view
pkill -INT -f ur7e_visual_detector
pkill -INT -f v4l2_camera_node
pkill -INT -f 'ros2 launch ur7e_visual_rl_demo system.launch.py'
sleep 3
