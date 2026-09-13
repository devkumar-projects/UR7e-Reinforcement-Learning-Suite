#!/usr/bin/env bash
set -Eeo pipefail

# Stop only the processes belonging to this demo and do it gracefully.
# This avoids a broad pkill/SIGKILL that could terminate unrelated ROS work.
stop_pattern() {
  local pattern="$1"
  local pid
  while read -r pid; do
    [[ "$pid" == "$$" ]] && continue
    kill -INT "$pid" 2>/dev/null || true
  done < <(pgrep -f -- "$pattern" || true)
}

for pattern in   'camera_laser_calibrator'   'visual_policy_runner'   'rqt_image_view'   'ur7e_visual_detector'   'v4l2_camera_node'   'ros2 launch ur7e_visual_rl_demo system.launch.py'; do
  stop_pattern "$pattern"
done

sleep 3

# Escalate only to SIGTERM for processes that are still alive.
for pattern in   'camera_laser_calibrator'   'visual_policy_runner'   'ur7e_visual_detector'   'v4l2_camera_node'   'ros2 launch ur7e_visual_rl_demo system.launch.py'; do
  while read -r pid; do
    [[ "$pid" == "$$" ]] && continue
    kill -TERM "$pid" 2>/dev/null || true
  done < <(pgrep -f -- "$pattern" || true)
done
