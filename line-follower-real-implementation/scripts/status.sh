#!/usr/bin/env bash
set -Eeo pipefail
source "$(dirname "$0")/common.sh"
echo '=== NODES ==='; ros2 node list || true
echo '=== CONTROLLERS ==='; ros2 control list_controllers || true
echo '=== TOPICS ==='
for t in /joint_states /tcp_pose_broadcaster/pose /line_camera /line_detection /line_guidance /camera_wall_measurement /line_debug; do
  printf '%-38s' "$t"
  timeout 3 ros2 topic echo "$t" --once >/dev/null 2>&1 && echo OK || echo ABSENT
done
