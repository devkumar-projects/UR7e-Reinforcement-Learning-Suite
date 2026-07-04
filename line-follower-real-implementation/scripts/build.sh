#!/usr/bin/env bash
set -Eeo pipefail
source "$(dirname "$0")/common.sh"
cd "$ROOT/ros2_ws"
rm -rf build/ur7e_visual_rl_demo install/ur7e_visual_rl_demo
colcon build --symlink-install --packages-select ur7e_visual_rl_demo
