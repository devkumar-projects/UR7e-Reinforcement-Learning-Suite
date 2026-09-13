#!/usr/bin/env bash
set -Eeo pipefail
source "$(dirname "$0")/common.sh"
cd "$ROOT/ros2_ws"
colcon build --symlink-install --packages-select ur7e_visual_rl_demo
