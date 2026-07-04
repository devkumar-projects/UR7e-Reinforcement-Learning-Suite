#!/usr/bin/env bash
set -Eeo pipefail
source "$(dirname "$0")/common.sh"
mkdir -p "$(dirname "$CALIBRATION_FILE")"
ros2 launch ur_calibration calibration_correction.launch.py \
  robot_ip:="$ROBOT_IP" target_filename:="$CALIBRATION_FILE"
