#!/usr/bin/env bash
set -Eeo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${UR7E_CONFIG:-$ROOT/config/real.env}"
[[ -f "$CONFIG" ]] || { echo "[ERREUR] Config absente: $CONFIG"; return 1 2>/dev/null || exit 1; }
set -a
# shellcheck disable=SC1090
source "$CONFIG"
set +a
CALIBRATION_FILE="${CALIBRATION_FILE/\$HOME/$HOME}"
CAMERA_HOMOGRAPHY_FILE="${CAMERA_HOMOGRAPHY_FILE/\$HOME/$HOME}"
MODEL_FILE="${MODEL_FILE/__ROOT__/$ROOT}"
export ROOT CONFIG CALIBRATION_FILE CAMERA_HOMOGRAPHY_FILE MODEL_FILE
set +u
source /opt/ros/jazzy/setup.bash
[[ -f "$HOME/venv_ur7e_visual_rl/bin/activate" ]] && source "$HOME/venv_ur7e_visual_rl/bin/activate"
[[ -f "$ROOT/ros2_ws/install/local_setup.bash" ]] && source "$ROOT/ros2_ws/install/local_setup.bash"
set +u
export PYTHONNOUSERSITE=1
