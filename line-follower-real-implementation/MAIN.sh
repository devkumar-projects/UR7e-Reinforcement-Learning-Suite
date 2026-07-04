#!/usr/bin/env bash
set -Eeo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
source "$ROOT/scripts/common.sh"
MODE="${1:-help}"

wait_topic() {
  local topic="$1" tries="${2:-35}"
  for ((i=1;i<=tries;i++)); do
    if timeout 2 ros2 topic echo "$topic" --once >/dev/null 2>&1; then
      echo "[OK] $topic"; return 0
    fi
    printf '.'; sleep 1
  done
  echo; echo "[ERREUR] Topic absent: $topic"; return 1
}

start_stack() {
  local debug="${1:-false}"
  [[ -s "$CALIBRATION_FILE" ]] || { echo "[ERREUR] Calibration UR absente. Lance: ./MAIN.sh extract-calibration"; return 2; }
  CAMERA="$($ROOT/scripts/detect_camera.sh)"
  echo "[OK] Caméra couleur: $CAMERA"
  "$ROOT/scripts/stop.sh" >/dev/null 2>&1 || true
  mkdir -p "$HOME/.ros/ur7e_line_follower/session_logs"
  STACK_LOG="$HOME/.ros/ur7e_line_follower/session_logs/stack_$(date +%Y%m%d_%H%M%S).log"
  ros2 launch ur7e_visual_rl_demo system.launch.py \
    robot_ip:="$ROBOT_IP" ur_type:="$UR_TYPE" \
    calibration_file:="$CALIBRATION_FILE" \
    video_device:="$CAMERA" camera_topic:="$CAMERA_TOPIC" \
    homography_file:="$CAMERA_HOMOGRAPHY_FILE" \
    debug_overlay:="$debug" launch_rviz:="$LAUNCH_RVIZ" \
    >"$STACK_LOG" 2>&1 &
  STACK_PID=$!
  echo "[INFO] Stack PID=$STACK_PID | log=$STACK_LOG"
  READY=1
  wait_topic /joint_states || READY=0
  wait_topic /tcp_pose_broadcaster/pose || READY=0
  wait_topic /line_camera || READY=0
  wait_topic /line_detection || READY=0
  wait_topic /line_guidance || READY=0
  wait_topic /camera_wall_measurement || READY=0
  if [[ "$READY" != 1 ]]; then
    tail -100 "$STACK_LOG"; return 2
  fi
}

case "$MODE" in
  install)
    sudo apt update
    sudo apt install -y python3-colcon-common-extensions python3-rosdep v4l-utils \
      ros-jazzy-ur-robot-driver ros-jazzy-ur-calibration ros-jazzy-v4l2-camera \
      ros-jazzy-rqt-image-view ros-jazzy-ros2-control ros-jazzy-ros2-controllers
    python3 -m venv --system-site-packages "$HOME/venv_ur7e_visual_rl"
    source "$HOME/venv_ur7e_visual_rl/bin/activate"
    pip install --upgrade pip
    pip install 'numpy<2' opencv-python PyYAML scipy stable-baselines3 gymnasium
    ;;
  build)
    "$ROOT/scripts/build.sh"
    ;;
  extract-calibration)
    "$ROOT/scripts/extract_calibration.sh"
    ;;
  calibrate)
    start_stack true
    wait_topic /line_measurement
    ros2 run rqt_image_view rqt_image_view /line_debug >/tmp/ur7e_line_view.log 2>&1 &
    echo '[INFO] Fenêtre /line_debug ouverte. Vérifier que le cercle suit le vrai laser.'
    "$HOME/venv_ur7e_visual_rl/bin/python" -u -m ur7e_visual_rl_demo.camera_laser_calibrator --ros-args \
      -p calibration_file:="$CALIBRATION_FILE" \
      -p output_file:="$CAMERA_HOMOGRAPHY_FILE" \
      -p wall_x:="$WALL_X_M" -p laser_axis:="$LASER_AXIS" \
      -p laser_origin_offset_m:="$LASER_ORIGIN_OFFSET_M" -p required_samples:=12
    ;;
  shadow)
    [[ -s "$CAMERA_HOMOGRAPHY_FILE" ]] || { echo '[ERREUR] Homographie absente. Lance: ./MAIN.sh calibrate'; exit 2; }
    start_stack false
    "$HOME/venv_ur7e_visual_rl/bin/python" -u -m ur7e_visual_rl_demo.visual_policy_runner --ros-args \
      -p model_path:="$MODEL_FILE" -p calibration_file:="$CALIBRATION_FILE" \
      -p wall_x:="$WALL_X_M" -p laser_axis:="$LASER_AXIS" -p laser_origin_offset_m:="$LASER_ORIGIN_OFFSET_M" \
      -p mode:=shadow -p control_mode:="$CONTROL_MODE" -p duration_s:="$SHADOW_DURATION_S" \
      -p segment_s:="$SEGMENT_DURATION_S" -p max_wall_speed_m_s:="$MAX_WALL_SPEED_M_S" \
      -p max_joint_speed_rad_s:="$MAX_JOINT_SPEED_RAD_S" -p rl_weight:="$RL_WEIGHT" \
      -p min_klt_confidence:="$MIN_KLT_CONFIDENCE" -p max_sensor_age_s:="$MAX_SENSOR_AGE_S" \
      -p max_cross_track_m:="$MAX_CROSS_TRACK_M" -p max_mgd_camera_disagreement_m:="$MAX_MGD_CAMERA_DISAGREEMENT_M" \
      -p max_task_condition:="$MAX_TASK_CONDITION" -p max_ekf_nis:="$MAX_EKF_NIS"
    ;;
  demo)
    [[ -s "$CAMERA_HOMOGRAPHY_FILE" ]] || { echo '[ERREUR] Homographie absente. Lance: ./MAIN.sh calibrate'; exit 2; }
    echo 'ATTENTION: déplacement réel du UR7e.'
    read -r -p 'Tape MOVE_UR7E_CAMERA_LASER_RL : ' CONFIRM
    [[ "$CONFIRM" == MOVE_UR7E_CAMERA_LASER_RL ]] || { echo 'Annulé.'; exit 2; }
    start_stack false
    "$HOME/venv_ur7e_visual_rl/bin/python" -u -m ur7e_visual_rl_demo.visual_policy_runner --ros-args \
      -p model_path:="$MODEL_FILE" -p calibration_file:="$CALIBRATION_FILE" \
      -p wall_x:="$WALL_X_M" -p laser_axis:="$LASER_AXIS" -p laser_origin_offset_m:="$LASER_ORIGIN_OFFSET_M" \
      -p mode:=move -p confirmation:=MOVE_UR7E_CAMERA_LASER_RL -p control_mode:="$CONTROL_MODE" \
      -p duration_s:="$MOVE_DURATION_S" -p segment_s:="$SEGMENT_DURATION_S" \
      -p max_wall_speed_m_s:="$MAX_WALL_SPEED_M_S" -p max_joint_speed_rad_s:="$MAX_JOINT_SPEED_RAD_S" \
      -p rl_weight:="$RL_WEIGHT" -p visual_cross_track_gain:="$VISUAL_CROSS_TRACK_GAIN" \
      -p visual_forward_fraction:="$VISUAL_FORWARD_FRACTION" -p mgi_damping:="$MGI_DAMPING" \
      -p min_klt_confidence:="$MIN_KLT_CONFIDENCE" -p max_sensor_age_s:="$MAX_SENSOR_AGE_S" \
      -p max_cross_track_m:="$MAX_CROSS_TRACK_M" -p max_mgd_camera_disagreement_m:="$MAX_MGD_CAMERA_DISAGREEMENT_M" \
      -p max_task_condition:="$MAX_TASK_CONDITION" -p max_ekf_nis:="$MAX_EKF_NIS"
    ;;
  view)
    ros2 run rqt_image_view rqt_image_view /line_debug
    ;;
  status)
    "$ROOT/scripts/status.sh"
    ;;
  stop)
    "$ROOT/scripts/stop.sh"
    ;;
  *)
    cat <<USAGE
Usage: ./MAIN.sh MODE
  install               Installer les dépendances
  build                 Compiler le package
  extract-calibration   Extraire la calibration usine du UR7e
  calibrate             Lancer stack + vue caméra + calibration plan
  shadow                Test complet sans mouvement
  demo                  Suivi de ligne sur UR7e réel
  view                  Ouvrir /line_debug
  status                Vérifier nœuds, contrôleurs et topics
  stop                  Arrêter les processus
USAGE
    ;;
esac
