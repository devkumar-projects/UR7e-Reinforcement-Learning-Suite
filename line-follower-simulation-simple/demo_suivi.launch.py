"""
Launch file — Démo suivi de trajectoire UR7e
──────────────────────────────────────────────
Lance tout en une commande :
  1. robot_state_publisher  (URDF UR7e)
  2. viz_traj               (agent SAC suivi + génère les lignes)
  3. rviz2                  (config ur7e_rl.rviz : mur + ligne + trace)

Usage :
  source /opt/ros/humble/setup.bash
  ros2 launch demo_suivi.launch.py

Ou directement depuis le dossier du lot :
  cd <dossier-du-lot>
  ros2 launch ./demo_suivi.launch.py
"""
import os
from launch import LaunchDescription
from launch.actions import TimerAction, ExecuteProcess
from launch_ros.actions import Node

# ── Chemins ──────────────────────────────────
TRAJ_DIR    = os.path.dirname(os.path.abspath(__file__))
UR7E_WS     = os.path.dirname(TRAJ_DIR)
VENV_PYTHON = os.environ.get('UR7E_VENV_PYTHON', 'python3')
URDF_FILE   = os.path.join(TRAJ_DIR, 'ur7e_generated.urdf')
RVIZ_CONFIG = os.path.join(TRAJ_DIR, 'ur7e_rl.rviz')

# PYTHONPATH complet : projet + venv éventuel + ROS
VENV_SITE  = os.path.expanduser('~/venv_ur7e/lib/python3.10/site-packages')
ROS_PYTHON = '/opt/ros/humble/local/lib/python3.10/dist-packages'
ROS_LIB    = '/opt/ros/humble/lib/python3.10/site-packages'

ENV_VARS = {
    'PYTHONUNBUFFERED' : '1',
    'PYTHONPATH'       : ':'.join([UR7E_WS, TRAJ_DIR, VENV_SITE,
                                    ROS_PYTHON, ROS_LIB]),
    'DISPLAY'          : os.environ.get('DISPLAY', ':0'),
    'XLA_FLAGS'        : '--xla_gpu_cuda_data_dir=/usr/local/cuda-12.3',
}


def get_urdf():
    with open(URDF_FILE) as f:
        return f.read()


def generate_launch_description():
    urdf = get_urdf()

    return LaunchDescription([

        # ── 1. robot_state_publisher ──────────────────
        Node(
            package    = 'robot_state_publisher',
            executable = 'robot_state_publisher',
            name       = 'robot_state_publisher',
            parameters = [{'robot_description': urdf}],
            output     = 'screen',
        ),

        # ── 2. Agent suivi (démarre après 1s) ─────────
        TimerAction(
            period  = 1.0,
            actions = [
                ExecuteProcess(
                    cmd = [VENV_PYTHON,
                           os.path.join(TRAJ_DIR, 'viz_traj.py')],
                    additional_env = ENV_VARS,
                    output         = 'screen',
                )
            ]
        ),

        # ── 3. RViz2 avec config suivi (démarre après 3s) ──
        TimerAction(
            period  = 3.0,
            actions = [
                Node(
                    package    = 'rviz2',
                    executable = 'rviz2',
                    name       = 'rviz2',
                    arguments  = ['-d', RVIZ_CONFIG],
                    output     = 'screen',
                )
            ]
        ),

    ])
