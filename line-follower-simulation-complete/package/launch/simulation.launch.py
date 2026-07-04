"""
Launch : Gazebo + UR7e avec laser pointer + caméra statique eye-to-hand + ros2_control.
Scène line_follower.sdf (mur blanc + ligne bleue + point laser rouge + caméra sur support).

Séquence :
  1. OpaqueFunction : xacro → /tmp/ur_laser_robot.urdf
  2. robot_state_publisher
  3. Gazebo Harmonic
  4. gz_sim create (5 s timer) → spawn robot
  5. joint_state_broadcaster → forward_velocity_controller
  6. (pas de pose_bridge — pas d'objets dynamiques à tracker via TF)

Usage :
    ros2 launch ur7e_line_follower simulation.launch.py
    ros2 launch ur7e_line_follower simulation.launch.py headless:=true
"""
import subprocess
import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def _write_urdf(context, *args, **kwargs):
    ur_type_value = LaunchConfiguration('ur_type').perform(context)
    pkg_share     = get_package_share_directory('ur7e_line_follower')
    xacro_file    = os.path.join(pkg_share, 'urdf', 'ur7e_with_laser.urdf.xacro')
    controllers   = os.path.join(pkg_share, 'config', 'ur7e_laser_controllers.yaml')

    result = subprocess.run(
        [
            'xacro', xacro_file,
            f'ur_type:={ur_type_value}',
            f'simulation_controllers:={controllers}',
            'safety_limits:=true',
        ],
        capture_output=True, text=True, check=True,
    )
    with open('/tmp/ur_laser_robot.urdf', 'w') as f:
        f.write(result.stdout)
    return []


RUNTIME_WORLD = '/tmp/ur7e_line_follower_runtime.sdf'


def _write_runtime_world(context, *args, **kwargs):
    import numpy as np
    from ur7e_line_follower.target_line import curriculum_line_from_start, DEFAULT_HOME_DOT, arc_length
    from ur7e_line_follower.trajectory_store import (save_current_trajectory,
                                                      save_current_model_name,
                                                      LAUNCH_MODEL_NAME)
    from ur7e_line_follower.trajectory_visual import inject_trajectory_into_world

    seed_text = LaunchConfiguration('scene_seed').perform(context).strip()
    seed = None if seed_text in ('', 'none', 'None') else int(seed_text)
    rng = np.random.default_rng(seed)
    level = int(LaunchConfiguration('scene_level').perform(context))
    waypoints = curriculum_line_from_start(rng, DEFAULT_HOME_DOT, level=level)
    pkg_share = get_package_share_directory('ur7e_line_follower')
    base_world = os.path.join(pkg_share, 'worlds', 'line_follower.sdf')
    inject_trajectory_into_world(base_world, RUNTIME_WORLD, waypoints)
    path = save_current_trajectory(waypoints)
    save_current_model_name(LAUNCH_MODEL_NAME)
    print(f'[scene] runtime world généré : drapeaux + dessin en un seul modèle | longueur={arc_length(waypoints):.2f} m | curriculum={level}')
    print(f'[scene] trajectoire partagée: {path}')
    return []


def generate_launch_description():
    pkg            = FindPackageShare('ur7e_line_follower')
    ros_gz_sim_pkg = FindPackageShare('ros_gz_sim')

    ur_type = LaunchConfiguration('ur_type')
    headless = LaunchConfiguration('headless')
    scene_seed = LaunchConfiguration('scene_seed')
    scene_level = LaunchConfiguration('scene_level')

    xacro_file       = PathJoinSubstitution([pkg, 'urdf', 'ur7e_with_laser.urdf.xacro'])
    controllers_yaml = PathJoinSubstitution([pkg, 'config', 'ur7e_laser_controllers.yaml'])
    world_sdf        = RUNTIME_WORLD

    declare_ur_type = DeclareLaunchArgument(
        'ur_type', default_value='ur7e', choices=['ur5e', 'ur7e'],
        description='Universal Robots model type',
    )
    declare_headless = DeclareLaunchArgument(
        'headless', default_value='false', choices=['true', 'false'],
        description='Gazebo sans GUI si true',
    )
    declare_scene_seed = DeclareLaunchArgument(
        'scene_seed', default_value='',
        description='Seed optionnelle du dessin initial aléatoire',
    )
    declare_scene_level = DeclareLaunchArgument(
        'scene_level', default_value='0', choices=['0', '1', '2'],
        description='Difficulté du dessin initial: 0 simple, 1 modérée, 2 complète',
    )

    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name='xacro')]),
        ' ', xacro_file,
        ' ur_type:=', ur_type,
        ' simulation_controllers:=', controllers_yaml,
        ' safety_limits:=true',
    ])
    robot_description = {
        'robot_description': ParameterValue(robot_description_content, value_type=str)
    }

    write_urdf = OpaqueFunction(function=_write_urdf)
    write_runtime_world = OpaqueFunction(function=_write_runtime_world)

    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description, {'use_sim_time': True}],
    )

    gz_sim_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([ros_gz_sim_pkg, 'launch', 'gz_sim.launch.py'])
        ]),
        launch_arguments=[('gz_args', [world_sdf, ' -r -v 2'])],
        condition=UnlessCondition(headless),
    )
    gz_sim_headless = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([ros_gz_sim_pkg, 'launch', 'gz_sim.launch.py'])
        ]),
        launch_arguments=[('gz_args', [world_sdf,
            ' -s -r -v 2 --headless-rendering'])],
        condition=IfCondition(headless),
    )

    _spawn_node = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=[
            '-file', '/tmp/ur_laser_robot.urdf',
            '-name', 'ur',
            '-allow_renaming', 'true',
            '-x', '0', '-y', '0', '-z', '0',
        ],
    )
    # Gazebo avec la GUI met ~8-10 s à démarrer; 12 s pour laisser de la marge
    spawn_robot = TimerAction(period=12.0, actions=[_spawn_node])

    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='clock_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    # Bridge image caméra Gazebo → ROS2 + nœud KLT
    camera_bridge = Node(
        package='ros_gz_image',
        executable='image_bridge',
        name='camera_image_bridge',
        arguments=['/line_camera'],
        output='screen',
    )
    line_detector = Node(
        package='ur7e_line_follower',
        executable='line_detector',
        name='camera_line_detector',
        output='screen',
        parameters=[{'use_sim_laser_overlay': True}],
    )

    # Le dessin initial est déjà inclus dans le monde runtime avant Gazebo.

    jsb_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '-c', '/controller_manager'],
        output='screen',
    )
    arm_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['forward_velocity_controller', '-c', '/controller_manager'],
        output='screen',
    )

    load_jsb = RegisterEventHandler(OnProcessExit(
        target_action=_spawn_node,
        on_exit=[TimerAction(period=6.0, actions=[jsb_spawner])],
    ))
    load_arm = RegisterEventHandler(OnProcessExit(target_action=jsb_spawner, on_exit=[arm_spawner]))

    return LaunchDescription([
        declare_ur_type,
        declare_headless,
        declare_scene_seed,
        declare_scene_level,
        write_urdf,
        write_runtime_world,
        robot_state_pub,
        gz_sim_gui,
        gz_sim_headless,
        spawn_robot,
        clock_bridge,
        camera_bridge,
        line_detector,
        load_jsb,
        load_arm,
    ])
