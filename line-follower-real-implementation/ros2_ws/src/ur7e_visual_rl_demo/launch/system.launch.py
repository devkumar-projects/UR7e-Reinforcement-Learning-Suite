from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    robot_ip = LaunchConfiguration('robot_ip')
    ur_type = LaunchConfiguration('ur_type')
    calibration_file = LaunchConfiguration('calibration_file')
    video_device = LaunchConfiguration('video_device')
    camera_topic = LaunchConfiguration('camera_topic')
    homography_file = LaunchConfiguration('homography_file')
    debug_overlay = LaunchConfiguration('debug_overlay')
    launch_rviz = LaunchConfiguration('launch_rviz')

    ur_driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('ur_robot_driver'),
                'launch',
                'ur_control.launch.py',
            ])
        ),
        launch_arguments={
            'ur_type': ur_type,
            'robot_ip': robot_ip,
            'kinematics_params_file': calibration_file,
            'initial_joint_controller': 'scaled_joint_trajectory_controller',
            'launch_rviz': launch_rviz,
            'headless_mode': 'false',
        }.items(),
    )

    camera = Node(
        package='v4l2_camera',
        executable='v4l2_camera_node',
        name='line_camera',
        output='screen',
        parameters=[{
            'video_device': video_device,
            'pixel_format': 'YUYV',
            'output_encoding': 'rgb8',
            'image_size': [1280, 720],
            'time_per_frame': [1, 15],
        }],
        remappings=[('image_raw', camera_topic)],
    )

    detector = Node(
        package='ur7e_visual_rl_demo',
        executable='visual_detector',
        name='ur7e_visual_detector',
        output='screen',
        parameters=[{
            'homography_file': homography_file,
            'blue_hsv_lo': [95, 55, 35],
            'blue_hsv_hi': [145, 255, 255],
            'red_hsv_lo1': [0, 35, 100],
            'red_hsv_hi1': [22, 255, 255],
            'red_hsv_lo2': [155, 35, 100],
            'red_hsv_hi2': [180, 255, 255],
            'green_hsv_lo': [38, 60, 35],
            'green_hsv_hi': [92, 255, 255],
            'debug_overlay': debug_overlay,
        }],
        remappings=[('/line_camera', camera_topic)],
    )

    return LaunchDescription([
        DeclareLaunchArgument('robot_ip', default_value='192.168.186.141'),
        DeclareLaunchArgument('ur_type', default_value='ur7e'),
        DeclareLaunchArgument('calibration_file'),
        DeclareLaunchArgument('video_device'),
        DeclareLaunchArgument('camera_topic', default_value='/line_camera'),
        DeclareLaunchArgument('homography_file'),
        DeclareLaunchArgument('debug_overlay', default_value='false'),
        DeclareLaunchArgument('launch_rviz', default_value='false'),
        ur_driver,
        camera,
        detector,
    ])
