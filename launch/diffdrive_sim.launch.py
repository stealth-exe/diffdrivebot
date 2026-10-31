import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    """
    Gazebo world + differential-drive bot (diffdrive_bot.xacro) + ROS<->Gazebo bridge
    (+ the diff-drive autopilot node unless autopilot:=false).

    ros2 launch diffdrivebot diffdrive_sim.launch.py world:=/path/to/line_world.sdf \
        x:=0.0 y:=0.0 yaw:=0.0

    Worlds like hexagon.sdf / circle_2_5.sdf load the Sensors system themselves, so the robot must not
    (diffdrive_bot.xacro default sensors_plugin:=false). For a world without it, add the xacro argument.
    """
    pkg = get_package_share_directory('diffdrivebot')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')
    xacro_file = os.path.join(pkg, 'urdf', 'diffdrive_bot.xacro')
    params_file = os.path.join(pkg, 'config', 'diffdrive_parameters.yaml')

    args = [
        DeclareLaunchArgument('world', default_value='empty.sdf', description='SDF world to load'),
        DeclareLaunchArgument('x', default_value='0.0'),
        DeclareLaunchArgument('y', default_value='0.0'),
        DeclareLaunchArgument('z', default_value='0.06'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
        DeclareLaunchArgument('autopilot', default_value='true', description='start the autopilot node'),
    ]

    robot_description = ParameterValue(Command([FindExecutable(name='xacro'), ' ', xacro_file]), value_type=str)

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': ['-r -v 2 ', LaunchConfiguration('world')],
                          'on_exit_shutdown': 'true'}.items())

    robot_state_publisher = Node(package='robot_state_publisher', executable='robot_state_publisher',
                                 parameters=[{'robot_description': robot_description}], output='screen')

    spawn = Node(package='ros_gz_sim', executable='create', output='screen',
                 arguments=['-topic', 'robot_description', '-name', 'diffdrive_bot',
                            '-x', LaunchConfiguration('x'), '-y', LaunchConfiguration('y'),
                            '-z', LaunchConfiguration('z'), '-Y', LaunchConfiguration('yaw')])

    # ']' = ROS -> Gazebo, '[' = Gazebo -> ROS
    bridge = Node(package='ros_gz_bridge', executable='parameter_bridge', output='screen',
                  arguments=['/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
                             '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
                             '/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
                             '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model'])

    autopilot = Node(package='diffdrivebot', executable='autopilot_diffdrive', output='screen',
                     parameters=[params_file], condition=IfCondition(LaunchConfiguration('autopilot')))

    return LaunchDescription(args + [gazebo, robot_state_publisher, spawn, bridge, autopilot])
