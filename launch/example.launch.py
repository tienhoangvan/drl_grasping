"""Launch example (Python) of throwing an object"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    # Launch Arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default=True)
    config_rviz2 = LaunchConfiguration('config_rviz2', default=os.path.join(get_package_share_directory("drl_grasping"),
                                                                            "launch", "rviz.rviz"))

    return LaunchDescription([
        # Launch Arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value=use_sim_time,
            description='If true, use simulated clock'),
        DeclareLaunchArgument(
            'config_rviz2',
            default_value=config_rviz2,
            description='Path to config for RViz2'),

        # MoveIt2 move_group action server with necessary ROS2 <-> Ignition bridges
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [os.path.join(get_package_share_directory("ign_moveit2"),
                              "launch", "ign_moveit2.launch.py")]),
            launch_arguments=[('use_sim_time', use_sim_time),
                              ('config_rviz2', config_rviz2)]),

        # RGB-D camera (IGN -> ROS2)
        Node(package='ros_ign_bridge',
             executable='parameter_bridge',
             name='parameter_bridge_rs_d435',
             output="screen",
             arguments=[
                 # '/rs_d435/image@sensor_msgs/msg/Image[ignition.msgs.Image',
                 # '/rs_d435/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image',
                 '/rs_d435/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked'
             ],
             parameters=[{'use_sim_time': use_sim_time}]),


        Node(package='drl_grasping',
             executable='train_grasping.py',
             name='drl_grasping_train_grasping',
             output='screen',
             parameters=[{'use_sim_time': use_sim_time}]),
    ])
