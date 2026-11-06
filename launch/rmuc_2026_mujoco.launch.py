#!/usr/bin/env python3
"""Compatibility entry: RMUC 2027, starting at the base with chassis yaw zero."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("start_x", default_value="3.37"),
        DeclareLaunchArgument("start_y", default_value="9.03"),
        DeclareLaunchArgument("start_z", default_value="0.465"),
        DeclareLaunchArgument("start_yaw", default_value="0.0"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare("ats_mujoco_sim"),
                    "launch",
                    "rmuc_2027_mujoco.launch.py",
                ])
            ])
        )
    ])
