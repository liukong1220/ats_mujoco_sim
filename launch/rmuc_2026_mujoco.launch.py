#!/usr/bin/env python3
"""Compatibility entry: RMUC 2027, starting at the base with chassis yaw zero."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

from ats_mujoco_sim.launch_args import load_mujoco_launch_defaults


def generate_launch_description() -> LaunchDescription:
    d = load_mujoco_launch_defaults("rmuc_2027")
    return LaunchDescription([
        DeclareLaunchArgument("start_x", default_value=d["start_x"]),
        DeclareLaunchArgument("start_y", default_value=d["start_y"]),
        DeclareLaunchArgument("start_z", default_value=d["start_z"]),
        DeclareLaunchArgument("start_yaw", default_value=d["start_yaw"]),
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
