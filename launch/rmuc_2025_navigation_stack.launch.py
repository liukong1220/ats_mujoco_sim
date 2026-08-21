#!/usr/bin/env python3
"""Start only the RMUC 2025 ATS localization, planning and control stack."""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare("ats_mujoco_sim"),
                    "launch",
                    "rmuc_2025_mujoco.launch.py",
                ])
            ]),
            launch_arguments={
                "launch_physics": "false",
                "launch_navigation": "true",
            }.items(),
        )
    ])
