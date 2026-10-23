#!/usr/bin/env python3
"""2027 tunnel profile; all common navigation ownership stays in the 2025 launch."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

from ats_mujoco_sim.launch_args import SIM_LIO_ARGUMENTS

# The 2027 scene uses the chassis-mounted Mid360 (report 2.1.8): front tilt 35
# degrees at the left edge midpoint.  Older scenes keep the legacy gimbal pose.
LIO_ARGUMENTS_2027 = {**SIM_LIO_ARGUMENTS, "lidar_mount_preset": "chassis"}


def generate_launch_description() -> LaunchDescription:
    share = FindPackageShare("ats_mujoco_sim")
    bringup_share = FindPackageShare("ats_sentry_bringup")
    return LaunchDescription([
        *[DeclareLaunchArgument(n, default_value=v) for n, v in LIO_ARGUMENTS_2027.items()],
        DeclareLaunchArgument("sim_rate_hz", default_value="auto"),
        DeclareLaunchArgument("lidar_rate_hz", default_value="auto"),
        DeclareLaunchArgument("start_x", default_value="13.40"),
        DeclareLaunchArgument("start_y", default_value="2.00"),
        DeclareLaunchArgument("start_z", default_value="0.330"),
        DeclareLaunchArgument("start_yaw", default_value="0.0"),
        DeclareLaunchArgument("planning_grid_owner", default_value="rog_map"),
        DeclareLaunchArgument("use_viewer", default_value="false"),
        DeclareLaunchArgument("show_viewer", default_value="false"),
        DeclareLaunchArgument("launch_physics", default_value="true"),
        DeclareLaunchArgument("launch_navigation", default_value="true"),
        DeclareLaunchArgument("use_rviz", default_value="false"),
        DeclareLaunchArgument("launch_mujoco_rviz", default_value="false"),
        DeclareLaunchArgument("force_body_yaw_follow", default_value="false"),
        DeclareLaunchArgument("body_yaw_follow_clearance", default_value="0.55"),
        DeclareLaunchArgument("enable_lidar", default_value="true"),
        DeclareLaunchArgument("lidar_backend", default_value="cpu"),
        DeclareLaunchArgument("lidar_downsample", default_value="auto"),
        DeclareLaunchArgument("enable_tof", default_value="false"),
        DeclareLaunchArgument("freeze_motion", default_value="false"),
        DeclareLaunchArgument("nav_start_delay_sec", default_value="6.0"),
        DeclareLaunchArgument("rog_map_start_delay_sec", default_value="12.0"),
        DeclareLaunchArgument("map_start_delay_sec", default_value="1.0"),
        DeclareLaunchArgument("rviz_delay_sec", default_value="4.0"),
        DeclareLaunchArgument("solver_mode", default_value="ilqr"),
        DeclareLaunchArgument("telemetry_sampling_window_cycles", default_value="0"),
        DeclareLaunchArgument("log_level", default_value="info"),
        DeclareLaunchArgument("enable_test_fault_injection", default_value="false"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([share, "launch", "rmuc_2025_mujoco.launch.py"])
            ),
            launch_arguments={
                "model_file": "rmuc_2027_swerve.xml",
                "map_yaml_file": PathJoinSubstitution([bringup_share, "map", "rmuc_2027.yaml"]),
                "prior_pcd_file": PathJoinSubstitution([bringup_share, "pcd", "rmuc_2027.pcd"]),
                "start_x": LaunchConfiguration("start_x"),
                "start_y": LaunchConfiguration("start_y"),
                "start_z": LaunchConfiguration("start_z"),
                "start_yaw": LaunchConfiguration("start_yaw"),
                "planning_grid_owner": LaunchConfiguration("planning_grid_owner"),
                "use_viewer": LaunchConfiguration("use_viewer"),
                "show_viewer": LaunchConfiguration("show_viewer"),
                "launch_physics": LaunchConfiguration("launch_physics"),
                "launch_navigation": LaunchConfiguration("launch_navigation"),
                "use_rviz": LaunchConfiguration("use_rviz"),
                "launch_mujoco_rviz": LaunchConfiguration("launch_mujoco_rviz"),
                "force_body_yaw_follow": LaunchConfiguration("force_body_yaw_follow"),
                "body_yaw_follow_clearance": LaunchConfiguration("body_yaw_follow_clearance"),
                "enable_lidar": LaunchConfiguration("enable_lidar"),
                "lidar_backend": LaunchConfiguration("lidar_backend"),
                "lidar_downsample": LaunchConfiguration("lidar_downsample"),
                "lidar_rate_hz": LaunchConfiguration("lidar_rate_hz"),
                "sim_rate_hz": LaunchConfiguration("sim_rate_hz"),
                **{name: LaunchConfiguration(name) for name in LIO_ARGUMENTS_2027},
                "enable_tof": LaunchConfiguration("enable_tof"),
                "freeze_motion": LaunchConfiguration("freeze_motion"),
                "nav_start_delay_sec": LaunchConfiguration("nav_start_delay_sec"),
                "rog_map_start_delay_sec": LaunchConfiguration("rog_map_start_delay_sec"),
                "map_start_delay_sec": LaunchConfiguration("map_start_delay_sec"),
                "rviz_delay_sec": LaunchConfiguration("rviz_delay_sec"),
                "solver_mode": LaunchConfiguration("solver_mode"),
                "telemetry_sampling_window_cycles": LaunchConfiguration(
                    "telemetry_sampling_window_cycles"
                ),
                "log_level": LaunchConfiguration("log_level"),
                "enable_test_fault_injection": LaunchConfiguration("enable_test_fault_injection"),
            }.items(),
        )
    ])
