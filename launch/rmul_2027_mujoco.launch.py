#!/usr/bin/env python3
"""RMUL 2027 3V3 field (12.6 x 8.6 m); all common navigation ownership stays in the 2025 launch.

The field assets come from scripts/convert_rmul_2027_map.py.  Map frame: origin at the outer lower-left
corner, the start zone is the lower-left corner, the two tunnels run along x with the lintel lifted.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

from ats_mujoco_sim.launch_args import load_lio_arguments, load_mujoco_launch_defaults

DEFAULTS_RMUL_2027 = load_mujoco_launch_defaults("rmul_2027")
LIO_ARGUMENTS_RMUL_2027 = load_lio_arguments("rmul_2027")


def generate_launch_description() -> LaunchDescription:
    share = FindPackageShare("ats_mujoco_sim")
    bringup_share = FindPackageShare("ats_sentry_bringup")
    d = DEFAULTS_RMUL_2027
    return LaunchDescription([
        DeclareLaunchArgument("model_file", default_value=d["model_file"]),
        DeclareLaunchArgument("map_yaml_file", default_value=PathJoinSubstitution(
            [bringup_share, "map", d["map_yaml_name"]])),
        DeclareLaunchArgument("prior_pcd_file", default_value=PathJoinSubstitution(
            [bringup_share, "pcd", d["prior_pcd_name"]])),
        DeclareLaunchArgument("scene_navigation_profile", default_value=PathJoinSubstitution(
            [share, "config", d["scene_navigation_profile_name"]])),
        *[DeclareLaunchArgument(n, default_value=v) for n, v in LIO_ARGUMENTS_RMUL_2027.items()],
        DeclareLaunchArgument("lio_lease_wall_scale", default_value=d["lio_lease_wall_scale"]),
        DeclareLaunchArgument("sim_rate_hz", default_value=d["sim_rate_hz"]),
        DeclareLaunchArgument("lidar_rate_hz", default_value=d["lidar_rate_hz"]),
        DeclareLaunchArgument("start_x", default_value=d["start_x"]),
        DeclareLaunchArgument("start_y", default_value=d["start_y"]),
        DeclareLaunchArgument("start_z", default_value=d["start_z"]),
        DeclareLaunchArgument("start_yaw", default_value=d["start_yaw"]),
        DeclareLaunchArgument("planning_grid_owner", default_value=d["planning_grid_owner"]),
        DeclareLaunchArgument("slam", default_value=d["slam"]),
        DeclareLaunchArgument("use_viewer", default_value=d["use_viewer"]),
        DeclareLaunchArgument("show_viewer", default_value=d["show_viewer"]),
        DeclareLaunchArgument("launch_physics", default_value=d["launch_physics"]),
        DeclareLaunchArgument("launch_navigation", default_value=d["launch_navigation"]),
        DeclareLaunchArgument("use_rviz", default_value=d["use_rviz"]),
        DeclareLaunchArgument("launch_mujoco_rviz", default_value=d["launch_mujoco_rviz"]),
        DeclareLaunchArgument("force_body_yaw_follow", default_value=d["force_body_yaw_follow"]),
        DeclareLaunchArgument("body_yaw_follow_clearance", default_value=d["body_yaw_follow_clearance"]),
        DeclareLaunchArgument("spin_enabled", default_value=d["spin_enabled"]),
        DeclareLaunchArgument("spin_rate", default_value=d["spin_rate"]),
        DeclareLaunchArgument("enable_lidar", default_value=d["enable_lidar"]),
        DeclareLaunchArgument("lidar_backend", default_value=d["lidar_backend"]),
        DeclareLaunchArgument("lidar_downsample", default_value=d["lidar_downsample"]),
        DeclareLaunchArgument("enable_tof", default_value=d["enable_tof"]),
        DeclareLaunchArgument("freeze_motion", default_value=d["freeze_motion"]),
        DeclareLaunchArgument("nav_start_delay_sec", default_value=d["nav_start_delay_sec"]),
        DeclareLaunchArgument("rog_map_start_delay_sec", default_value=d["rog_map_start_delay_sec"]),
        DeclareLaunchArgument("map_start_delay_sec", default_value=d["map_start_delay_sec"]),
        DeclareLaunchArgument("rviz_delay_sec", default_value=d["rviz_delay_sec"]),
        DeclareLaunchArgument("solver_mode", default_value=d["solver_mode"]),
        DeclareLaunchArgument("max_angular_z", default_value=d["max_angular_z"]),
        DeclareLaunchArgument("telemetry_sampling_window_cycles", default_value=d["telemetry_sampling_window_cycles"]),
        DeclareLaunchArgument("log_level", default_value=d["log_level"]),
        DeclareLaunchArgument("save_odom_layer_on_shutdown", default_value=d["save_odom_layer_on_shutdown"]),
        DeclareLaunchArgument("mapping_output_prefix", default_value=d["mapping_output_prefix"]),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([share, "launch", "rmuc_2025_mujoco.launch.py"])
            ),
            launch_arguments={
                "model_file": LaunchConfiguration("model_file"),
                "map_yaml_file": LaunchConfiguration("map_yaml_file"),
                "prior_pcd_file": LaunchConfiguration("prior_pcd_file"),
                "scene_navigation_profile": LaunchConfiguration("scene_navigation_profile"),
                "start_x": LaunchConfiguration("start_x"),
                "start_y": LaunchConfiguration("start_y"),
                "start_z": LaunchConfiguration("start_z"),
                "start_yaw": LaunchConfiguration("start_yaw"),
                "planning_grid_owner": LaunchConfiguration("planning_grid_owner"),
                "slam": LaunchConfiguration("slam"),
                "use_viewer": LaunchConfiguration("use_viewer"),
                "show_viewer": LaunchConfiguration("show_viewer"),
                "launch_physics": LaunchConfiguration("launch_physics"),
                "launch_navigation": LaunchConfiguration("launch_navigation"),
                "use_rviz": LaunchConfiguration("use_rviz"),
                "launch_mujoco_rviz": LaunchConfiguration("launch_mujoco_rviz"),
                "force_body_yaw_follow": LaunchConfiguration("force_body_yaw_follow"),
                "body_yaw_follow_clearance": LaunchConfiguration("body_yaw_follow_clearance"),
                "spin_enabled": LaunchConfiguration("spin_enabled"),
                "spin_rate": LaunchConfiguration("spin_rate"),
                "enable_lidar": LaunchConfiguration("enable_lidar"),
                "lidar_backend": LaunchConfiguration("lidar_backend"),
                "lidar_downsample": LaunchConfiguration("lidar_downsample"),
                "lidar_rate_hz": LaunchConfiguration("lidar_rate_hz"),
                "sim_rate_hz": LaunchConfiguration("sim_rate_hz"),
                "lio_lease_wall_scale": LaunchConfiguration("lio_lease_wall_scale"),
                **{name: LaunchConfiguration(name) for name in LIO_ARGUMENTS_RMUL_2027},
                "enable_tof": LaunchConfiguration("enable_tof"),
                "freeze_motion": LaunchConfiguration("freeze_motion"),
                "nav_start_delay_sec": LaunchConfiguration("nav_start_delay_sec"),
                "rog_map_start_delay_sec": LaunchConfiguration("rog_map_start_delay_sec"),
                "map_start_delay_sec": LaunchConfiguration("map_start_delay_sec"),
                "rviz_delay_sec": LaunchConfiguration("rviz_delay_sec"),
                "solver_mode": LaunchConfiguration("solver_mode"),
                "max_angular_z": LaunchConfiguration("max_angular_z"),
                "telemetry_sampling_window_cycles": LaunchConfiguration(
                    "telemetry_sampling_window_cycles"
                ),
                "log_level": LaunchConfiguration("log_level"),
                "save_odom_layer_on_shutdown": LaunchConfiguration("save_odom_layer_on_shutdown"),
                "mapping_output_prefix": LaunchConfiguration("mapping_output_prefix"),
            }.items(),
        )
    ])
