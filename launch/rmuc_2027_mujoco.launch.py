#!/usr/bin/env python3
"""2027 tunnel profile; all common navigation ownership stays in the 2025 launch."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    IfElseSubstitution,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.substitutions import FindPackageShare

from ats_mujoco_sim.launch_args import SIM_LIO_ARGUMENTS

# The 2027 scene uses the chassis-mounted Mid360 (report 2.1.8): front tilt 30
# degrees (the real vehicle) at the front (+x) edge midpoint, pulled in 0.05 m.  Older scenes keep
# the legacy gimbal pose.
LIO_ARGUMENTS_2027 = {**SIM_LIO_ARGUMENTS, "lidar_mount_preset": "chassis"}


def generate_launch_description() -> LaunchDescription:
    share = FindPackageShare("ats_mujoco_sim")
    bringup_share = FindPackageShare("ats_sentry_bringup")
    slam = LaunchConfiguration("slam")
    return LaunchDescription([
        DeclareLaunchArgument("model_file", default_value="rmuc_2027_swerve.xml"),
        DeclareLaunchArgument("map_yaml_file", default_value=PathJoinSubstitution(
            [bringup_share, "map", "rmuc_2027.yaml"])),
        DeclareLaunchArgument("prior_pcd_file", default_value=PathJoinSubstitution(
            [bringup_share, "pcd", "rmuc_2027.pcd"])),
        DeclareLaunchArgument("scene_navigation_profile", default_value=PathJoinSubstitution(
            [share, "config", "rmuc_2027_navigation.yaml"])),
        *[DeclareLaunchArgument(n, default_value=v) for n, v in LIO_ARGUMENTS_2027.items()],
        DeclareLaunchArgument("lio_lease_wall_scale", default_value="auto"),
        DeclareLaunchArgument("sim_rate_hz", default_value="auto"),
        DeclareLaunchArgument("lidar_rate_hz", default_value="auto"),
        DeclareLaunchArgument("start_x", default_value="3.37"),
        DeclareLaunchArgument("start_y", default_value="9.03"),
        # RMUC 2027 base start zone floor is z = 0.285 m; base_link sits 0.180 m above wheel contact.
        DeclareLaunchArgument("start_z", default_value="0.465"),
        DeclareLaunchArgument("start_yaw", default_value="0.0"),
        DeclareLaunchArgument("planning_grid_owner", default_value="rog_map"),
        DeclareLaunchArgument("slam", default_value="false"),
        DeclareLaunchArgument("use_viewer", default_value="false"),
        DeclareLaunchArgument("show_viewer", default_value="false"),
        DeclareLaunchArgument("viewer_rate_hz", default_value="15.0"),
        DeclareLaunchArgument("launch_physics", default_value="true"),
        DeclareLaunchArgument("launch_navigation", default_value="true"),
        DeclareLaunchArgument("use_rviz", default_value="false"),
        DeclareLaunchArgument("launch_mujoco_rviz", default_value="false"),
        DeclareLaunchArgument("force_body_yaw_follow", default_value="false"),
        DeclareLaunchArgument("body_yaw_follow_clearance", default_value="0.55"),
        DeclareLaunchArgument("spin_enabled", default_value="true"),
        DeclareLaunchArgument("spin_rate", default_value="3.0"),
        DeclareLaunchArgument("enable_lidar", default_value="true"),
        DeclareLaunchArgument("lidar_backend", default_value="cpu"),
        DeclareLaunchArgument("lidar_downsample", default_value="auto"),
        DeclareLaunchArgument("enable_tof", default_value="false"),
        DeclareLaunchArgument("freeze_motion", default_value="false"),
        DeclareLaunchArgument("nav_start_delay_sec", default_value="6.0"),
        DeclareLaunchArgument("rog_map_start_delay_sec", default_value="12.0"),
        DeclareLaunchArgument("map_start_delay_sec", default_value="1.0"),
        DeclareLaunchArgument("rviz_delay_sec", default_value="1.0"),
        DeclareLaunchArgument("solver_mode", default_value="qp"),
        # Bridge command cap, not a change to physical wheel/steer limits.
        DeclareLaunchArgument("max_angular_z", default_value="3.0"),
        DeclareLaunchArgument("telemetry_sampling_window_cycles", default_value="0"),
        DeclareLaunchArgument("log_level", default_value="info"),
        DeclareLaunchArgument("save_odom_layer_on_shutdown", default_value="false"),
        DeclareLaunchArgument("mapping_output_prefix", default_value=""),
        # Center the 40x30 m persistent odom_layer on the 28x16 m RMUC 2027 arena for any start pose
        # (e.g. base (3.37, 9.03) -> (-9.35, -16.05), covering [-9.35, +30.65] x [-16.05, +13.95]).
        DeclareLaunchArgument("odom_layer_origin_x", default_value="auto"),
        DeclareLaunchArgument("odom_layer_origin_y", default_value="auto"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([share, "launch", "rmuc_2025_mujoco.launch.py"])
            ),
            launch_arguments={
                "model_file": LaunchConfiguration("model_file"),
                "map_yaml_file": IfElseSubstitution(slam, "", LaunchConfiguration("map_yaml_file")),
                "prior_pcd_file": IfElseSubstitution(slam, "", LaunchConfiguration("prior_pcd_file")),
                "scene_navigation_profile": LaunchConfiguration("scene_navigation_profile"),
                "start_x": LaunchConfiguration("start_x"),
                "start_y": LaunchConfiguration("start_y"),
                "start_z": LaunchConfiguration("start_z"),
                "start_yaw": LaunchConfiguration("start_yaw"),
                "planning_grid_owner": LaunchConfiguration("planning_grid_owner"),
                "slam": LaunchConfiguration("slam"),
                "use_viewer": LaunchConfiguration("use_viewer"),
                "show_viewer": LaunchConfiguration("show_viewer"),
                "viewer_rate_hz": LaunchConfiguration("viewer_rate_hz"),
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
                **{name: LaunchConfiguration(name) for name in LIO_ARGUMENTS_2027},
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
                "odom_layer_origin_x": PythonExpression([
                    "f'{round((-6.0 - float(\"", LaunchConfiguration("start_x"),
                    "\")) * 20.0) / 20.0:.2f}' if '", LaunchConfiguration("odom_layer_origin_x"),
                    "' == 'auto' else '", LaunchConfiguration("odom_layer_origin_x"), "'",
                ]),
                "odom_layer_origin_y": PythonExpression([
                    "f'{round((-7.0 - float(\"", LaunchConfiguration("start_y"),
                    "\")) * 20.0) / 20.0:.2f}' if '", LaunchConfiguration("odom_layer_origin_y"),
                    "' == 'auto' else '", LaunchConfiguration("odom_layer_origin_y"), "'",
                ]),
            }.items(),
        )
    ])
