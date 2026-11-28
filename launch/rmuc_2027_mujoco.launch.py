#!/usr/bin/env python3
"""2027 tunnel profile; all common navigation ownership stays in the 2025 launch."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    IfElseSubstitution,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from ats_mujoco_sim.launch_args import load_lio_arguments, load_mujoco_launch_defaults

# All scene, auto-mapping, and localization defaults are parameterized in
# config/mujoco_launch_defaults.yaml (profile: rmuc_2027).
DEFAULTS_2027 = load_mujoco_launch_defaults("rmuc_2027")
LIO_ARGUMENTS_2027 = load_lio_arguments("rmuc_2027")


def generate_launch_description() -> LaunchDescription:
    share = FindPackageShare("ats_mujoco_sim")
    bringup_share = FindPackageShare("ats_sentry_bringup")
    slam = LaunchConfiguration("slam")

    def mapping_cruise_after_ready(context):
        lc = context.launch_configurations
        if lc.get("launch_mapping_cruise", "false").lower() not in ("true", "1"):
            return []
        stable_sec = max(0.0, float(lc.get("mapping_cruise_stable_sec", "1.5")))
        client_delay = max(0.0, float(lc.get("mapping_cruise_start_delay_sec", "1.0")))
        wait_cmd = [
            "ros2", "run", "ats_mujoco_sim", "wait_nav_ready",
            "--stable-sec", f"{stable_sec:.2f}",
        ]
        if lc.get("launch_physics", "true").lower() in ("true", "1"):
            wait_cmd.append("--require-sim-truth")
        wait_nav = ExecuteProcess(cmd=wait_cmd, output="screen")
        behavior_server = Node(
            package="ats_sentry_behavior",
            executable="ats_sentry_behavior_server",
            name="ats_sentry_behavior_server",
            output="screen",
            parameters=[
                PathJoinSubstitution([bringup_share, "params", "node_params.yaml"]),
                {
                    "use_sim_time": True,
                    "decision.rmuc.csv_waypoints_file": lc["mapping_cruise_csv"],
                    "decision.rmuc.patrol_csv_file": lc["mapping_cruise_csv"],
                },
            ],
        )
        behavior_client = Node(
            package="ats_sentry_behavior",
            executable="ats_sentry_behavior_client",
            name="ats_sentry_behavior_client",
            output="screen",
            parameters=[
                PathJoinSubstitution([bringup_share, "params", "node_params.yaml"]),
                {
                    "use_sim_time": True,
                    "target_tree": lc["mapping_cruise_tree"],
                },
            ],
        )

        def _on_nav_ready(event, _ctx):
            if event.returncode != 0:
                return []
            # 先起 behavior_server 完成插件与 action client 初始化，错峰 client_delay 后再起 client 触发执行。
            return [
                behavior_server,
                TimerAction(period=client_delay, actions=[behavior_client]),
            ]

        return [
            wait_nav,
            RegisterEventHandler(OnProcessExit(target_action=wait_nav, on_exit=_on_nav_ready)),
        ]

    d = DEFAULTS_2027
    return LaunchDescription([
        DeclareLaunchArgument("model_file", default_value=d["model_file"]),
        DeclareLaunchArgument("map_yaml_file", default_value=PathJoinSubstitution(
            [bringup_share, "map", d["map_yaml_name"]])),
        DeclareLaunchArgument("prior_pcd_file", default_value=PathJoinSubstitution(
            [bringup_share, "pcd", d["prior_pcd_name"]])),
        DeclareLaunchArgument("scene_navigation_profile", default_value=PathJoinSubstitution(
            [share, "config", d["scene_navigation_profile_name"]])),
        *[DeclareLaunchArgument(n, default_value=v) for n, v in LIO_ARGUMENTS_2027.items()],
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
        DeclareLaunchArgument("viewer_rate_hz", default_value=d["viewer_rate_hz"]),
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
        DeclareLaunchArgument("odom_layer_origin_x", default_value=d["odom_layer_origin_x"]),
        DeclareLaunchArgument("odom_layer_origin_y", default_value=d["odom_layer_origin_y"]),
        DeclareLaunchArgument("launch_mapping_cruise", default_value=d["launch_mapping_cruise"]),
        DeclareLaunchArgument("mapping_cruise_tree", default_value=d["mapping_cruise_tree"]),
        DeclareLaunchArgument(
            "mapping_cruise_csv",
            default_value=PathJoinSubstitution([share, "config", d["mapping_cruise_csv_name"]]),
        ),
        DeclareLaunchArgument("mapping_cruise_stable_sec", default_value=d["mapping_cruise_stable_sec"]),
        DeclareLaunchArgument("mapping_cruise_start_delay_sec", default_value=d["mapping_cruise_start_delay_sec"]),
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
        ),
        OpaqueFunction(function=mapping_cruise_after_ready),
    ])
