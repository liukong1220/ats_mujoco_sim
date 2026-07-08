#!/usr/bin/env python3
"""Launch MuJoCo, Nav2, trajectory optimizer, command bridge, and RViz2."""

from __future__ import annotations

import uuid
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import GroupAction
from launch.actions import IncludeLaunchDescription
from launch.actions import TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch.substitutions import PythonExpression
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterFile
from launch_ros.substitutions import FindPackageShare
from nav2_common.launch import ReplaceString
from nav2_common.launch import RewrittenYaml


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _quoted(value):
    return ["'", value, "'"]


def _map_directory_substitution(
    output_root: LaunchConfiguration,
    map_name: LaunchConfiguration,
    seed: LaunchConfiguration,
):
    return PythonExpression(
        [
            "__import__('ats_mujoco_sim.terrain_assets', "
            "fromlist=['map_output_dir']).map_output_dir(",
            *_quoted(output_root),
            ", ",
            *_quoted(map_name),
            ", int(",
            *_quoted(seed),
            ")).as_posix()",
        ]
    )


def generate_launch_description() -> LaunchDescription:
    output_root = LaunchConfiguration("output_root")
    map_name = LaunchConfiguration("map_name")
    seed = LaunchConfiguration("seed")
    map_directory = _map_directory_substitution(output_root, map_name, seed)
    map_yaml = PathJoinSubstitution([map_directory, "map.yaml"])

    params_file = LaunchConfiguration("params_file")
    nav_params = ReplaceString(
        source_file=params_file,
        replacements={
            "<robot_namespace>": "",
            "base_frame: \"base_footprint\"": "base_frame: \"gimbal_yaw_odom\"",
            "    robot_base_frame: gimbal_yaw_fake": "    robot_base_frame: gimbal_yaw_odom",
            "    robot_base_frame: \"gimbal_yaw_fake\"": "    robot_base_frame: \"gimbal_yaw_odom\"",
            "      robot_base_frame: gimbal_yaw_fake": "      robot_base_frame: gimbal_yaw_odom",
            "      robot_base_frame: \"gimbal_yaw_fake\"": "      robot_base_frame: \"gimbal_yaw_odom\"",
        },
    )
    configured_map_params = ParameterFile(
        RewrittenYaml(
            source_file=nav_params,
            root_key="",
            param_rewrites={
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "yaml_filename": map_yaml,
            },
            convert_types=True,
        ),
        allow_substs=True,
    )

    mujoco_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            (_package_root() / "launch" / "planner_mujoco.launch.py").as_posix()
        ),
        launch_arguments={
            "map_config": LaunchConfiguration("map_config"),
            "output_root": output_root,
            "map_name": map_name,
            "seed": seed,
            "resolution": LaunchConfiguration("resolution"),
            "map_ready_token": LaunchConfiguration("map_ready_token"),
            "map_wait_timeout_sec": LaunchConfiguration("map_wait_timeout_sec"),
            "odom_topic": "/localization",
            "lidar_odometry_topic": "/lidar_odometry",
            "registered_scan_topic": "/registered_scan",
            "registered_scan_frame_id": "odom",
            "pose_cmd_topic": LaunchConfiguration("pose_cmd_topic"),
            "start_x": LaunchConfiguration("start_x"),
            "start_y": LaunchConfiguration("start_y"),
            "start_z": LaunchConfiguration("start_z"),
            "start_yaw": LaunchConfiguration("start_yaw"),
            "use_viewer": LaunchConfiguration("use_viewer"),
            "show_viewer": LaunchConfiguration("show_viewer"),
            "launch_mujoco_rviz": "false",
            "sim_rate_hz": LaunchConfiguration("sim_rate_hz"),
            "feedback_rate_hz": LaunchConfiguration("feedback_rate_hz"),
            "truth_rate_hz": LaunchConfiguration("truth_rate_hz"),
            "command_timeout": LaunchConfiguration("command_timeout"),
            "enable_lidar": LaunchConfiguration("enable_lidar"),
            "lidar_backend": LaunchConfiguration("lidar_backend"),
            "lidar_model": LaunchConfiguration("lidar_model"),
            "lidar_downsample": LaunchConfiguration("lidar_downsample"),
            "lidar_rate_hz": LaunchConfiguration("lidar_rate_hz"),
            "lidar_rate_clock": LaunchConfiguration("lidar_rate_clock"),
            "lidar_state_rate_hz": LaunchConfiguration("lidar_state_rate_hz"),
            "lidar_topic": "/local_pointcloud",
            "lidar_frame_id": "front_mid360",
            "enable_tof": LaunchConfiguration("enable_tof"),
            "tof_backend": LaunchConfiguration("tof_backend"),
            "tof_rate_hz": LaunchConfiguration("tof_rate_hz"),
            "merged_tof_topic": "/perception/tof/points_merged",
        }.items(),
    )
    mujoco_group = GroupAction(
        scoped=True,
        actions=[mujoco_launch],
    )

    map_server = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[configured_map_params],
        arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
    )
    map_lifecycle = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_mujoco_map",
        output="screen",
        parameters=[
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
            {"autostart": LaunchConfiguration("autostart")},
            {"node_names": ["map_server"]},
        ],
        arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
    )
    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("ats_nav_bringup"),
                "launch",
                "navigation_launch.py",
            ])
        ]),
        condition=IfCondition(LaunchConfiguration("launch_nav2")),
        launch_arguments={
            "namespace": "",
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "autostart": LaunchConfiguration("autostart"),
            "params_file": nav_params,
            "use_composition": "False",
            "use_respawn": LaunchConfiguration("use_respawn"),
            "launch_trajectory_optimizer": LaunchConfiguration(
                "launch_trajectory_optimizer"
            ),
            "launch_chassis_vel_transform": "False",
            "log_level": LaunchConfiguration("log_level"),
        }.items(),
    )
    twist_bridge = Node(
        condition=IfCondition(LaunchConfiguration("launch_twist_bridge")),
        package="ats_mujoco_sim",
        executable="twist_to_motion_ctrl",
        name="twist_to_motion_ctrl",
        output="screen",
        parameters=[{
            "input_topic": LaunchConfiguration("cmd_vel_topic"),
            "output_topic": "/motion_control",
            "max_linear_x": LaunchConfiguration("max_linear_x"),
            "max_linear_y": LaunchConfiguration("max_linear_y"),
            "max_angular_z": LaunchConfiguration("max_angular_z"),
        }],
    )
    map_group = TimerAction(
        period=LaunchConfiguration("map_start_delay_sec"),
        actions=[
            GroupAction([
                map_server,
                map_lifecycle,
            ])
        ],
    )
    nav_group = TimerAction(
        period=LaunchConfiguration("nav_start_delay_sec"),
        actions=[
            GroupAction([
                navigation_launch,
                twist_bridge,
            ])
        ],
    )
    rviz = TimerAction(
        period=LaunchConfiguration("rviz_delay_sec"),
        actions=[
            Node(
                condition=IfCondition(LaunchConfiguration("use_rviz")),
                package="rviz2",
                executable="rviz2",
                name="mujoco_navigation_rviz2",
                output="screen",
                arguments=["-d", LaunchConfiguration("rviz_config_file")],
            )
        ],
    )

    declarations = [
        DeclareLaunchArgument(
            "map_config",
            default_value=(_package_root() / "config" / "random_map.yaml").as_posix(),
        ),
        DeclareLaunchArgument("output_root", default_value="/tmp/ats_mujoco_sim_maps"),
        DeclareLaunchArgument("map_name", default_value=""),
        DeclareLaunchArgument("seed", default_value="-1"),
        DeclareLaunchArgument("resolution", default_value="0.03"),
        DeclareLaunchArgument("map_ready_token", default_value=str(uuid.uuid4())),
        DeclareLaunchArgument("map_wait_timeout_sec", default_value="30.0"),
        DeclareLaunchArgument("pose_cmd_topic", default_value="/simulation/PoseSub"),
        DeclareLaunchArgument("start_x", default_value="0.0"),
        DeclareLaunchArgument("start_y", default_value="0.0"),
        DeclareLaunchArgument("start_z", default_value="0.18"),
        DeclareLaunchArgument("start_yaw", default_value="0.0"),
        DeclareLaunchArgument("use_viewer", default_value="true"),
        DeclareLaunchArgument("show_viewer", default_value="true"),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument(
            "launch_mujoco_rviz",
            default_value="false",
            description=(
                "Whether to also start the lightweight MuJoCo-only RViz. "
                "Keep false when using the navigation RViz."
            ),
        ),
        DeclareLaunchArgument("map_start_delay_sec", default_value="2.0"),
        DeclareLaunchArgument("nav_start_delay_sec", default_value="6.0"),
        DeclareLaunchArgument("rviz_delay_sec", default_value="10.0"),
        DeclareLaunchArgument(
            "rviz_config_file",
            default_value=PathJoinSubstitution([
                FindPackageShare("ats_mujoco_sim"),
                "rviz",
                "mujoco_navigation.rviz",
            ]),
        ),
        DeclareLaunchArgument("sim_rate_hz", default_value="300.0"),
        DeclareLaunchArgument("feedback_rate_hz", default_value="10.0"),
        DeclareLaunchArgument("truth_rate_hz", default_value="10.0"),
        DeclareLaunchArgument("command_timeout", default_value="0.5"),
        DeclareLaunchArgument("enable_lidar", default_value="true"),
        DeclareLaunchArgument("lidar_backend", default_value="cpu"),
        DeclareLaunchArgument("lidar_model", default_value="mid360"),
        DeclareLaunchArgument("lidar_downsample", default_value="24"),
        DeclareLaunchArgument("lidar_rate_hz", default_value="10.0"),
        DeclareLaunchArgument("lidar_rate_clock", default_value="wall"),
        DeclareLaunchArgument("lidar_state_rate_hz", default_value="30.0"),
        DeclareLaunchArgument("enable_tof", default_value="false"),
        DeclareLaunchArgument("tof_backend", default_value="cpu"),
        DeclareLaunchArgument("tof_rate_hz", default_value="10.0"),
        DeclareLaunchArgument("launch_nav2", default_value="true"),
        DeclareLaunchArgument("launch_trajectory_optimizer", default_value="false"),
        DeclareLaunchArgument("launch_twist_bridge", default_value="true"),
        DeclareLaunchArgument("cmd_vel_topic", default_value="cmd_vel_gimbal_yaw_odom"),
        DeclareLaunchArgument("max_linear_x", default_value="3.0"),
        DeclareLaunchArgument("max_linear_y", default_value="3.0"),
        DeclareLaunchArgument("max_angular_z", default_value="6.0"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("autostart", default_value="true"),
        DeclareLaunchArgument("use_respawn", default_value="false"),
        DeclareLaunchArgument(
            "params_file",
            default_value=PathJoinSubstitution([
                FindPackageShare("ats_nav_bringup"),
                "config",
                "simulation",
                "nav2_params.yaml",
            ]),
        ),
        DeclareLaunchArgument("log_level", default_value="info"),
    ]

    ld = LaunchDescription()
    for declaration in declarations:
        ld.add_action(declaration)
    ld.add_action(mujoco_group)
    ld.add_action(map_group)
    ld.add_action(nav_group)
    ld.add_action(rviz)
    return ld
