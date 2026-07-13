#!/usr/bin/env python3
"""Launch the MuJoCo swerve chassis on the RMUC 2026 mesh field."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch.substitutions import PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from nav2_common.launch import ReplaceString


def generate_launch_description() -> LaunchDescription:
    nav_params = ReplaceString(
        source_file=LaunchConfiguration("params_file"),
        replacements={
            "<robot_namespace>": "",
            "base_frame: \"base_footprint\"": "base_frame: \"gimbal_yaw_odom\"",
            "base_frame: base_footprint": "base_frame: gimbal_yaw_odom",
            "target_frame: base_footprint": "target_frame: gimbal_yaw_odom",
            "    robot_base_frame: gimbal_yaw_fake": "    robot_base_frame: gimbal_yaw_odom",
            '    robot_base_frame: "gimbal_yaw_fake"': '    robot_base_frame: "gimbal_yaw_odom"',
            "      robot_base_frame: gimbal_yaw_fake": "      robot_base_frame: gimbal_yaw_odom",
            '      robot_base_frame: "gimbal_yaw_fake"': '      robot_base_frame: "gimbal_yaw_odom"',
            "    odom_topic: odometry": "    odom_topic: /localization",
            '    odom_topic: "odometry"': '    odom_topic: "/localization"',
            "      odom_topic: odometry": "      odom_topic: /localization",
            '      odom_topic: "odometry"': '      odom_topic: "/localization"',
            "    local_plan_topic: \"transformed_global_plan\"": (
                "    local_plan_topic: \"local_plan\""
            ),
            (
                '      plugins: ["static_layer", "intensity_voxel_layer", '
                '"inflation_layer"]'
            ): '      plugins: ["static_layer", "inflation_layer"]',
            "      width: 40": "      width: 30",
            "      height: 25": "      height: 17",
            "      origin_x: -3.58": "      origin_x: -14.575",
            "      origin_y: -9.44": "      origin_y: -8.025",
            "      required_movement_radius: 0.25": "      required_movement_radius: 0.10",
            "      movement_time_allowance: 6.0": "      movement_time_allowance: 12.0",
            "      xy_goal_tolerance: 0.20": "      xy_goal_tolerance: 0.35",
            "        self_filter_radius: 0.32": "        self_filter_radius: 0.40",
            "        self_filter_radius: 0.43": "        self_filter_radius: 0.40",
            "        inflation_radius: 0.50": "        inflation_radius: 0.60",
            "        inflation_radius: 0.55": "        inflation_radius: 0.65",
            "        collision_margin_distance: 0.08": (
                "        collision_margin_distance: 0.10"
            ),
            "        collision_margin_distance: 0.12": (
                "        collision_margin_distance: 0.10"
            ),
            "      obstacle_safe_distance: 0.30": (
                "      obstacle_safe_distance: 0.35"
            ),
            (
                '      footprint: "[[0.25, 0.25], [0.25, -0.25], '
                '[-0.25, -0.25], [-0.25, 0.25]]"'
            ): (
                '      footprint: "[[0.30, 0.25], [0.30, -0.25], '
                '[-0.30, -0.25], [-0.30, 0.25]]"'
            ),
            (
                '      footprint: "[[0.20, 0.20], [0.20, -0.20], '
                '[-0.20, -0.20], [-0.20, 0.20]]"'
            ): (
                '      footprint: "[[0.30, 0.25], [0.30, -0.25], '
                '[-0.30, -0.25], [-0.30, 0.25]]"'
            ),
            (
                '      footprint: "[[0.23, 0.23], [0.23, -0.23], '
                '[-0.23, -0.23], [-0.23, 0.23]]"'
            ): (
                '      footprint: "[[0.30, 0.25], [0.30, -0.25], '
                '[-0.30, -0.25], [-0.30, 0.25]]"'
            ),
            (
                '      footprint: "[[0.30, 0.30], [0.30, -0.30], '
                '[-0.30, -0.30], [-0.30, 0.30]]"'
            ): (
                '      footprint: "[[0.30, 0.25], [0.30, -0.25], '
                '[-0.30, -0.25], [-0.30, 0.25]]"'
            ),
            "    robot_footprint_radius: 0.38": "    robot_footprint_radius: 0.40",
            "      robot_footprint_radius: 0.38": "      robot_footprint_radius: 0.40",
            "    robot_footprint_radius: 0.31": "    robot_footprint_radius: 0.40",
            "      robot_footprint_radius: 0.31": "      robot_footprint_radius: 0.40",
            "    robot_footprint_radius: 0.43": "    robot_footprint_radius: 0.40",
            "      robot_footprint_radius: 0.43": "      robot_footprint_radius: 0.40",
        },
    )

    map_server = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[
            {"yaml_filename": LaunchConfiguration("map_yaml_file")},
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
    )
    map_lifecycle = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_rmuc_2026_map",
        output="screen",
        parameters=[
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
            {"autostart": LaunchConfiguration("autostart")},
            {"node_names": ["map_server"]},
        ],
    )
    map_group = TimerAction(
        period=LaunchConfiguration("map_start_delay_sec"),
        actions=[map_server, map_lifecycle],
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
            "launch_fake_vel_transform": PythonExpression(
                ["'", LaunchConfiguration("launch_swerve_mpc"), "'.lower() != 'true'"]
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
            "input_topic": PythonExpression([
                "'", LaunchConfiguration("mpc_cmd_vel_topic"),
                "' if '", LaunchConfiguration("launch_swerve_mpc"),
                "'.lower() == 'true' else '", LaunchConfiguration("cmd_vel_topic"), "'",
            ]),
            "output_topic": "/motion_control",
            "max_linear_x": LaunchConfiguration("max_linear_x"),
            "max_linear_y": LaunchConfiguration("max_linear_y"),
            "max_angular_z": LaunchConfiguration("max_angular_z"),
        }],
    )
    minco_planner = Node(
        condition=IfCondition(LaunchConfiguration("launch_swerve_mpc")),
        package="minco_planner",
        executable="minco_planner_node",
        name="minco_planner",
        output="screen",
        parameters=[
            LaunchConfiguration("minco_params_file"),
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
    )
    swerve_mpc = Node(
        condition=IfCondition(LaunchConfiguration("launch_swerve_mpc")),
        package="ats_swerve_mpc",
        executable="ats_swerve_mpc_node",
        name="ats_swerve_mpc",
        output="screen",
        parameters=[
            LaunchConfiguration("mpc_params_file"),
            {
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "command_topic": LaunchConfiguration("mpc_cmd_vel_topic"),
            },
        ],
    )
    nav_group = TimerAction(
        period=LaunchConfiguration("nav_start_delay_sec"),
        actions=[navigation_launch, minco_planner, swerve_mpc, twist_bridge],
    )

    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("ats_mujoco_sim"),
                "launch",
                "ats_mujoco_sim.launch.py",
            ])
        ]),
        launch_arguments={
            "model_path": PathJoinSubstitution([
                FindPackageShare("ats_mujoco_sim"),
                "models",
                "rmuc_2026_swerve.xml",
            ]),
            "scene_file": PathJoinSubstitution([
                FindPackageShare("ats_mujoco_sim"),
                "models",
                "rmuc_2026_swerve.xml",
            ]),
            "map_ready_file": "",
            "map_wait_timeout_sec": "0.0",
            "start_x": LaunchConfiguration("start_x"),
            "start_y": LaunchConfiguration("start_y"),
            "start_z": LaunchConfiguration("start_z"),
            "start_yaw": LaunchConfiguration("start_yaw"),
            "use_viewer": LaunchConfiguration("use_viewer"),
            "show_viewer": LaunchConfiguration("show_viewer"),
            "launch_mujoco_rviz": LaunchConfiguration("launch_mujoco_rviz"),
            "rviz_config_file": LaunchConfiguration("rviz_config_file"),
            "rviz_delay_sec": LaunchConfiguration("rviz_delay_sec"),
            "sim_rate_hz": LaunchConfiguration("sim_rate_hz"),
            "feedback_rate_hz": LaunchConfiguration("feedback_rate_hz"),
            "truth_rate_hz": LaunchConfiguration("truth_rate_hz"),
            "command_timeout": LaunchConfiguration("command_timeout"),
            "enable_lidar": LaunchConfiguration("enable_lidar"),
            "lidar_backend": LaunchConfiguration("lidar_backend"),
            "lidar_model": "mid360",
            "lidar_downsample": LaunchConfiguration("lidar_downsample"),
            "lidar_rate_hz": LaunchConfiguration("lidar_rate_hz"),
            "lidar_rate_clock": LaunchConfiguration("lidar_rate_clock"),
            "lidar_state_rate_hz": LaunchConfiguration("lidar_state_rate_hz"),
            "lidar_topic": "/local_pointcloud",
            "lidar_frame_id": "front_mid360",
            "registered_scan_topic": "/registered_scan",
            "registered_scan_frame_id": "odom",
            "enable_tof": LaunchConfiguration("enable_tof"),
            "tof_backend": LaunchConfiguration("tof_backend"),
            "tof_footprint_length": "0.60",
            "tof_footprint_width": "0.50",
            "tof_footprint_z_min": "-0.10",
            "tof_footprint_z_max": "0.13",
            "tof_rate_hz": LaunchConfiguration("tof_rate_hz"),
            "merged_tof_topic": "/perception/tof/points_merged",
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument("start_x", default_value="-10.66"),
        DeclareLaunchArgument("start_y", default_value="1.47"),
        DeclareLaunchArgument("start_z", default_value="0.42"),
        DeclareLaunchArgument("start_yaw", default_value="0.0"),
        DeclareLaunchArgument("use_viewer", default_value="true"),
        DeclareLaunchArgument("show_viewer", default_value="true"),
        DeclareLaunchArgument("launch_mujoco_rviz", default_value="true"),
        DeclareLaunchArgument(
            "rviz_config_file",
            default_value=PathJoinSubstitution([
                FindPackageShare("ats_mujoco_sim"),
                "rviz",
                "mujoco_navigation.rviz",
            ]),
        ),
        DeclareLaunchArgument(
            "map_yaml_file",
            default_value=PathJoinSubstitution([
                FindPackageShare("ats_sentry_bringup"),
                "map",
                "rmuc_2026.yaml",
            ]),
        ),
        DeclareLaunchArgument("map_start_delay_sec", default_value="1.0"),
        DeclareLaunchArgument("nav_start_delay_sec", default_value="6.0"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("autostart", default_value="true"),
        DeclareLaunchArgument("use_respawn", default_value="false"),
        DeclareLaunchArgument("rviz_delay_sec", default_value="4.0"),
        DeclareLaunchArgument("sim_rate_hz", default_value="300.0"),
        DeclareLaunchArgument("feedback_rate_hz", default_value="10.0"),
        DeclareLaunchArgument("truth_rate_hz", default_value="10.0"),
        DeclareLaunchArgument("command_timeout", default_value="0.5"),
        DeclareLaunchArgument("enable_lidar", default_value="true"),
        DeclareLaunchArgument("lidar_backend", default_value="cpu"),
        DeclareLaunchArgument("lidar_downsample", default_value="24"),
        DeclareLaunchArgument("lidar_rate_hz", default_value="10.0"),
        DeclareLaunchArgument("lidar_rate_clock", default_value="wall"),
        DeclareLaunchArgument("lidar_state_rate_hz", default_value="30.0"),
        DeclareLaunchArgument("enable_tof", default_value="false"),
        DeclareLaunchArgument("tof_backend", default_value="cpu"),
        DeclareLaunchArgument("tof_rate_hz", default_value="10.0"),
        DeclareLaunchArgument("launch_nav2", default_value="true"),
        DeclareLaunchArgument(
            "launch_trajectory_optimizer",
            default_value="true",
            description="Whether to start the RC-ESDF local elastic path optimizer.",
        ),
        DeclareLaunchArgument("launch_twist_bridge", default_value="true"),
        DeclareLaunchArgument(
            "launch_swerve_mpc",
            default_value="false",
            description="Use JPS/MINCO and the holonomic SE2 MPC instead of Nav2 velocity output.",
        ),
        DeclareLaunchArgument("cmd_vel_topic", default_value="cmd_vel_gimbal_yaw_odom"),
        DeclareLaunchArgument("mpc_cmd_vel_topic", default_value="/cmd_vel_mpc"),
        DeclareLaunchArgument(
            "minco_params_file",
            default_value=PathJoinSubstitution([
                FindPackageShare("minco_planner"), "config", "minco_planner.yaml",
            ]),
        ),
        DeclareLaunchArgument(
            "mpc_params_file",
            default_value=PathJoinSubstitution([
                FindPackageShare("ats_swerve_mpc"), "config", "ats_swerve_mpc.yaml",
            ]),
        ),
        DeclareLaunchArgument("max_linear_x", default_value="1.0"),
        DeclareLaunchArgument("max_linear_y", default_value="1.0"),
        DeclareLaunchArgument("max_angular_z", default_value="2.0"),
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
        map_group,
        nav_group,
        sim_launch,
    ])
