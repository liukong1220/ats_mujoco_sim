#!/usr/bin/env python3
"""Launch the RMUC 2026 MuJoCo field with the ATS navigation chain."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    use_sim_time = LaunchConfiguration("use_sim_time")
    planning_grid_owner = LaunchConfiguration("planning_grid_owner")

    static_map = Node(
        package="ats_mujoco_sim",
        executable="static_map_publisher",
        name="static_map_publisher",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "map_yaml_file": LaunchConfiguration("map_yaml_file"),
            "map_topic": "/map",
            "frame_id": "map",
        }],
    )
    terrain = Node(
        package="terrain_analysis",
        executable="terrainAnalysis",
        name="terrain_analysis",
        output="screen",
        parameters=[LaunchConfiguration("params_file"), {"use_sim_time": use_sim_time}],
    )
    terrain_ext = Node(
        package="terrain_analysis_ext",
        executable="terrainAnalysisExt",
        name="terrain_analysis_ext",
        output="screen",
        parameters=[LaunchConfiguration("params_file"), {"use_sim_time": use_sim_time}],
    )
    localization_fusion = Node(
        package="small_gicp_relocalization",
        executable="localization_fusion_node",
        name="localization_fusion",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "odom_topic": "/odometry",
            "localization_topic": "/localization",
            "observation_topic": "/relocalization_observation",
            "status_topic": "/localization/status",
            "map_frame": "map",
            "odom_frame": "odom",
            "robot_base_frame": "gimbal_yaw_odom",
            "publish_tf": True,
            "allow_initial_identity": True,
            "odom_timeout_s": 0.5,
            "observation_timeout_s": 3.0,
            "observation_lost_timeout_s": 10.0,
        }],
    )
    rog_map = Node(
        package="ats_rog_map",
        executable="ats_rog_map_node",
        name="ats_rog_map",
        output="screen",
        parameters=[LaunchConfiguration("params_file"), {"use_sim_time": use_sim_time}],
    )
    rog_map_adapter = Node(
        package="ats_rog_map_adapter",
        executable="ats_rog_map_adapter_node",
        name="ats_rog_map_adapter",
        output="screen",
        condition=IfCondition(PythonExpression(["'", planning_grid_owner, "' == 'rog_map'"])),
        parameters=[LaunchConfiguration("params_file"), {
            "use_sim_time": use_sim_time,
            "planning_grid_owner": planning_grid_owner,
        }],
    )
    goal_manager = Node(
        package="ats_goal_manager",
        executable="ats_goal_manager_node",
        name="ats_goal_manager",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            {"use_sim_time": use_sim_time, "require_localization_status": True},
        ],
    )
    minco = Node(
        package="minco_planner",
        executable="minco_planner_node",
        name="minco_planner",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            {
                "use_sim_time": use_sim_time,
                "force_body_yaw_follow": LaunchConfiguration("force_body_yaw_follow"),
                "body_yaw_follow_clearance": LaunchConfiguration("body_yaw_follow_clearance"),
            },
        ],
    )
    mpc = Node(
        package="ats_swerve_mpc",
        executable="ats_swerve_mpc_node",
        name="ats_swerve_mpc",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            {
                "use_sim_time": use_sim_time,
                "command_topic": "/cmd_vel_mpc",
                "execution_command_topic": "/planner/execution_command",
                "require_localization_status": True,
            },
        ],
    )
    twist_bridge = Node(
        package="ats_mujoco_sim",
        executable="twist_to_motion_ctrl",
        name="twist_to_motion_ctrl",
        output="screen",
        parameters=[{
            "input_topic": "/cmd_vel_mpc",
            "output_topic": "/motion_control",
            "max_linear_x": LaunchConfiguration("max_linear_x"),
            "max_linear_y": LaunchConfiguration("max_linear_y"),
            "max_angular_z": LaunchConfiguration("max_angular_z"),
        }],
    )
    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([FindPackageShare("ats_mujoco_sim"), "launch", "ats_mujoco_sim.launch.py"])
        ]),
        launch_arguments={
            "model_path": PathJoinSubstitution([
                FindPackageShare("ats_mujoco_sim"), "models", "rmuc_2026_swerve.xml"
            ]),
            "scene_file": PathJoinSubstitution([
                FindPackageShare("ats_mujoco_sim"), "models", "rmuc_2026_swerve.xml"
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
            "odom_topic": "/odometry",
            "publish_map_to_odom_tf": "false",
            "enable_tof": LaunchConfiguration("enable_tof"),
            "tof_backend": LaunchConfiguration("tof_backend"),
            "tof_rate_hz": LaunchConfiguration("tof_rate_hz"),
            "merged_tof_topic": "/perception/tof/points_merged",
        }.items(),
    )
    rviz = TimerAction(
        period=LaunchConfiguration("rviz_delay_sec"),
        actions=[Node(
            condition=IfCondition(LaunchConfiguration("use_rviz")),
            package="rviz2",
            executable="rviz2",
            name="mujoco_navigation_rviz2",
            output="screen",
            arguments=["-d", LaunchConfiguration("rviz_config_file")],
        )],
    )

    return LaunchDescription([
        DeclareLaunchArgument("start_x", default_value="-10.66"),
        DeclareLaunchArgument("start_y", default_value="1.47"),
        DeclareLaunchArgument("start_z", default_value="0.42"),
        DeclareLaunchArgument("start_yaw", default_value="0.0"),
        DeclareLaunchArgument("use_viewer", default_value="true"),
        DeclareLaunchArgument("show_viewer", default_value="true"),
        DeclareLaunchArgument("use_rviz", default_value="false"),
        DeclareLaunchArgument("launch_mujoco_rviz", default_value="false"),
        DeclareLaunchArgument(
            "rviz_config_file",
            default_value=PathJoinSubstitution([
                FindPackageShare("ats_mujoco_sim"), "rviz", "mujoco_navigation.rviz"
            ]),
        ),
        DeclareLaunchArgument(
            "map_yaml_file",
            default_value=PathJoinSubstitution([
                FindPackageShare("ats_sentry_bringup"), "map", "rmuc_2026.yaml"
            ]),
        ),
        DeclareLaunchArgument(
            "params_file",
            default_value=PathJoinSubstitution([
                FindPackageShare("ats_sentry_bringup"), "params", "node_params.yaml"
            ]),
        ),
        DeclareLaunchArgument("map_start_delay_sec", default_value="1.0"),
        DeclareLaunchArgument("rog_map_start_delay_sec", default_value="12.0"),
        DeclareLaunchArgument("nav_start_delay_sec", default_value="6.0"),
        DeclareLaunchArgument("rviz_delay_sec", default_value="4.0"),
        DeclareLaunchArgument(
            "planning_grid_owner",
            default_value="rog_map",
            description="Launch-time planning grid owner; only rog_map is implemented in this chain.",
        ),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
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
        DeclareLaunchArgument("force_body_yaw_follow", default_value="false"),
        DeclareLaunchArgument("body_yaw_follow_clearance", default_value="0.55"),
        DeclareLaunchArgument("max_linear_x", default_value="1.0"),
        DeclareLaunchArgument("max_linear_y", default_value="1.0"),
        DeclareLaunchArgument("max_angular_z", default_value="2.0"),
        DeclareLaunchArgument("log_level", default_value="info"),
        sim_launch,
        TimerAction(period=LaunchConfiguration("map_start_delay_sec"), actions=[static_map]),
        TimerAction(period=LaunchConfiguration("rog_map_start_delay_sec"), actions=[rog_map, rog_map_adapter]),
        TimerAction(
            period=LaunchConfiguration("nav_start_delay_sec"),
            actions=[terrain, terrain_ext, localization_fusion, goal_manager, minco, mpc, twist_bridge],
        ),
        rviz,
    ])
