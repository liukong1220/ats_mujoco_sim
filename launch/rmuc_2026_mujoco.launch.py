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
    # P3 关闭 Nav2 时不能遗留 map_server 或 Nav2 lifecycle manager。
    map_group = TimerAction(
        period=LaunchConfiguration("map_start_delay_sec"),
        condition=IfCondition(LaunchConfiguration("launch_nav2")),
        actions=[map_server, map_lifecycle],
    )
    # P3 仍需要静态墙体语义，但不能为此启动任何 Nav2 节点。该发布器保持
    # map_server 的 /map、map frame 和 transient-local 数据契约。
    p3_static_map = Node(
        package="ats_mujoco_sim",
        executable="static_map_publisher",
        name="static_map_publisher",
        output="screen",
        parameters=[{
            "map_yaml_file": LaunchConfiguration("map_yaml_file"),
            "map_topic": "/map",
            "frame_id": "map",
            "use_sim_time": LaunchConfiguration("use_sim_time"),
        }],
    )
    p3_static_map_group = TimerAction(
        period=LaunchConfiguration("map_start_delay_sec"),
        condition=IfCondition(PythonExpression([
            "'", LaunchConfiguration("launch_swerve_mpc"), "'.lower() == 'true' and '",
            LaunchConfiguration("launch_nav2"), "'.lower() == 'false'",
        ])),
        actions=[p3_static_map],
    )

    # Nav2 对照链只在 launch_nav2=true 时启动；P3 不包含此 Include。
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
            "planning_grid_owner": LaunchConfiguration("planning_grid_owner"),
            "launch_fake_vel_transform": PythonExpression(
                ["'", LaunchConfiguration("launch_swerve_mpc"), "'.lower() != 'true'"]
            ),
            "launch_chassis_vel_transform": "False",
            "fake_vel_output_topic": LaunchConfiguration("cmd_vel_topic"),
            "log_level": LaunchConfiguration("log_level"),
        }.items(),
    )
    # 舵轮 MPC 模式只桥接 /cmd_vel_mpc，避免 Nav2 与 MPC 同时向底盘发送速度。
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
            {
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "map_ready_topic": PythonExpression([
                    "'/rog_map_adapter/ready' if '",
                    LaunchConfiguration("planning_grid_owner"),
                    "'.lower() == 'rog_map' else ''",
                ]),
                # P3 只接收目标管理器的带编号请求，显式断开 /plan 和直接 goal 订阅。
                "goal_topic": PythonExpression([
                    "'' if '", LaunchConfiguration("launch_nav2"), "'.lower() == 'false' else 'goal_pose'",
                ]),
                "global_plan_topic": PythonExpression([
                    "'' if '", LaunchConfiguration("launch_nav2"), "'.lower() == 'false' else '/plan'",
                ]),
                "goal_request_topic": PythonExpression([
                    "'/ats_goal_manager/planner_goal' if '", LaunchConfiguration("launch_nav2"),
                    "'.lower() == 'false' else ''",
                ]),
                "planner_status_topic": PythonExpression([
                    "'/minco/planning_status' if '", LaunchConfiguration("launch_nav2"),
                    "'.lower() == 'false' else ''",
                ]),
                "candidate_reference_path_topic": PythonExpression([
                    "'/minco/reference_path_candidate' if '", LaunchConfiguration("launch_nav2"),
                    "'.lower() == 'false' else ''",
                ]),
                "planner_manages_emergency_stop": PythonExpression([
                    "'false' if '", LaunchConfiguration("launch_nav2"), "'.lower() == 'false' else 'true'",
                ]),
            },
        ],
    )
    goal_manager = Node(
        condition=IfCondition(PythonExpression([
            "'", LaunchConfiguration("launch_swerve_mpc"), "'.lower() == 'true' and '",
            LaunchConfiguration("launch_nav2"), "'.lower() == 'false'",
        ])),
        package="ats_goal_manager",
        executable="ats_goal_manager_node",
        name="ats_goal_manager",
        output="screen",
        parameters=[
            LaunchConfiguration("goal_manager_params_file"),
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
    )
    # terrain 节点是 ROGMap 2.5D 融合的输入生产者，不属于 Nav2；P3 必须单独保留。
    p3_terrain_analysis = Node(
        condition=IfCondition(PythonExpression([
            "'", LaunchConfiguration("launch_swerve_mpc"), "'.lower() == 'true' and '",
            LaunchConfiguration("launch_nav2"), "'.lower() == 'false'",
        ])),
        package="terrain_analysis",
        executable="terrainAnalysis",
        name="terrain_analysis",
        output="screen",
        # 顶层 MuJoCo 默认 wall clock，必须覆盖 YAML 中 Nav2 对照遗留的 use_sim_time=true。
        parameters=[nav_params, {"use_sim_time": LaunchConfiguration("use_sim_time")}],
        arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
    )
    p3_terrain_analysis_ext = Node(
        condition=IfCondition(PythonExpression([
            "'", LaunchConfiguration("launch_swerve_mpc"), "'.lower() == 'true' and '",
            LaunchConfiguration("launch_nav2"), "'.lower() == 'false'",
        ])),
        package="terrain_analysis_ext",
        executable="terrainAnalysisExt",
        name="terrain_analysis_ext",
        output="screen",
        parameters=[nav_params, {"use_sim_time": LaunchConfiguration("use_sim_time")}],
        arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
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
    rog_map = Node(
        condition=IfCondition(PythonExpression([
            "'", LaunchConfiguration("launch_rog_map"), "'.lower() == 'true' or '",
            LaunchConfiguration("planning_grid_owner"), "'.lower() == 'rog_map'",
        ])),
        package="ats_rog_map",
        executable="ats_rog_map_node",
        name="ats_rog_map",
        output="screen",
        parameters=[{
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "map_frame": "odom",
            "base_frame": "gimbal_yaw_odom",
            "sensor_frame": "front_mid360",
            "odom_topic": "/localization",
            "cloud_topic": "/registered_scan",
            "map_config_file": LaunchConfiguration("rog_map_config_file"),
            "cloud_timeout_sec": 2.0,
            "odom_timeout_sec": 2.0,
        }],
    )
    rog_map_adapter = Node(
        condition=IfCondition(PythonExpression([
            "'", LaunchConfiguration("planning_grid_owner"), "'.lower() == 'rog_map'",
        ])),
        package="ats_rog_map_adapter",
        executable="ats_rog_map_adapter_node",
        name="ats_rog_map_adapter",
        output="screen",
        parameters=[
            LaunchConfiguration("rog_map_adapter_params_file"),
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
    )
    rog_map_group = TimerAction(
        period=LaunchConfiguration("rog_map_start_delay_sec"),
        actions=[rog_map, rog_map_adapter],
    )
    nav_group = TimerAction(
        period=LaunchConfiguration("nav_start_delay_sec"),
        actions=[
            navigation_launch,
            p3_terrain_analysis,
            p3_terrain_analysis_ext,
            goal_manager,
            minco_planner,
            swerve_mpc,
            twist_bridge,
        ],
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
        DeclareLaunchArgument(
            "rog_map_start_delay_sec",
            default_value="12.0",
            description="在 Nav2 lifecycle 后错峰启动 ROGMap/adapter。",
        ),
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
        DeclareLaunchArgument(
            "launch_nav2",
            default_value="true",
            description="Nav2 对照模式；P3 正式入口必须显式设为 false。",
        ),
        DeclareLaunchArgument(
            "launch_trajectory_optimizer",
            default_value="true",
            description="是否启动 RC-ESDF 局部弹性路径优化器。",
        ),
        DeclareLaunchArgument(
            "launch_twist_bridge",
            default_value="true",
            description="是否启动 Twist 到 MuJoCo /motion_control 的唯一速度桥接器。",
        ),
        DeclareLaunchArgument(
            "launch_swerve_mpc",
            default_value="false",
            description="启用 JPS/MINCO/全向 SE2 MPC；launch_nav2=false 时同时启动 ATS 目标/action 管理。",
        ),
        DeclareLaunchArgument(
            "launch_rog_map",
            default_value="false",
            description="Start the ROGMap 3D occupancy/ESDF perception node for observation only.",
        ),
        DeclareLaunchArgument(
            "planning_grid_owner",
            default_value="rc_esdf",
            choices=["rc_esdf", "rog_map"],
            description="Single planning-grid owner: rc_esdf or rog_map.",
        ),
        DeclareLaunchArgument(
            "rog_map_config_file",
            default_value=PathJoinSubstitution([
                FindPackageShare("ats_rog_map"), "config", "rog_map_mujoco.yaml",
            ]),
            description="ROGMap config; MuJoCo defaults to sparse-scan occupancy fusion.",
        ),
        DeclareLaunchArgument(
            "rog_map_adapter_params_file",
            default_value=PathJoinSubstitution([
                FindPackageShare("ats_rog_map_adapter"),
                "config",
                "rog_map_ground_planning.yaml",
            ]),
            description="ROGMap ground projection and terrain-fusion parameters.",
        ),
        DeclareLaunchArgument(
            "cmd_vel_topic",
            default_value="cmd_vel_gimbal_yaw_odom",
            description="未启用舵轮 MPC 时桥接的 Nav2 速度话题。",
        ),
        DeclareLaunchArgument(
            "mpc_cmd_vel_topic",
            default_value="/cmd_vel_mpc",
            description="启用舵轮 MPC 时桥接的唯一速度话题。",
        ),
        DeclareLaunchArgument(
            "minco_params_file",
            default_value=PathJoinSubstitution([
                FindPackageShare("minco_planner"), "config", "minco_planner.yaml",
            ]),
            description="MINCO/JPS/足迹安全参数 YAML 路径。",
        ),
        DeclareLaunchArgument(
            "goal_manager_params_file",
            default_value=PathJoinSubstitution([
                FindPackageShare("ats_goal_manager"), "config", "ats_goal_manager.yaml",
            ]),
            description="ATS Nav2-free 目标管理/action 状态机参数。",
        ),
        DeclareLaunchArgument(
            "mpc_params_file",
            default_value=PathJoinSubstitution([
                FindPackageShare("ats_swerve_mpc"), "config", "ats_swerve_mpc.yaml",
            ]),
            description="舵轮 SE2 MPC 参数 YAML 路径。",
        ),
        DeclareLaunchArgument(
            "max_linear_x",
            default_value="1.0",
            description="MuJoCo bridge 前后速度限幅（m/s）。",
        ),
        DeclareLaunchArgument(
            "max_linear_y",
            default_value="1.0",
            description="MuJoCo bridge 横移速度限幅（m/s）；舵轮模式不应误设为零。",
        ),
        DeclareLaunchArgument(
            "max_angular_z",
            default_value="2.0",
            description="MuJoCo bridge yaw 角速度限幅（rad/s）。",
        ),
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
        p3_static_map_group,
        rog_map_group,
        nav_group,
        sim_launch,
    ])
