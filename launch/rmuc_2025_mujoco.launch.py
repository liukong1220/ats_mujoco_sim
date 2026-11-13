#!/usr/bin/env python3
"""Launch the RMUC 2025 MuJoCo field with the ATS navigation chain."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import IfElseSubstitution, LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from ats_mujoco_sim.launch_args import SIM_LIO_ARGUMENTS
from ats_mujoco_sim.lio_launch import lio_chain_nodes
from ats_mujoco_sim.robot_description import robot_state_publisher_nodes

# The Mid360 sits on the chassis (report 2.1.8, real vehicle since 2026-10-23), as in the 2027 scene.
# The legacy gimbal pose of rmuc_2025_swerve.xml put about 33 rays per frame on the rear-left steer
# post and wheel at 0.37-0.49 m, past Point-LIO's 0.3 m blind ball: they landed 0.05-0.16 m above
# the wheel contact around the robot, terrain_analysis_ext made 0.2 m hard cells of them on the
# south ramp (3.0-4.5, -6.6..-5.8) and the nav tracking gate failed there (report 5.1.8 B).
# lidar_mount_preset:=legacy_gimbal still reproduces the old pose.
LIO_ARGUMENTS_2025 = {**SIM_LIO_ARGUMENTS, "lidar_mount_preset": "chassis"}


def generate_launch_description() -> LaunchDescription:
    localization_source = LaunchConfiguration("localization_source")
    lio_mode = PythonExpression(["'", localization_source, "' == 'lio'"])
    truth_mode = PythonExpression(["'", localization_source, "' != 'lio'"])
    # localization_source:=lio runs the whole stack on the simulator's /clock.
    def _auto(name, truth_default, lio_default):
        """'auto' resolves to the truth or the lio default; anything else passes through."""
        return PythonExpression([
            "('", lio_default, "' if '", localization_source, "' == 'lio' else '",
            truth_default, "') if '", LaunchConfiguration(name), "' == 'auto' else '",
            LaunchConfiguration(name), "'",
        ])

    def _wall_window(seconds):
        """A node's wall-clock window, stretched by lio_lease_wall_scale in lio (slower than real time)."""
        return ParameterValue(
            PythonExpression([
                str(seconds), " * float('", LaunchConfiguration("lio_lease_wall_scale"),
                "') if '", localization_source, "' == 'lio' else ", str(seconds),
            ]),
            value_type=float,
        )

    use_sim_time = PythonExpression([
        "'", localization_source, "' == 'lio' or '",
        LaunchConfiguration("use_sim_time"), "'.lower() in ('true', '1')",
    ])
    planning_grid_owner = LaunchConfiguration("planning_grid_owner")
    slam = LaunchConfiguration("slam")
    model_file = LaunchConfiguration("model_file")
    rmuc_2025_navigation_profile = PathJoinSubstitution([
        FindPackageShare("ats_mujoco_sim"),
        "config",
        "rmuc_2025_navigation.yaml",
    ])
    lio_odom_navigation_profile = PathJoinSubstitution([
        FindPackageShare("ats_mujoco_sim"), "config", "rog_map_lio_odom.yaml",
    ])

    static_map = Node(
        package="ats_mujoco_sim",
        executable="static_map_publisher",
        name="static_map_publisher",
        output="screen",
        condition=UnlessCondition(slam),
        parameters=[{
            "use_sim_time": use_sim_time,
            "map_yaml_file": LaunchConfiguration("map_yaml_file"),
            "map_topic": "/map",
            "frame_id": "map",
        }],
    )
    # /terrain_map 只进 terrain_analysis_ext 的 4 m 近域点云 /terrain_map_ext（建图模式的
    # pointcloud_to_laserscan 与 RViz 在用），/traversability_grid 由 terrain_analysis_ext 自己的
    # 体素统计算出、不读它（调研报告 9.4 第 5 项），所以导航默认不启动。
    terrain = Node(
        package="terrain_analysis",
        executable="terrainAnalysis",
        name="terrain_analysis",
        output="screen",
        parameters=[LaunchConfiguration("params_file"), {"use_sim_time": use_sim_time}],
        condition=IfCondition(LaunchConfiguration("launch_terrain_analysis")),
    )
    # The RMUC profile overrides the planar lattice resolution here: the
    # traversability/slope grids are published at planarVoxelSize and sampled
    # nearest-cell into the 0.10 m planning grid, so the lattice size bounds how
    # far a single coarse obstacle verdict can over-report westward.
    terrain_ext = Node(
        package="terrain_analysis_ext",
        executable="terrainAnalysisExt",
        name="terrain_analysis_ext",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            rmuc_2025_navigation_profile,
            LaunchConfiguration("scene_navigation_profile"),
            {"use_sim_time": use_sim_time},
        ],
    )
    fusion_parameters = {
            "use_identified_odometry": ParameterValue(LaunchConfiguration("use_identified_odometry"), value_type=bool),
            "identified_odom_topic": "/odometry_identified",
            "identity_monitor_topic": "/aft_mapped_to_init_identified",
            "use_sim_time": use_sim_time,
            "odom_topic": LaunchConfiguration("fusion_odom_topic"),
            "localization_topic": "/localization",
            "observation_topic": "/relocalization_observation",
            "status_topic": "/localization/status",
            "map_frame": "map",
            "odom_frame": "odom",
            "robot_base_frame": "gimbal_yaw_odom",
            "publish_tf": True,
            "allow_initial_identity": True,
            # Match ROGMap/planning odom patience. A 0.5 s blip under dense GICP
            # CPU load forces LOST before the first coarse+fine window can accept,
            # which opens multi_guess and historically wandered off origin.
            "odom_timeout_s": 2.0,
            "observation_timeout_s": 3.0,
            # Cold-start identity prior needs longer than one multi_guess sweep
            # before LOST opens the lattice (straight188 free-space failed while
            # epoch churned under 10 s lost timeout).
            "observation_lost_timeout_s": 30.0,
            # 超过 epoch 阈值（0.05 m/0.05 rad）但不超过 0.20 m/0.10 rad 的 GICP 修正直接改
            # map->odom、不推进 epoch。此前出发后首次确认修正（实测 5~9 cm）都推进 epoch，
            # GoalManager 急停并整链重规划约 1.35 s；规划器复检按当前 TF 重新表达已提交参考，
            # MPC 在 odom 执行。超过上界的跳变仍推进 epoch。
            "continuous_correction_max_translation": 0.20,
            "continuous_correction_max_yaw": 0.10,
    }
    localization_fusion = Node(
        package="small_gicp_relocalization",
        executable="localization_fusion_node",
        name="localization_fusion",
        output="screen",
        condition=IfCondition(truth_mode),
        parameters=[fusion_parameters],
    )
    # In lio mode odom is Point-LIO's start frame (origin on the ground below
    # the start pose, yaw = start yaw), so the known start pose is the initial
    # map->odom seed instead of the identity truth mode uses.
    localization_fusion_lio = Node(
        package="small_gicp_relocalization",
        executable="localization_fusion_node",
        name="localization_fusion",
        output="screen",
        condition=IfCondition(lio_mode),
        parameters=[{
            **fusion_parameters,
            "allow_initial_identity": False,
            "use_initial_map_to_odom": True,
            "initial_map_to_odom_x": ParameterValue(LaunchConfiguration("start_x"), value_type=float),
            "initial_map_to_odom_y": ParameterValue(LaunchConfiguration("start_y"), value_type=float),
            # odom's initial datum is start_z minus the configured base height.
            # It approximates supporting ground on a level, settled spawn; a
            # contact-free elevated spawn or uneven ground need not equal terrain z.
            # A fixed 0.0 left GICP a 0.2-0.3 m vertical error on raised ground, which is
            # outside its 0.45 m correspondence reach once the first guess is off in x/y/yaw.
            "initial_map_to_odom_z": ParameterValue(
                PythonExpression([
                    "float('", LaunchConfiguration("start_z"), "') - float('",
                    LaunchConfiguration("base_footprint_height"), "')",
                ]),
                value_type=float,
            ),
            "initial_map_to_odom_yaw": ParameterValue(LaunchConfiguration("start_yaw"), value_type=float),
            # The fusion node ages the accepted observation on the steady (wall) clock: 3 s against a
            # GICP refresh of 1 s scan time (node_params.yaml: "smaller than fusion's 3 s lease").
            # The simulation runs at about 0.4x real time, so one refresh + accumulation + two
            # confirmations takes about 6.5 wall seconds and the lease expired every cycle
            # (DEGRADED 44 % of a red-box run, report 5.1.4 B2).  Scale the lease by the slowdown so
            # the design ratio holds; GICP's quality gates are untouched.
            "observation_timeout_s": _wall_window(3.0),
        }],
    )
    # MuJoCo publishes /registered_scan in odom/world coordinates.  Its matching
    # prior PCD uses the same map coordinates, so GICP consumes odom coordinates
    # directly.  Fusion remains the sole map->odom authority.
    relocalization = Node(
        package="small_gicp_relocalization",
        executable="small_gicp_relocalization_node",
        name="small_gicp_relocalization",
        output="screen",
        condition=IfCondition(PythonExpression([
            "'", LaunchConfiguration("launch_small_gicp_relocalization"),
            "'.lower() in ('true', '1') and ", truth_mode,
        ])),
        # The scene profile may carry a small_gicp_relocalization section (RMUL needs finer covariance
        # scales than the RMUC defaults); scenes without that section keep node_params.yaml unchanged.
        parameters=[LaunchConfiguration("params_file"), LaunchConfiguration("scene_navigation_profile"), {
            "use_sim_time": use_sim_time,
            "prior_pcd_file": LaunchConfiguration("prior_pcd_file"),
            "map_frame": "map",
            "odom_frame": "odom",
            "robot_base_frame": "gimbal_yaw_odom",
            "publish_tf": False,
        }],
    )

    def lio_relocalization(context, *args, **kwargs):
        # GICP's initial guess must equal the fusion seed: odom is the
        # Point-LIO start frame, so map->odom is the known start pose.
        lc = context.launch_configurations
        mapping_mode = lc["slam"].lower() in ("true", "1")
        if (lc["localization_source"] != "lio" or
                (not mapping_mode and
                 lc["launch_small_gicp_relocalization"].lower() not in ("true", "1"))):
            return []
        init_pose = [
            float(lc["start_x"]), float(lc["start_y"]),
            float(lc["start_z"]) - float(lc["base_footprint_height"]),
            0.0, 0.0, float(lc["start_yaw"]),
        ]
        return [Node(
            package="small_gicp_relocalization",
            executable="small_gicp_relocalization_node",
            name="small_gicp_relocalization",
            output="screen",
            parameters=[lc["params_file"], lc["scene_navigation_profile"], {
                "use_sim_time": True,
                "prior_pcd_file": lc["prior_pcd_file"],
                "mapping_mode": mapping_mode,
                "require_odom_identity": lc["use_identified_odometry"].lower() == "true",
                "map_frame": "map",
                "odom_frame": "odom",
                "robot_base_frame": "gimbal_yaw_odom",
                "publish_tf": False,
                "init_pose": init_pose,
            }],
        )]

    def lio_chain(context, *args, **kwargs):
        return lio_chain_nodes(context.launch_configurations)

    def robot_model(context, *args, **kwargs):
        # Vehicle shape for the RViz RobotModel; it publishes no TF (see robot_description.py).
        return robot_state_publisher_nodes(context.launch_configurations)

    # Startup-only test-fault authorization. It defaults to false so nominal and
    # real-vehicle launches can never clear the map; only the isolated
    # P2_FAULT_CASE=unknown run passes true explicitly.
    enable_test_fault_injection = LaunchConfiguration("enable_test_fault_injection")
    # One process: the map node (/ats_rog_map) and the planning-map owner
    # (/ats_rog_map_adapter), which calls the projection in-process. `name=` is
    # left unset so launch_ros writes the override dict under '/**' and both nodes
    # receive it; each declares only its own keys, and the shared keys
    # (use_sim_time, enable_test_fault_injection, require_odom_identity,
    # planning_grid_owner) mean the same thing in both. YAML sections stay keyed
    # by node name. planning_grid_owner:=rc_esdf starts the map node alone.
    rog_map = Node(
        package="ats_rog_map",
        executable="ats_rog_map_node",
        output="screen",
        parameters=[LaunchConfiguration("params_file"), rmuc_2025_navigation_profile,
                    LaunchConfiguration("scene_navigation_profile"),
                    IfElseSubstitution(lio_mode, lio_odom_navigation_profile,
                                       LaunchConfiguration("scene_navigation_profile")), {
            "use_sim_time": use_sim_time,
            "planning_grid_owner": planning_grid_owner,
            "enable_test_fault_injection": ParameterValue(
                enable_test_fault_injection, value_type=bool
            ),
            "require_odom_identity": ParameterValue(LaunchConfiguration("use_identified_odometry"), value_type=bool),
            "identity_timeout_sec": _wall_window(1.0),
            "odom_layer.base_ground_offset": ParameterValue(
                LaunchConfiguration("base_footprint_height"), value_type=float),
            "mapping_mode": ParameterValue(slam, value_type=bool),
            "allow_missing_static_map": ParameterValue(slam, value_type=bool),
            "robot_unknown_clear_footprint": ParameterValue(slam, value_type=bool),
            "use_static_map_obstacles": ParameterValue(
                PythonExpression(["'", slam, "'.lower() != 'true'"]), value_type=bool),
            # The adapter ages /localization/status on the wall clock (1.0 s); its
            # projection TF lookup waits 0.1 s of sim time, i.e. 0.1 * slowdown wall
            # seconds with no status received. Scale it by the slowdown like the
            # fusion lease; truth mode keeps node_params.yaml.
            "localization_status_timeout_sec": _wall_window(1.0),
        }],
    )
    goal_manager = Node(
        package="ats_goal_manager",
        executable="ats_goal_manager_node",
        name="ats_goal_manager",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            rmuc_2025_navigation_profile,
            {
                "use_sim_time": use_sim_time,
                "require_localization_status": True,
                "require_accepted_localization": True,
                "planner_start_in_planning_frame": True,
                # Keep estimator/control convergence tighter than the independent
                # 8 cm truth acceptance gate; mapping has no absolute prior.
                "goal_position_tolerance": ParameterValue(
                    PythonExpression(["0.03 if '", slam, "'.lower() == 'true' else 0.08"]),
                    value_type=float),
                # The progress watchdog counts wall seconds without 0.10 m of progress; at 0.1x real
                # time the 4 s default is 0.4 sim seconds, less than the vehicle needs to start moving,
                # and two such replans abort the goal.  Same stretch as the other wall windows.
                "replan_stall_timeout_sec": _wall_window(4.0),
                "replan_min_interval_sec": _wall_window(2.0),
            },
        ],
    )
    minco = Node(
        package="minco_planner",
        executable="minco_planner_node",
        name="minco_planner",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            rmuc_2025_navigation_profile,
            LaunchConfiguration("scene_navigation_profile"),
            {
                "use_sim_time": use_sim_time,
                # Admission offset plus MPC stopping error must fit inside
                # the manager's unchanged 0.03 m mapping success disk.
                "goal_admission_position_tolerance": ParameterValue(
                    PythonExpression(["0.02 if '", slam, "'.lower() == 'true' else 0.08"]),
                    value_type=float),
                "goal_admission_position_step": ParameterValue(
                    PythonExpression(["0.005 if '", slam, "'.lower() == 'true' else 0.02"]),
                    value_type=float),
                "force_body_yaw_follow": LaunchConfiguration("force_body_yaw_follow"),
                "body_yaw_follow_clearance": LaunchConfiguration("body_yaw_follow_clearance"),
                "spin_enabled": ParameterValue(LaunchConfiguration("spin_enabled"), value_type=bool),
                "spin_rate": ParameterValue(LaunchConfiguration("spin_rate"), value_type=float),
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
                "command_topic": "/cmd_vel/autonomy_raw",
                "goal_position_tolerance": ParameterValue(
                    PythonExpression(["0.01 if '", slam, "'.lower() == 'true' else 0.08"]),
                    value_type=float),
                # MINCO can admit up to 0.09 rad of yaw offset. Leave room
                # for it within the manager's original 0.15 rad success arc.
                "goal_yaw_tolerance": ParameterValue(
                    PythonExpression(["0.05 if '", slam, "'.lower() == 'true' else 0.15"]),
                    value_type=float),
                "execution_command_topic": "/planner/execution_command",
                "require_localization_status": True,
                "solver_mode": LaunchConfiguration("solver_mode"),
                "telemetry_sampling_window_cycles": LaunchConfiguration(
                    "telemetry_sampling_window_cycles"
                ),
                # /odometry arrives at the LIO scan rate in sim time (20 Hz), i.e. one frame per
                # 0.58 wall seconds at 0.087x real time, and the MPC ages it on the wall clock
                # (0.25 s): it printed 【传感器超时】 every cycle and held zero velocity.
                "odometry_timeout": _wall_window(0.25),
            },
        ],
    )
    twist_bridge = Node(
        package="ats_mujoco_sim",
        executable="twist_to_motion_ctrl",
        name="twist_to_motion_ctrl",
        output="screen",
        parameters=[{
            "input_topic": "/cmd_vel/selected",
            "output_topic": "/motion_control",
            "max_linear_speed": LaunchConfiguration("max_linear_speed"),
            "max_angular_z": LaunchConfiguration("max_angular_z"),
        }],
    )
    cmd_vel_arbiter = Node(
        package="ats_cmd_vel_arbiter",
        executable="cmd_vel_arbiter_node",
        name="cmd_vel_arbiter",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            {
                "use_sim_time": use_sim_time,
                "manual_cmd_vel_topic": "/cmd_vel",
                "autonomy_cmd_vel_topic": "/cmd_vel/autonomy_raw",
                "selected_cmd_vel_topic": "/cmd_vel/selected",
                "planner_status_topic": "/minco/planning_status",
                "map_ready_topic": "/rog_map_adapter/ready",
                "require_serial_link": False,
            },
        ],
    )
    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([FindPackageShare("ats_mujoco_sim"), "launch", "ats_mujoco_sim.launch.py"])
        ]),
        launch_arguments={
            "model_path": PathJoinSubstitution([
                FindPackageShare("ats_mujoco_sim"), "models", model_file
            ]),
            "scene_file": PathJoinSubstitution([
                FindPackageShare("ats_mujoco_sim"), "models", model_file
            ]),
            "enable_static_walls": LaunchConfiguration("enable_static_walls"),
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
            "sim_rate_hz": _auto("sim_rate_hz", "300.0", "400.0"),
            "feedback_rate_hz": LaunchConfiguration("feedback_rate_hz"),
            "truth_rate_hz": LaunchConfiguration("truth_rate_hz"),
            "command_timeout": LaunchConfiguration("command_timeout"),
            "max_wheel_acceleration": LaunchConfiguration("max_wheel_acceleration"),
            "max_steer_rate": LaunchConfiguration("max_steer_rate"),
            "freeze_motion": LaunchConfiguration("freeze_motion"),
            "enable_lidar": LaunchConfiguration("enable_lidar"),
            "lidar_backend": LaunchConfiguration("lidar_backend"),
            "lidar_model": "mid360",
            "lidar_downsample": _auto("lidar_downsample", "24", "1"),
            "lidar_rate_hz": _auto("lidar_rate_hz", "10.0", "20.0"),
            "lidar_rate_clock": LaunchConfiguration("lidar_rate_clock"),
            "lidar_state_rate_hz": LaunchConfiguration("lidar_state_rate_hz"),
            "lidar_topic": "/local_pointcloud",
            "lidar_frame_id": "front_mid360",
            "registered_scan_topic": "/registered_scan",
            "registered_scan_frame_id": "odom",
            "odom_topic": LaunchConfiguration("mujoco_odom_topic"),
            "publish_map_to_odom_tf": "false",
            "enable_tof": LaunchConfiguration("enable_tof"),
            "tof_backend": LaunchConfiguration("tof_backend"),
            "tof_rate_hz": LaunchConfiguration("tof_rate_hz"),
            "merged_tof_topic": "/perception/tof/points_merged",
            **{name: LaunchConfiguration(name) for name in LIO_ARGUMENTS_2025},
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
        ), Node(
            condition=IfCondition(LaunchConfiguration("use_rviz")),
            package="ats_mujoco_sim",
            executable="robot_footprint_marker",
            name="robot_footprint_marker",
            output="screen",
        )],
    )

    return LaunchDescription([
        DeclareLaunchArgument("start_x", default_value="-0.18"),
        DeclareLaunchArgument("start_y", default_value="0.06"),
        # Base-link qpos must place the 42.5 mm wheels on the RMUC hfield: the
        # start cell is about 0.201 m high and the wheel-bottom offset is 0.180 m.
        DeclareLaunchArgument("start_z", default_value="0.381"),
        DeclareLaunchArgument("start_yaw", default_value="0.0"),
        DeclareLaunchArgument("model_file", default_value="rmuc_2025_swerve.xml"),
        DeclareLaunchArgument("use_viewer", default_value="true"),
        DeclareLaunchArgument("show_viewer", default_value="true"),
        DeclareLaunchArgument(
            "launch_physics",
            default_value="true",
            description="Start MuJoCo physics and sensor processes.",
        ),
        DeclareLaunchArgument(
            "launch_navigation",
            default_value="true",
            description="Start map, localization, planning and control processes.",
        ),
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
                FindPackageShare("ats_sentry_bringup"), "map", "rmuc_2025.yaml"
            ]),
        ),
        DeclareLaunchArgument(
            "prior_pcd_file",
            default_value=PathJoinSubstitution([
                FindPackageShare("ats_sentry_bringup"), "pcd", "rmuc_2025.pcd"
            ]),
            description="GICP prior map; regenerate with scripts/transform_prior_pcd.py.",
        ),
        DeclareLaunchArgument(
            "params_file",
            default_value=PathJoinSubstitution([
                FindPackageShare("ats_sentry_bringup"), "params", "node_params.yaml"
            ]),
        ),
        # Per-scene overlay layered after the common profile.  The 2025 default is the 2025 profile
        # itself (read twice, no change); rmuc_2027_mujoco.launch.py passes rmuc_2027_navigation.yaml.
        DeclareLaunchArgument(
            "scene_navigation_profile",
            default_value=rmuc_2025_navigation_profile,
            description="Scene-specific parameter overlay applied after the common RMUC profile.",
        ),
        DeclareLaunchArgument(
            "solver_mode",
            default_value="ilqr",
            description="ats_swerve_mpc solver mode; qp_shadow is diagnostic-only.",
        ),
        DeclareLaunchArgument(
            "telemetry_sampling_window_cycles",
            default_value="0",
            description="Fixed MPC telemetry sample count; 0 keeps rolling diagnostics.",
        ),
        DeclareLaunchArgument(
            "launch_terrain_analysis",
            default_value="false",
            description="Start terrain_analysis; only the near field of /terrain_map_ext uses it.",
        ),
        DeclareLaunchArgument("map_start_delay_sec", default_value="1.0"),
        DeclareLaunchArgument("rog_map_start_delay_sec", default_value="12.0"),
        DeclareLaunchArgument("nav_start_delay_sec", default_value="6.0"),
        DeclareLaunchArgument("rviz_delay_sec", default_value="4.0"),
        DeclareLaunchArgument(
            "enable_test_fault_injection",
            default_value="false",
            description=(
                "Startup-only authorization for ROGMap/adapter test fault fixtures. "
                "Keep false for nominal and real-vehicle runs; only the isolated "
                "P2_FAULT_CASE=unknown run may set it true."
            ),
        ),
        DeclareLaunchArgument("save_odom_layer_on_shutdown", default_value="false"),
        DeclareLaunchArgument("mapping_output_prefix", default_value=""),
        DeclareLaunchArgument(
            "planning_grid_owner",
            default_value="rog_map",
            description="Launch-time planning grid owner; only rog_map is implemented in this chain.",
        ),
        DeclareLaunchArgument(
            "slam", default_value="false",
            description="Mapping mode: omit the static PGM publisher and prior-PCD relocalization.",
        ),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument(
            "enable_static_walls",
            default_value="false",
            description=(
                "兼容通用 MuJoCo 场景的静态墙体开关；"
                "当前 RMUC 主场景源 XML 已移除墙体 include，参数不会恢复墙体，保留 heightfield 地形支撑。"
            ),
        ),
        DeclareLaunchArgument("mujoco_odom_topic", default_value="/odometry"),
        DeclareLaunchArgument("fusion_odom_topic", default_value="/odometry"),
        DeclareLaunchArgument(
            "launch_small_gicp_relocalization",
            default_value="true",
            description="Run real GICP; disable only for an isolated synthetic observation fixture.",
        ),
        *[DeclareLaunchArgument(n, default_value=v) for n, v in LIO_ARGUMENTS_2025.items()],
        # lio only: multiplier for localization_fusion's wall-clock observation lease (see above).
        DeclareLaunchArgument("lio_lease_wall_scale", default_value="4.0"),
        # 'auto' = 300 Hz in truth mode, 400 Hz in lio mode (a multiple of the 200 Hz IMU).
        DeclareLaunchArgument("sim_rate_hz", default_value="auto"),
        DeclareLaunchArgument("feedback_rate_hz", default_value="10.0"),
        DeclareLaunchArgument("truth_rate_hz", default_value="10.0"),
        DeclareLaunchArgument("command_timeout", default_value="0.5"),
        # Chassis actuator limits (same defaults as ats_mujoco_sim.launch.py, i.e. the real
        # 120 rpm steer motor and 2.0 m/s^2 wheel acceleration); exposed for the report 5.1.10
        # item 3 causal experiment, not for tuning the plant to fit the controller.
        DeclareLaunchArgument("max_wheel_acceleration", default_value="2.0"),
        DeclareLaunchArgument("max_steer_rate", default_value="10.4719755"),
        DeclareLaunchArgument(
            "freeze_motion",
            default_value="false",
            description="Hold the simulated chassis while keeping sensors and localization active.",
        ),
        DeclareLaunchArgument("enable_lidar", default_value="true"),
        DeclareLaunchArgument("lidar_backend", default_value="cpu"),
        # 'auto' = downsample 24 / 10 Hz in truth mode, 1 / 20 Hz in lio mode.
        DeclareLaunchArgument("lidar_downsample", default_value="auto"),
        DeclareLaunchArgument("lidar_rate_hz", default_value="auto"),
        DeclareLaunchArgument("lidar_rate_clock", default_value="wall"),
        DeclareLaunchArgument("lidar_state_rate_hz", default_value="30.0"),
        DeclareLaunchArgument("enable_tof", default_value="false"),
        DeclareLaunchArgument("tof_backend", default_value="cpu"),
        DeclareLaunchArgument("tof_rate_hz", default_value="10.0"),
        DeclareLaunchArgument("force_body_yaw_follow", default_value="false"),
        DeclareLaunchArgument("body_yaw_follow_clearance", default_value="0.55"),
        DeclareLaunchArgument("spin_enabled", default_value="true",
                              description="Spin while translating in safe open segments of an active goal."),
        DeclareLaunchArgument("spin_rate", default_value="3.0",
                              description="Requested spin yaw rate [rad/s]; safety/actuator limits still apply."),
        # Same vector limit as the real exit (node_params.yaml chassis_vel_transform
        # max_linear_speed 1.5); the old per-axis 1.0 bent 32-48 % of driving commands.
        DeclareLaunchArgument("max_linear_speed", default_value="1.5"),
        DeclareLaunchArgument("max_angular_z", default_value="2.0"),
        DeclareLaunchArgument("log_level", default_value="info"),
        GroupAction(
            condition=IfCondition(LaunchConfiguration("launch_physics")),
            actions=[sim_launch],
        ),
        GroupAction(
            condition=IfCondition(LaunchConfiguration("launch_navigation")),
            actions=[
                TimerAction(
                    period=LaunchConfiguration("map_start_delay_sec"),
                    actions=[static_map, OpaqueFunction(function=lio_chain)],
                ),
                TimerAction(
                    period=LaunchConfiguration("rog_map_start_delay_sec"),
                    actions=[rog_map],
                ),
                TimerAction(
                    period=LaunchConfiguration("nav_start_delay_sec"),
                    actions=[
                        terrain,
                        terrain_ext,
                        localization_fusion,
                        localization_fusion_lio,
                        relocalization,
                        OpaqueFunction(function=lio_relocalization),
                        goal_manager,
                        minco,
                        mpc,
                        cmd_vel_arbiter,
                        twist_bridge,
                    ],
                ),
                OpaqueFunction(function=robot_model),
                rviz,
            ],
        ),
    ])
