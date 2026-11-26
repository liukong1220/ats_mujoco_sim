"""ROS2 interface that drives the MuJoCo four-swerve chassis model."""

from ats_mujoco_sim.sim_sensor_math import *  # noqa: F401,F403
from ats_mujoco_sim.sim_sensor_math import (
    _cast_tof_rays,
    _import_mujoco_lidar,
    _import_multi_ray,
    _livox_pointcloud_message,
    _mat_to_xyzw,
    _pointcloud_message,
    _publish_pointcloud,
    _quat_wxyz_from_yaw,
    _quat_xyzw_from_rpy,
    _refresh_viewer_data,
    _relative_site_transform,
    _shutdown_rclpy_if_needed,
    _sim_time_to_stamp,
    _site_points_to_world,
    _timeout_clock,
    _tof_scan_pattern,
    _tof_side_targets,
    _transform_site_points,
    _yaw_from_quat_wxyz,
)
from ats_mujoco_sim.sim_sensor_workers import (
    _lidar_process_lio_main,
    _lidar_process_main,
    _tof_process_main,
)
from ats_mujoco_sim.sim_assets_obstacles import SimAssetsAndObstaclesMixin
from ats_mujoco_sim.sim_lio_lidar import SimLioAndLidarMixin
from ats_mujoco_sim.sim_drive_telemetry import SimDriveAndTelemetryMixin


class SwerveMujocoSim(
    SimAssetsAndObstaclesMixin,
    SimLioAndLidarMixin,
    SimDriveAndTelemetryMixin,
    Node,
):
    """Simulate the real WL100 control interface on top of MuJoCo."""

    def __init__(self):
        super().__init__("ats_mujoco_sim")

        self.declare_parameter("model_path", "")
        self.declare_parameter("scene_file", "")
        self.declare_parameter("enable_static_walls", True)
        self.declare_parameter("map_dir", "")
        self.declare_parameter("map_manifest_path", "")
        self.declare_parameter("map_ready_file", "")
        self.declare_parameter("map_ready_token", "")
        self.declare_parameter("map_wait_timeout_sec", 0.0)
        self.declare_parameter("sim_rate_hz", 300.0)
        self.declare_parameter("feedback_rate_hz", 10.0)
        self.declare_parameter("command_timeout", 0.5)
        self.declare_parameter("cmd_timeout", 0.5)
        # Fault injection keeps sensor/localization publishers alive while the
        # simulated chassis refuses motion commands.
        self.declare_parameter("freeze_motion", False)
        # Test-only sensor fault.  The LiDAR process continues publishing
        # timestamped empty returns so this remains distinct from input stale.
        self.declare_parameter("lidar_occlusion_enabled", False)
        self.declare_parameter("enable_test_obstacle_fixture", False,
                               ParameterDescriptor(read_only=True))
        self.declare_parameter("test_obstacle_position", [2.0, 0.0, -10.0])
        self.declare_parameter("wheel_radius", WHEEL_RADIUS_M)
        self.declare_parameter("max_wheel_speed", MAX_WHEEL_SPEED_MPS)
        self.declare_parameter("max_wheel_acceleration", 2.0)
        self.declare_parameter("max_steer_rate", MAX_STEER_RATE_RADPS)
        self.declare_parameter("emergency_stop_topic", "/planner/emergency_stop")
        self.declare_parameter("swerve_telemetry_topic", "/swerve/telemetry")
        self.declare_parameter("gimbal_status_topic", "/gimbal/yaw_status")
        self.declare_parameter(
            "yaw_authority_request_topic", "/gimbal/yaw_authority_request"
        )
        self.declare_parameter("show_viewer", False)
        self.declare_parameter("use_viewer", False)
        self.declare_parameter("viewer_rate_hz", 15.0)
        self.declare_parameter("truth_rate_hz", 10.0)
        self.declare_parameter("enable_lidar", True)
        self.declare_parameter("lidar_backend", "gpu")
        self.declare_parameter("lidar_model", "mid360")
        self.declare_parameter("lidar_downsample", 0)
        self.declare_parameter("lidar_rate_hz", 10.0)
        self.declare_parameter("lidar_rate_clock", "sim")
        self.declare_parameter("lidar_state_rate_hz", 0.0)
        self.declare_parameter("lidar_site", "lidar_raycast_site")
        self.declare_parameter("lidar_frame_site", "lidar_site")
        self.declare_parameter("lidar_topic", "/local_pointcloud")
        self.declare_parameter("lidar_frame_id", "front_mid360")
        self.declare_parameter("registered_scan_topic", "/registered_scan")
        self.declare_parameter("registered_scan_frame_id", "")
        self.declare_parameter("enable_tof", True)
        self.declare_parameter("tof_backend", "cpu")
        self.declare_parameter("tof_range", 1.0)
        self.declare_parameter("tof_min_range", 0.03)
        self.declare_parameter("tof_rate_hz", 10.0)
        self.declare_parameter("tof_width", 248)
        self.declare_parameter("tof_height", 180)
        self.declare_parameter("tof_horizontal_fov_deg", 98.0)
        self.declare_parameter("tof_vertical_fov_deg", 72.0)
        self.declare_parameter("tof_footprint_length", 0.58)
        self.declare_parameter("tof_footprint_width", 0.58)
        self.declare_parameter("tof_footprint_expand", 1.0)
        self.declare_parameter("tof_footprint_resolution", 0.05)
        self.declare_parameter("tof_footprint_z_min", -0.10)
        self.declare_parameter("tof_footprint_z_max", 0.13)
        self.declare_parameter("merged_tof_topic", "/perception/tof/points_merged")
        self.declare_parameter("left_tof_site", "left_tof_site")
        self.declare_parameter("right_tof_site", "right_tof_site")
        self.declare_parameter("left_tof_topic", "/left_tof/points")
        self.declare_parameter("right_tof_topic", "/right_tof/points")
        self.declare_parameter("left_tof_frame_id", "left_tof_link")
        self.declare_parameter("right_tof_frame_id", "right_tof_link")
        self.declare_parameter("odom_topic", "/localization")
        self.declare_parameter("lidar_odometry_topic", "/lidar_odometry")
        self.declare_parameter("map_frame_id", "map")
        self.declare_parameter("odom_frame_id", "odom")
        self.declare_parameter("publish_map_to_odom_tf", True)
        self.declare_parameter("lidar_tf_frame_id", "front_mid360")
        self.declare_parameter("robot_base_frame_id", "gimbal_yaw_odom")
        self.declare_parameter("publish_robot_base_tf", True)
        self.declare_parameter("base_footprint_frame_id", "base_footprint")
        # Height of robot_base_frame_id (base_link body) above the wheel contact
        # plane: steer z -0.1375 + wheel radius 0.0425 in models/*.xml.
        self.declare_parameter("base_footprint_height", 0.18)
        self.declare_parameter("base_frame_id", "base_link")
        self.declare_parameter("pose_cmd_topic", "/simulation/PoseSub")
        self.declare_parameter("reset_pose_service_topic", "/simulation/reset_pose")
        self.declare_parameter("start_x", 0.0)
        self.declare_parameter("start_y", 0.0)
        self.declare_parameter("start_z", 0.18)
        self.declare_parameter("start_roll", 0.0)
        self.declare_parameter("start_pitch", 0.0)
        self.declare_parameter("start_yaw", 0.0)
        self.declare_parameter("dynamic_obstacles_enabled", False)
        self.declare_parameter("dynamic_obstacle_count", 0)
        self.declare_parameter("dynamic_obstacle_shape", "cylinder")
        self.declare_parameter("dynamic_obstacle_radius", 0.25)
        self.declare_parameter("dynamic_obstacle_height", 1.0)
        self.declare_parameter("dynamic_obstacle_mass", 20.0)
        self.declare_parameter("dynamic_obstacle_speed", 0.4)
        self.declare_parameter("dynamic_obstacle_path_length", 6.0)
        self.declare_parameter("dynamic_obstacle_min_robot_distance", 1.8)
        self.declare_parameter("dynamic_obstacle_near_robot_radius", 4.0)
        self.declare_parameter("dynamic_obstacle_map_clearance", 0.35)
        self.declare_parameter("dynamic_obstacle_replan_rate_hz", 1.0)

        # --- LIO mode (report appendix A.21) --------------------------------
        # "truth" keeps the original behaviour bit for bit.  "lio" publishes
        # only raw sensors (/clock, /livox/imu, a time-resolved Mid360 cloud)
        # plus an evaluation-only ground truth; localization then comes from
        # point_lio -> loam_interface -> sensor_scan_generation.
        self.declare_parameter("localization_source", "truth")
        self.declare_parameter("imu_topic", "/livox/imu")
        self.declare_parameter("imu_frame_id", "front_mid360")
        self.declare_parameter("imu_rate_hz", 200.0)
        self.declare_parameter("clock_topic", "/clock")
        self.declare_parameter(
            "ground_truth_odom_topic", "/simulation/ground_truth/odometry"
        )
        self.declare_parameter("ground_truth_rate_hz", 50.0)
        self.declare_parameter("livox_points_topic", "/mujoco/livox_points")
        self.declare_parameter("lidar_stats_topic", "/simulation/lidar_frame_stats")
        self.declare_parameter("lidar_segment_ms", 5.0)
        # LIO tracing threads (CPU backend). 0 budgets one third of allowed CPUs, at most 8.
        self.declare_parameter("lidar_threads", 0)
        self.declare_parameter("lidar_max_range", 40.0)
        # Backpressure preserves every state. A deep buffer adds sensor latency
        # (64 physics steps = 160 ms at 400 Hz) before navigation even sees it.
        self.declare_parameter("lidar_state_queue_depth", 2)
        # False lets the chassis' own geometry return echoes (S0 item 8 check).
        self.declare_parameter("lidar_self_exclusion", True)
        # Noise models, all off by default so regressions stay deterministic.
        self.declare_parameter("noise_seed", 20260723)
        self.declare_parameter("enable_range_noise", False)
        self.declare_parameter("range_noise_sigma_m", 0.02)
        self.declare_parameter("enable_imu_noise", False)
        self.declare_parameter("enable_imu_saturation", False)
        self.declare_parameter("imu_gyro_noise_std", 7.0e-4)
        self.declare_parameter("imu_accel_noise_std", 1.4e-3)
        self.declare_parameter("imu_gyro_bias_walk", 2.0e-5)
        self.declare_parameter("imu_accel_bias_walk", 1.0e-5)
        # Lidar mount on the chassis.  "legacy_gimbal" keeps the model's own
        # (pre-2026-10-23) pose so older scenes reproduce exactly; "chassis"
        # applies the parameters below (report 2.1.8).
        self.declare_parameter("lidar_mount_preset", "chassis")
        self.declare_parameter("lidar_mount_x", 0.2405)
        self.declare_parameter("lidar_mount_y", 0.0)
        self.declare_parameter("lidar_mount_height", 0.20)
        self.declare_parameter("lidar_mount_tilt_direction", "front")
        self.declare_parameter("lidar_mount_tilt_deg", 30.0)

        self.localization_source = str(
            self.get_parameter("localization_source").value
        ).lower()
        if self.localization_source not in ("truth", "lio"):
            raise ValueError("localization_source must be 'truth' or 'lio'")
        self.lio_mode = self.localization_source == "lio"
        self.lidar_mount_preset = str(
            self.get_parameter("lidar_mount_preset").value
        ).lower()
        if self.lidar_mount_preset not in ("legacy_gimbal", "chassis"):
            raise ValueError("lidar_mount_preset must be 'legacy_gimbal' or 'chassis'")
        tilt_direction = str(
            self.get_parameter("lidar_mount_tilt_direction").value
        ).lower()
        if tilt_direction not in TILT_DIRECTIONS:
            raise ValueError(f"lidar_mount_tilt_direction must be one of {TILT_DIRECTIONS}")
        if self.lidar_mount_preset == "chassis":
            self.lidar_mount_pose = chassis_mount(
                x=float(self.get_parameter("lidar_mount_x").value),
                y=float(self.get_parameter("lidar_mount_y").value),
                height_above_ground=float(
                    self.get_parameter("lidar_mount_height").value
                ),
                tilt_direction=tilt_direction,
                tilt_deg=float(self.get_parameter("lidar_mount_tilt_deg").value),
                body_height=float(self.get_parameter("base_footprint_height").value),
            )
        else:
            self.lidar_mount_pose = legacy_gimbal_mount()

        source_model_path = self._resolve_model_path()
        self._generated_model_path = None
        self._wait_for_map_assets()
        self.model_path = self._prepare_model_path(source_model_path)
        if not self.model_path.exists() or self.model_path.stat().st_size <= 0:
            raise FileNotFoundError(f"MuJoCo scene is not ready: {self.model_path}")
        self._check_model_assets(self.model_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)

        self.sim_rate_hz = float(self.get_parameter("sim_rate_hz").value)
        self.feedback_rate_hz = float(
            self.get_parameter("feedback_rate_hz").value
        )
        command_timeout = float(self.get_parameter("command_timeout").value)
        cmd_timeout = float(self.get_parameter("cmd_timeout").value)
        self.command_timeout = command_timeout
        if command_timeout == 0.5 and cmd_timeout != 0.5:
            self.command_timeout = cmd_timeout
        self.freeze_motion = self._get_bool_parameter("freeze_motion")
        self.lidar_occlusion_enabled = self._get_bool_parameter(
            "lidar_occlusion_enabled"
        )
        if self.freeze_motion:
            self.get_logger().warn(
                "【冻结运动】freeze_motion=true：传感器保持工作，底盘运动被保持"
            )
        self.wheel_radius = float(self.get_parameter("wheel_radius").value)
        self.max_wheel_speed = max(
            0.0, float(self.get_parameter("max_wheel_speed").value)
        )
        self.max_wheel_acceleration = max(
            0.0, float(self.get_parameter("max_wheel_acceleration").value)
        )
        self.max_steer_rate = max(
            0.0, float(self.get_parameter("max_steer_rate").value)
        )
        self.emergency_stop_topic = str(
            self.get_parameter("emergency_stop_topic").value
        )
        self.swerve_telemetry_topic = str(
            self.get_parameter("swerve_telemetry_topic").value
        )
        self.gimbal_status_topic = str(
            self.get_parameter("gimbal_status_topic").value
        )
        self.yaw_authority_request_topic = str(
            self.get_parameter("yaw_authority_request_topic").value
        )
        if self.wheel_radius <= 0.0:
            raise ValueError("wheel_radius must be positive")
        self.show_viewer = self._get_bool_parameter("show_viewer")
        if not self.show_viewer:
            self.show_viewer = self._get_bool_parameter("use_viewer")
        self.viewer_rate_hz = float(self.get_parameter("viewer_rate_hz").value)
        self.truth_rate_hz = float(self.get_parameter("truth_rate_hz").value)
        self.lidar_enabled = self._get_bool_parameter("enable_lidar")
        self.model.opt.timestep = 1.0 / self.sim_rate_hz
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.map_frame_id = str(self.get_parameter("map_frame_id").value)
        self.odom_frame_id = str(self.get_parameter("odom_frame_id").value)
        self.publish_map_to_odom_tf = self._get_bool_parameter(
            "publish_map_to_odom_tf"
        )
        self.lidar_tf_frame_id = str(self.get_parameter("lidar_tf_frame_id").value)
        self.robot_base_frame_id = str(
            self.get_parameter("robot_base_frame_id").value
        )
        self.publish_robot_base_tf = self._get_bool_parameter("publish_robot_base_tf")
        self.base_footprint_frame_id = str(
            self.get_parameter("base_footprint_frame_id").value
        )
        self.base_footprint_height = float(
            self.get_parameter("base_footprint_height").value
        )
        self.base_frame_id = str(self.get_parameter("base_frame_id").value)
        self.lidar_site = str(self.get_parameter("lidar_site").value)
        self.lidar_frame_site = str(
            self.get_parameter("lidar_frame_site").value
        )
        self.tof_enabled = self._get_bool_parameter("enable_tof")
        self.left_tof_site = str(self.get_parameter("left_tof_site").value)
        self.right_tof_site = str(self.get_parameter("right_tof_site").value)
        self.left_tof_frame_id = str(
            self.get_parameter("left_tof_frame_id").value
        )
        self.right_tof_frame_id = str(
            self.get_parameter("right_tof_frame_id").value
        )
        self.pose_cmd_topic = str(self.get_parameter("pose_cmd_topic").value)
        self.reset_pose_service_topic = str(
            self.get_parameter("reset_pose_service_topic").value
        )
        if self.lio_mode:
            self._configure_lio_mode()
        if self.viewer_rate_hz <= 0.0:
            raise ValueError("viewer_rate_hz must be positive")
        if self.truth_rate_hz <= 0.0:
            raise ValueError("truth_rate_hz must be positive")

        self.mode = MODE_SWERVE
        self.control_mode = 1
        self.motion_cmd = ChassisCommand()
        self.effective_motion_cmd = ChassisCommand()
        self.last_motion_time = self._clock_now()
        self.last_speed_time = 0.0

        self.direct_speeds = {name: 0.0 for name in WHEEL_ORDER}
        self.direct_steer_angles = {name: 0.0 for name in WHEEL_ORDER}
        self.last_steer_angles = {name: 0.0 for name in WHEEL_ORDER}
        self.last_wheel_speeds = {name: 0.0 for name in WHEEL_ORDER}
        self.drive_speed_saturated = {name: False for name in WHEEL_ORDER}
        self.drive_acceleration_saturated = {name: False for name in WHEEL_ORDER}
        self.steer_rate_saturated = {name: False for name in WHEEL_ORDER}
        self.drive_speed_saturation_count = 0
        self.drive_acceleration_saturation_count = 0
        self.steer_rate_saturation_count = 0
        self.contact_violation_count = 0
        self.contact_diagnostic_count = 0
        self.max_contact_force = 0.0
        self._contact_force_buf = np.zeros(6, dtype=np.float64)
        self._last_published_illegal_count = -1
        self._contact_eval_publish_stride = 0
        self.contact_eval_stats = {
            "evaluator": "mujoco_geom_normal_v1",
            "illegal_contact_count": 0,
            "illegal_chassis_count": 0,
            "illegal_wheel_side_count": 0,
            "illegal_wheel_ceiling_count": 0,
            "illegal_obstacle_count": 0,
            "illegal_max_force_n": 0.0,
            "illegal_max_penetration_m": 0.0,
            "legal_wheel_ground_count": 0,
            "legal_wheel_drivable_deck_count": 0,
            "legal_chassis_ramp_graze_count": 0,
            "speculative_margin_count": 0,
            "illegal_samples": [],
        }
        self.telemetry_sequence = 0
        self.gimbal_status_sequence = 0
        self.gimbal_request_sequence = 0
        self.gimbal_yaw_authority = GimbalYawStatus.YAW_AUTHORITY_GIMBAL_COMPENSATED
        self.gimbal_locked = False
        self.initial_body_yaw = 0.0
        self.emergency_stop_active = False
        self.hard_stop_requested = False
        self.current_targets = [
            WheelTarget(name, 0.0, 0.0)
            for name in WHEEL_ORDER
        ]
        self.viewer = None
        self.viewer_data = None
        self.sim_lock = threading.RLock()
        self.add_on_set_parameters_callback(self._on_set_parameters)
        self.stop_event = threading.Event()
        self.sim_thread = None
        self.dynamic_obstacle_planner_thread = None
        self.lidar_process = None
        self.lidar_state_queue = None
        self.lidar_stop_event = None
        self.lidar_occlusion_event = None
        self.tof_process = None
        self.tof_state_queue = None
        self.tof_stop_event = None
        self.lidar_state_timer = None
        self.lidar_state_thread = None
        self.lidar_process_dead_warned = False
        self.tof_process_dead_warned = False
        self.last_viewer_sync_time = 0.0

        self.steer_actuator_ids = self._name_ids(STEER_ACTUATORS, "actuator")
        self.wheel_actuator_ids = self._name_ids(WHEEL_ACTUATORS, "actuator")
        if self.max_wheel_speed > 0.0:
            configure_wheel_speed_limit(
                self.model, WHEEL_ACTUATORS.values(), self.max_wheel_speed, self.wheel_radius
            )
        self.steer_joint_ids = self._name_ids(STEER_JOINTS, "joint")
        # Wheel centres of the model that was actually loaded; the kinematics must use the same
        # geometry as the physics or a commanded yaw rate is realised scaled by the radius ratio.
        self.wheel_positions = wheel_positions_from_model(self.model)
        self.wheel_joint_ids = self._name_ids(WHEEL_JOINTS, "joint")
        self.base_body_id = self._body_id(self.base_frame_id)
        self.robot_body_ids = self._descendant_body_ids(self.base_body_id)
        self.robot_geom_ids = {
            geom_id
            for geom_id in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom_id]) in self.robot_body_ids
        }
        self.wheel_geom_ids = {
            self._geom_id(f"{name}_wheel")
            for name in ("front_left", "rear_left", "front_right", "rear_right")
        }
        self.ground_geom_ids = self._ground_geom_ids()
        self.geom_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"
            for geom_id in range(self.model.ngeom)
        ]
        self.geom_body_ids = [int(self.model.geom_bodyid[geom_id]) for geom_id in range(self.model.ngeom)]
        self.lidar_site_id = self._site_id(self.lidar_site)
        self.lidar_frame_site_id = self._site_id(self.lidar_frame_site)
        self.left_tof_site_id = self._site_id(self.left_tof_site)
        self.right_tof_site_id = self._site_id(self.right_tof_site)
        self.free_joint_id = self._base_free_joint_id()
        self.free_qpos_addr = self.model.jnt_qposadr[self.free_joint_id]
        self.free_dof_addr = self.model.jnt_dofadr[self.free_joint_id]
        self.dynamic_obstacle_config = self._dynamic_obstacle_scene_config()
        self.dynamic_obstacle_rng = np.random.default_rng(17)
        self.dynamic_obstacle_sampler = self._load_dynamic_obstacle_sampler(
            self.dynamic_obstacle_config
        )
        self.dynamic_obstacle_planner = (
            self.dynamic_obstacle_sampler.coarse_planner(0.25)
            if self.dynamic_obstacle_sampler is not None
            else None
        )
        self.dynamic_obstacles = self._resolve_dynamic_obstacles()
        self._set_initial_pose_from_parameters()
        self.obstacle_fixture = ObstacleFixture(
            self.model, self.data, self._get_bool_parameter("enable_test_obstacle_fixture"))
        if self.obstacle_fixture.enabled:
            self.obstacle_fixture.set_position(self.get_parameter("test_obstacle_position").value)
        self._update_dynamic_obstacles_locked(0.0)
        mujoco.mj_forward(self.model, self.data)
        _, initial_rotation = self._body_pose_locked()
        self.initial_body_yaw = float(np.arctan2(initial_rotation[1, 0], initial_rotation[0, 0]))

        self.motion_sub = self.create_subscription(
            MotionCtrl,
            "/motion_control",
            self._motion_control_callback,
            10,
        )
        self.emergency_stop_sub = None
        if self.emergency_stop_topic:
            self.emergency_stop_sub = self.create_subscription(
                Bool,
                self.emergency_stop_topic,
                self._emergency_stop_callback,
                10,
            )
        self.yaw_authority_request_sub = self.create_subscription(
            YawAuthorityRequest,
            self.yaw_authority_request_topic,
            self._yaw_authority_request_callback,
            10,
        )
        self.speed_sub = self.create_subscription(
            SpeedCtrl,
            "/speed_ctrl",
            self._speed_control_callback,
            10,
        )
        self.steer_sub = self.create_subscription(
            SteerCtrl,
            "/steer_ctrl",
            self._steer_control_callback,
            10,
        )
        self.pose_cmd_sub = self.create_subscription(
            CarState,
            self.pose_cmd_topic,
            self._pose_cmd_callback,
            10,
        )
        self.reset_pose_service = None
        if self.reset_pose_service_topic:
            self.reset_pose_service = self.create_service(
                Trigger,
                self.reset_pose_service_topic,
                self._reset_pose_callback,
            )

        self.motion_mode_srv = self.create_service(
            MotionMode,
            "/motion_mode",
            self._motion_mode_callback,
        )
        self.control_mode_srv = self.create_service(
            ControlMode,
            "/control_mode",
            self._control_mode_callback,
        )

        self.motion_fb_pub = self.create_publisher(MotionFb, "/motion_fb", 10)
        self.speed_fb_pub = self.create_publisher(SpeedFb, "/speed_fb", 10)
        self.steer_fb_pub = self.create_publisher(SteerFb, "/steer_fb", 10)
        self.swerve_telemetry_pub = self.create_publisher(
            SwerveTelemetry,
            self.swerve_telemetry_topic,
            10,
        )
        self.contact_eval_pub = self.create_publisher(
            String,
            "/simulation/contact_evaluation",
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        self.gimbal_status_pub = self.create_publisher(
            GimbalYawStatus,
            self.gimbal_status_topic,
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        self.system_fb_pub = self.create_publisher(
            SystemstateFb,
            "/system_state_fb",
            10,
        )
        self.battery_fb_pub = self.create_publisher(
            BatteryFb,
            "/battery_fb",
            10,
        )
        self.lidar_odometry_topic = str(
            self.get_parameter("lidar_odometry_topic").value
        )
        self.lidar_odometry_pub = None
        self.localization_pub = None
        self.truth_timer = None
        if self.lio_mode:
            # Localization is not simulated: /odometry, /registered_scan,
            # /lidar_odometry and odom->gimbal_yaw_odom belong to the real
            # chain.  Only raw sensors and an evaluation-only truth go out.
            self.clock_pub = self.create_publisher(
                Clock, str(self.get_parameter("clock_topic").value), 10
            )
            self.imu_pub = self.create_publisher(
                Imu,
                str(self.get_parameter("imu_topic").value),
                QoSProfile(depth=200, reliability=ReliabilityPolicy.BEST_EFFORT),
            )
            self.ground_truth_pub = self.create_publisher(
                Odometry,
                str(self.get_parameter("ground_truth_odom_topic").value),
                10,
            )
        else:
            self.localization_pub = self.create_publisher(
                Odometry,
                self.odom_topic,
                10,
            )
            if self.lidar_odometry_topic:
                self.lidar_odometry_pub = self.create_publisher(
                    Odometry,
                    self.lidar_odometry_topic,
                    10,
                )
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        self._publish_static_transforms()
        if self.lidar_enabled or self.tof_enabled:
            self._init_lidar_process()

        self.feedback_timer = self.create_timer(
            1.0 / self.feedback_rate_hz,
            self._publish_feedback,
        )
        if not self.lio_mode:
            self.truth_timer = self.create_timer(
                1.0 / self.truth_rate_hz,
                self._publish_truth,
            )
        if self.show_viewer:
            self._start_viewer()
        self._start_dynamic_obstacle_planner_thread()
        self._start_simulation_thread()

        self.get_logger().info(
            f"【仿真模型】已加载 MuJoCo 四舵轮场景：{self.model_path}"
        )
        self.get_logger().info(
            "【运动模式】0=四轮转向，2=蟹行，4=自旋，8=用户控制，16=驻车"
        )

    def destroy_node(self):
        """Close the optional MuJoCo viewer before shutting down ROS."""
        shutdown_started = time.monotonic()
        self.get_logger().info("【退出阶段】stop requested")
        self.stop_event.set()
        if self.dynamic_obstacle_planner_thread is not None:
            self.dynamic_obstacle_planner_thread.join(timeout=1.0)
            self.dynamic_obstacle_planner_thread = None
        if self.lidar_state_timer is not None:
            self.destroy_timer(self.lidar_state_timer)
            self.lidar_state_timer = None
        if self.lidar_stop_event is not None:
            self.lidar_stop_event.set()
        if self.tof_stop_event is not None:
            self.tof_stop_event.set()
        if self.lidar_state_thread is not None:
            self.lidar_state_thread.join(timeout=1.0)
            self.lidar_state_thread = None
        child_errors = []
        self.get_logger().info("【退出阶段】joining sensor processes")
        for process in (self.lidar_process, self.tof_process):
            if process is None:
                continue
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)
            if process.exitcode != 0:
                child_errors.append(f"{process.name}: exitcode={process.exitcode}")
            self.get_logger().info(
                f"【退出阶段】sensor={process.name} exitcode={process.exitcode} "
                f"elapsed_s={time.monotonic() - shutdown_started:.3f}")
        if self.lidar_state_queue is not None:
            self.lidar_state_queue.cancel_join_thread()
            self.lidar_state_queue.close()
        if self.tof_state_queue is not None:
            self.tof_state_queue.cancel_join_thread()
            self.tof_state_queue.close()
        if self.sim_thread is not None and self.sim_thread.is_alive():
            self.sim_thread.join(timeout=1.0)
        self.get_logger().info("【退出阶段】closing viewer")
        with self.sim_lock:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
        self.get_logger().info(
            f"【退出阶段】destroy ROS node elapsed_s={time.monotonic() - shutdown_started:.3f}")
        result = super().destroy_node()
        if self._generated_model_path is not None:
            try:
                self._generated_model_path.unlink(missing_ok=True)
            except OSError as exc:
                self.get_logger().warning(f"【仿真场景】临时场景清理失败：{exc}")
            self._generated_model_path = None
        if child_errors:
            raise RuntimeError("Sensor process teardown failed: " + "; ".join(child_errors))
        return result

    # ------------------------------------------------------------------
    # LIO mode (localization_source:=lio)
    # ------------------------------------------------------------------


def main(args=None):
    """Run the ROS2 MuJoCo simulation node."""
    rclpy.init(args=args)
    node = SwerveMujocoSim()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
        finally:
            _shutdown_rclpy_if_needed()


if __name__ == "__main__":
    main()
