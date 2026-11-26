"""LIO timing/IMU/ground-truth and LiDAR/ToF worker lifecycle mixin."""

from ats_mujoco_sim.sim_sensor_math import *  # noqa: F401,F403
from ats_mujoco_sim.sim_sensor_math import (
    _mat_to_xyzw,
    _quat_wxyz_from_yaw,
    _sim_time_to_stamp,
    _timeout_clock,
    _yaw_from_quat_wxyz,
)
from ats_mujoco_sim.sim_sensor_workers import (
    _lidar_process_lio_main,
    _lidar_process_main,
    _tof_process_main,
)


class SimLioAndLidarMixin:
    """Mixin handling LIO deterministic stepping and LiDAR/ToF worker IPC."""

    @staticmethod
    def _integer_stride(sim_rate_hz, target_hz, name):
        if target_hz <= 0.0:
            raise ValueError(f"{name} must be positive")
        ratio = sim_rate_hz / target_hz
        if ratio < 1.0 - 1e-9 or abs(ratio - round(ratio)) > 1e-6:
            raise ValueError(
                f"sim_rate_hz ({sim_rate_hz}) must be an integer multiple of "
                f"{name} ({target_hz}) in localization_source:=lio"
            )
        return int(round(ratio))

    def _configure_lio_mode(self):
        logger = self.get_logger()
        if self.publish_robot_base_tf:
            logger.warn(
                "【LIO模式】publish_robot_base_tf 已强制为 false："
                "odom->gimbal_yaw_odom 由 sensor_scan_generation 发布。"
            )
            self.publish_robot_base_tf = False
        if self.publish_map_to_odom_tf:
            logger.warn(
                "【LIO模式】publish_map_to_odom_tf 已强制为 false：map->odom 由 localization_fusion 发布。"
            )
            self.publish_map_to_odom_tf = False
        if self.tof_enabled:
            logger.warn("【LIO模式】侧向 ToF 本阶段不支持仿真时间戳，已关闭。")
            self.tof_enabled = False

        self.imu_rate_hz = float(self.get_parameter("imu_rate_hz").value)
        self.ground_truth_rate_hz = float(
            self.get_parameter("ground_truth_rate_hz").value
        )
        self.imu_stride = self._integer_stride(
            self.sim_rate_hz, self.imu_rate_hz, "imu_rate_hz"
        )
        self.imu_sample_sum = np.zeros(6, dtype=np.float64)
        self.imu_sample_count = 0
        self.imu_sample_time_sum = 0.0
        self.ground_truth_stride = self._integer_stride(
            self.sim_rate_hz, self.ground_truth_rate_hz, "ground_truth_rate_hz"
        )
        self.imu_frame_id = str(self.get_parameter("imu_frame_id").value)
        self.lidar_state_queue_depth = max(
            2, int(self.get_parameter("lidar_state_queue_depth").value)
        )
        self.lidar_segment_s = float(self.get_parameter("lidar_segment_ms").value) * 1e-3
        if self.lidar_segment_s <= 0.0:
            raise ValueError("lidar_segment_ms must be positive")

        self.imu_gyro_adr = self._sensor_address(IMU_GYRO_SENSOR)
        self.imu_accel_adr = self._sensor_address(IMU_ACCEL_SENSOR)
        self.imu_noise = ImuNoiseModel(ImuNoiseConfig(
            enable_noise=self._get_bool_parameter("enable_imu_noise"),
            enable_saturation=self._get_bool_parameter("enable_imu_saturation"),
            gyro_noise_std=float(self.get_parameter("imu_gyro_noise_std").value),
            accel_noise_std=float(self.get_parameter("imu_accel_noise_std").value),
            gyro_bias_walk=float(self.get_parameter("imu_gyro_bias_walk").value),
            accel_bias_walk=float(self.get_parameter("imu_accel_bias_walk").value),
            seed=int(self.get_parameter("noise_seed").value),
        ))

        self.lio_step_index = 0
        self.lio_outbox = []
        self.lio_backpressure_s = 0.0
        self.lio_report_wall = None
        self.lio_report_sim = 0.0
        self.lio_report_backpressure = 0.0
        logger.info(
            f"【LIO模式】已启用：IMU {self.imu_rate_hz:g} Hz（每 {self.imu_stride} 步），"
            f"真值 {self.ground_truth_rate_hz:g} Hz，仿真 {self.sim_rate_hz:g} Hz，"
            f"IMU噪声={self.imu_noise.config.enable_noise}，"
            f"IMU饱和={self.imu_noise.config.enable_saturation}"
        )

    def _sensor_address(self, sensor_name):
        sensor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name
        )
        if sensor_id < 0:
            raise ValueError(
                f"localization_source:=lio needs sensor '{sensor_name}' in the model "
                "(use rmuc_2027_swerve.xml)"
            )
        return int(self.model.sensor_adr[sensor_id])

    def _clock_now(self):
        """Timeout clock: simulation time in LIO mode, monotonic wall time otherwise.

        With a slowed simulation the wall interval between commands grows by the
        slowdown factor, so command freshness must be judged in simulation time.
        """
        return _timeout_clock(self)

    def _stamp_now(self):
        if self.lio_mode:
            return _sim_time_to_stamp(float(self.data.time))
        return self.get_clock().now().to_msg()

    def _lio_capture_before_step_locked(self):
        sample = {
            "time": float(self.data.time),
            "index": self.lio_step_index,
        }
        if self.lidar_state_queue is not None:
            sample["state"] = (
                sample["time"],
                np.array(self.data.qpos, dtype=np.float64, copy=True),
                np.array(self.data.qvel, dtype=np.float64, copy=True),
            )
        return sample

    def _lio_collect_after_step_locked(self, sample):
        # MuJoCo computes sensordata and xpos at the start of mj_step, so they
        # describe the pre-step instant sample["time"], the same instant as the
        # captured qpos.
        index = sample["index"]
        # Keep every physics observation when reducing the sensor rate. Point
        # sampling aliases contact impulses (400 -> 200 Hz can integrate a
        # stationary robot into fictitious motion). The published value is the
        # mean over the complete aperture, stamped at its mean observation time.
        self.imu_sample_sum[:3] += self.data.sensordata[
            self.imu_gyro_adr:self.imu_gyro_adr + 3]
        self.imu_sample_sum[3:] += self.data.sensordata[
            self.imu_accel_adr:self.imu_accel_adr + 3]
        self.imu_sample_count += 1
        self.imu_sample_time_sum += sample["time"]
        if self.imu_sample_count == self.imu_stride:
            mean = self.imu_sample_sum / self.imu_sample_count
            sample["imu_time"] = self.imu_sample_time_sum / self.imu_sample_count
            sample["imu"] = (
                mean[:3].copy(), mean[3:] / 9.81,
            )
            self.imu_sample_sum.fill(0.0)
            self.imu_sample_count = 0
            self.imu_sample_time_sum = 0.0
        if index % self.ground_truth_stride == 0:
            base_pos, base_mat = self._body_pose_locked()
            linear, angular = self._body_velocity_locked()
            sample["truth"] = (base_pos, _mat_to_xyzw(base_mat), linear, angular)
        sample["clock"] = float(self.data.time)
        self.lio_step_index += 1
        self.lio_outbox.append(sample)

    def _lio_flush_outbox(self):
        """Publish what the last step produced, outside the simulation lock."""
        outbox, self.lio_outbox = self.lio_outbox, []
        for sample in outbox:
            if "imu" in sample:
                self._publish_imu(sample["imu_time"], *sample["imu"])
            if "truth" in sample:
                self._publish_ground_truth(sample["time"], *sample["truth"])
            # /clock runs one step ahead of the sample it follows, so a
            # consumer never sees a sensor stamp from the future.
            self.clock_pub.publish(Clock(clock=_sim_time_to_stamp(sample["clock"])))
            state = sample.get("state")
            if state is not None:
                self._lio_push_lidar_state(state)
        if outbox:
            self._lio_report(outbox[-1]["clock"])

    def _lio_push_lidar_state(self, state):
        state_queue = self.lidar_state_queue
        process = self.lidar_process
        if state_queue is None or process is None:
            return
        blocked_since = None
        while not self.stop_event.is_set():
            try:
                state_queue.put(state, timeout=0.05)
                break
            except queue.Full:
                if process.exitcode is not None:
                    break
                if blocked_since is None:
                    blocked_since = time.perf_counter()
        if blocked_since is not None:
            self.lio_backpressure_s += time.perf_counter() - blocked_since

    def _lio_report(self, sim_time):
        now = time.perf_counter()
        if self.lio_report_wall is None:
            self.lio_report_wall = now
            self.lio_report_sim = sim_time
            self.lio_report_backpressure = self.lio_backpressure_s
            return
        wall = now - self.lio_report_wall
        if wall < 10.0:
            return
        blocked = self.lio_backpressure_s - self.lio_report_backpressure
        self.get_logger().info(
            f"【LIO模式】仿真/墙钟={(sim_time - self.lio_report_sim) / wall:.3f}，"
            f"雷达背压占比={blocked / wall:.2%}，仿真时间={sim_time:.1f} s"
        )
        self.lio_report_wall = now
        self.lio_report_sim = sim_time
        self.lio_report_backpressure = self.lio_backpressure_s

    def _publish_imu(self, sim_time, gyro, accel_g):
        if self.imu_noise.active:
            gyro, accel_g = self.imu_noise.apply(gyro, accel_g, 1.0 / self.imu_rate_hz)
        msg = Imu()
        msg.header.stamp = _sim_time_to_stamp(sim_time)
        msg.header.frame_id = self.imu_frame_id
        # No orientation estimate: -1 in the first covariance entry.
        msg.orientation_covariance[0] = -1.0
        msg.angular_velocity.x = float(gyro[0])
        msg.angular_velocity.y = float(gyro[1])
        msg.angular_velocity.z = float(gyro[2])
        # Units of g, matching the real Mid360 stream (Point-LIO acc_norm 1.0).
        msg.linear_acceleration.x = float(accel_g[0])
        msg.linear_acceleration.y = float(accel_g[1])
        msg.linear_acceleration.z = float(accel_g[2])
        self.imu_pub.publish(msg)

    def _publish_ground_truth(self, sim_time, base_pos, base_quat, linear, angular):
        msg = Odometry()
        msg.header.stamp = _sim_time_to_stamp(sim_time)
        # World frame, not the Point-LIO odom frame: the start pose is the
        # map->odom seed, so evaluation aligns the two with that known pose.
        msg.header.frame_id = self.map_frame_id
        msg.child_frame_id = self.robot_base_frame_id
        msg.pose.pose.position.x = float(base_pos[0])
        msg.pose.pose.position.y = float(base_pos[1])
        msg.pose.pose.position.z = float(base_pos[2])
        msg.pose.pose.orientation.x = base_quat[0]
        msg.pose.pose.orientation.y = base_quat[1]
        msg.pose.pose.orientation.z = base_quat[2]
        msg.pose.pose.orientation.w = base_quat[3]
        msg.twist.twist.linear.x = float(linear[0])
        msg.twist.twist.linear.y = float(linear[1])
        msg.twist.twist.linear.z = float(linear[2])
        msg.twist.twist.angular.x = float(angular[0])
        msg.twist.twist.angular.y = float(angular[1])
        msg.twist.twist.angular.z = float(angular[2])
        self.ground_truth_pub.publish(msg)


    def _init_lidar_process(self):
        self.lidar_topic = str(self.get_parameter("lidar_topic").value)
        self.lidar_frame_id = str(self.get_parameter("lidar_frame_id").value)
        if not self.lidar_frame_id:
            self.lidar_frame_id = self.lidar_tf_frame_id
        self.registered_scan_topic = str(
            self.get_parameter("registered_scan_topic").value
        )
        self.registered_scan_frame_id = str(
            self.get_parameter("registered_scan_frame_id").value
        )
        if not self.registered_scan_frame_id:
            self.registered_scan_frame_id = self.odom_frame_id
        self.lidar_backend = str(self.get_parameter("lidar_backend").value)
        self.lidar_model = str(self.get_parameter("lidar_model").value).lower()
        self.lidar_downsample = int(self.get_parameter("lidar_downsample").value)
        self.tof_backend = str(self.get_parameter("tof_backend").value)
        self.lidar_rate_hz = float(self.get_parameter("lidar_rate_hz").value)
        self.lidar_rate_clock = str(self.get_parameter("lidar_rate_clock").value)
        self.lidar_state_rate_hz = float(
            self.get_parameter("lidar_state_rate_hz").value
        )
        self.tof_rate_hz = float(self.get_parameter("tof_rate_hz").value)
        if self.lidar_rate_hz <= 0.0:
            raise ValueError("lidar_rate_hz must be positive")
        if self.lidar_rate_clock not in ("wall", "sim"):
            raise ValueError("lidar_rate_clock must be 'wall' or 'sim'")
        if self.lidar_state_rate_hz < 0.0:
            raise ValueError("lidar_state_rate_hz must be non-negative")
        if self.lidar_model not in ("mid360", "livox_mid360"):
            raise ValueError("lidar_model must be 'mid360'")
        lidar_threads = resolve_thread_count(int(self.get_parameter("lidar_threads").value))
        self.lidar_downsample = resolve_downsample(self.lidar_downsample, lidar_threads)
        lidar_cpu_ids = select_lidar_cpu_ids(available_cpu_ids(), lidar_threads)
        self.get_logger().info(
            f"【仿真CPU预算】LiDAR线程={lidar_threads}，CPU={lidar_cpu_ids}，"
            f"downsample={self.lidar_downsample}；物理/导航保留其余调度资源"
        )
        if self.tof_rate_hz <= 0.0:
            raise ValueError("tof_rate_hz must be positive")
        tof_range = float(self.get_parameter("tof_range").value)
        tof_min_range = float(self.get_parameter("tof_min_range").value)
        tof_width = int(self.get_parameter("tof_width").value)
        tof_height = int(self.get_parameter("tof_height").value)
        tof_horizontal_fov_deg = float(
            self.get_parameter("tof_horizontal_fov_deg").value
        )
        tof_vertical_fov_deg = float(
            self.get_parameter("tof_vertical_fov_deg").value
        )
        tof_footprint_length = float(
            self.get_parameter("tof_footprint_length").value
        )
        tof_footprint_width = float(
            self.get_parameter("tof_footprint_width").value
        )
        tof_footprint_expand = float(
            self.get_parameter("tof_footprint_expand").value
        )
        tof_footprint_resolution = float(
            self.get_parameter("tof_footprint_resolution").value
        )
        tof_footprint_z_min = float(
            self.get_parameter("tof_footprint_z_min").value
        )
        tof_footprint_z_max = float(
            self.get_parameter("tof_footprint_z_max").value
        )
        if tof_range <= 0.0:
            raise ValueError("tof_range must be positive")
        if tof_min_range < 0.0:
            raise ValueError("tof_min_range must be non-negative")
        if tof_min_range >= tof_range:
            raise ValueError("tof_min_range must be smaller than tof_range")
        if tof_width <= 0 or tof_height <= 0:
            raise ValueError("tof_width and tof_height must be positive")
        if tof_horizontal_fov_deg <= 0.0 or tof_vertical_fov_deg <= 0.0:
            raise ValueError("ToF field-of-view values must be positive")
        if tof_footprint_length <= 0.0 or tof_footprint_width <= 0.0:
            raise ValueError("ToF footprint size values must be positive")
        if tof_footprint_expand < 0.0:
            raise ValueError("tof_footprint_expand must be non-negative")
        if tof_footprint_resolution <= 0.0:
            raise ValueError("tof_footprint_resolution must be positive")
        if tof_footprint_z_max <= tof_footprint_z_min:
            raise ValueError("tof_footprint_z_max must be greater than z_min")

        config = {
            "lidar_self_exclusion": self._get_bool_parameter("lidar_self_exclusion"),
            "lidar_max_range": float(self.get_parameter("lidar_max_range").value),
            "lidar_segment_s": float(self.get_parameter("lidar_segment_ms").value) * 1e-3,
            "lidar_threads": lidar_threads,
            "lidar_cpu_ids": lidar_cpu_ids,
            "livox_points_topic": str(self.get_parameter("livox_points_topic").value),
            "lidar_stats_topic": str(self.get_parameter("lidar_stats_topic").value),
            "enable_range_noise": self._get_bool_parameter("enable_range_noise"),
            "range_noise_sigma_m": float(self.get_parameter("range_noise_sigma_m").value),
            "noise_seed": int(self.get_parameter("noise_seed").value),
            "model_path": str(self.model_path),
            "lidar_site": self.lidar_site,
            "lidar_frame_site": self.lidar_frame_site,
            "lidar_topic": self.lidar_topic,
            "lidar_frame_id": self.lidar_frame_id,
            "registered_scan_topic": self.registered_scan_topic,
            "registered_scan_frame_id": self.registered_scan_frame_id,
            "lidar_backend": self.lidar_backend,
            "lidar_model": self.lidar_model,
            "lidar_downsample": self.lidar_downsample,
            "lidar_rate_hz": self.lidar_rate_hz,
            "lidar_rate_clock": self.lidar_rate_clock,
            "lidar_state_rate_hz": self.lidar_state_rate_hz,
            "base_frame_id": self.base_frame_id,
            "lidar_enabled": self.lidar_enabled,
            "tof_enabled": self.tof_enabled,
            "tof_backend": self.tof_backend,
            "tof_range": tof_range,
            "tof_min_range": tof_min_range,
            "tof_rate_hz": self.tof_rate_hz,
            "tof_width": tof_width,
            "tof_height": tof_height,
            "tof_horizontal_fov_deg": tof_horizontal_fov_deg,
            "tof_vertical_fov_deg": tof_vertical_fov_deg,
            "tof_footprint_length": tof_footprint_length,
            "tof_footprint_width": tof_footprint_width,
            "tof_footprint_expand": tof_footprint_expand,
            "tof_footprint_resolution": tof_footprint_resolution,
            "tof_footprint_z_min": tof_footprint_z_min,
            "tof_footprint_z_max": tof_footprint_z_max,
            "merged_tof_topic": str(
                self.get_parameter("merged_tof_topic").value
            ),
            "tof_sensors": [
                {
                    "side": "left",
                    "site": self.left_tof_site,
                    "topic": str(self.get_parameter("left_tof_topic").value),
                    "frame_id": self.left_tof_frame_id,
                },
                {
                    "side": "right",
                    "site": self.right_tof_site,
                    "topic": str(self.get_parameter("right_tof_topic").value),
                    "frame_id": self.right_tof_frame_id,
                },
            ],
        }

        # Taichi/CUDA 后端要求在它所属进程的主线程里初始化。
        # 主进程只发送 MuJoCo 状态；LiDAR 和 ToF 分进程发布，避免 CPU ToF 慢帧拖住 GPU LiDAR。
        ctx = mp.get_context("spawn")
        if self.lidar_enabled:
            # LIO blocks physics at the bounded buffer instead of dropping
            # observations; truth mode keeps the latest-state-only queue.
            self.lidar_state_queue = ctx.Queue(
                maxsize=self.lidar_state_queue_depth if self.lio_mode else 1
            )
            self.lidar_stop_event = ctx.Event()
            self.lidar_occlusion_event = ctx.Event()
            if self.lidar_occlusion_enabled:
                self.lidar_occlusion_event.set()
            self.lidar_process = ctx.Process(
                target=_lidar_process_lio_main if self.lio_mode else _lidar_process_main,
                args=(
                    config,
                    self.lidar_state_queue,
                    self.lidar_stop_event,
                    self.lidar_occlusion_event,
                ),
                name="swerve_lidar_process",
                daemon=True,
            )
            self.lidar_process.start()
        if self.tof_enabled:
            self.tof_state_queue = ctx.Queue(maxsize=1)
            self.tof_stop_event = ctx.Event()
            self.tof_process = ctx.Process(
                target=_tof_process_main,
                args=(config, self.tof_state_queue, self.tof_stop_event),
                name="swerve_tof_process",
                daemon=True,
            )
            self.tof_process.start()
        sensor_state_rate_hz = 0.0
        if self.lidar_enabled:
            sensor_state_rate_hz = max(sensor_state_rate_hz, self.lidar_rate_hz)
        if self.tof_enabled:
            sensor_state_rate_hz = max(sensor_state_rate_hz, self.tof_rate_hz)
        state_rate_hz = (
            self.lidar_state_rate_hz
            if self.lidar_state_rate_hz > 0.0
            else sensor_state_rate_hz * 2.0
        )
        self.lidar_state_rate_hz = state_rate_hz
        if not self.lio_mode:
            # LIO mode pushes every physics step from the simulation thread so
            # the state buffer is dense and carries back-pressure.
            self.lidar_state_thread = threading.Thread(
                target=self._lidar_state_loop,
                name="swerve_lidar_state_sender",
                daemon=True,
            )
            self.lidar_state_thread.start()
        if self.lidar_enabled and self.lio_mode:
            self.get_logger().info(
                f"【LiDAR·LIO】进程已启用：频率={self.lidar_rate_hz} Hz，"
                f"降采样={self.lidar_downsample}，分段={self.lidar_segment_s * 1e3:.1f} ms，"
                f"状态缓冲={self.lidar_state_queue_depth} 步，"
                f"点云话题={config['livox_points_topic']}，坐标系={self.lidar_frame_id}"
            )
        elif self.lidar_enabled:
            self.get_logger().info(
                f"【LiDAR】进程已启用：降采样={self.lidar_downsample}，"
                f"频率={self.lidar_rate_hz} Hz，"
                f"时钟={self.lidar_rate_clock}，状态频率={self.lidar_state_rate_hz} Hz，"
                f"站点={self.lidar_site}，参考站点={self.lidar_frame_site}，"
                f"话题={self.lidar_topic}，坐标系={self.lidar_frame_id}，后端={self.lidar_backend}"
            )
        if self.tof_enabled:
            left_topic = str(self.get_parameter("left_tof_topic").value)
            right_topic = str(self.get_parameter("right_tof_topic").value)
            self.get_logger().info(
                "【侧向ToF】已启用："
                f"量程={tof_min_range}-{tof_range} m，"
                f"分辨率={tof_width}x{tof_height}，"
                f"足迹={tof_footprint_length}x{tof_footprint_width} m，"
                f"扩展=+{tof_footprint_expand} m，分辨率={tof_footprint_resolution} m，"
                f"高度={tof_footprint_z_min}-{tof_footprint_z_max} m，"
                f"左侧话题={left_topic}，右侧话题={right_topic}"
            )

    def _send_lidar_state(self):
        if (
            self.lidar_process is not None
            and self.lidar_process.exitcode is not None
        ):
            if not self.lidar_process_dead_warned:
                self.get_logger().error(
                    "【LiDAR进程】已退出，退出码="
                    f"{self.lidar_process.exitcode}"
                )
                self.lidar_process_dead_warned = True
        if (
            self.tof_process is not None
            and self.tof_process.exitcode is not None
        ):
            if not self.tof_process_dead_warned:
                self.get_logger().error(
                    "【ToF进程】已退出，退出码="
                    f"{self.tof_process.exitcode}"
                )
                self.tof_process_dead_warned = True

        queues = []
        if (
            self.lidar_process is not None
            and self.lidar_process.exitcode is None
            and self.lidar_state_queue is not None
        ):
            queues.append(self.lidar_state_queue)
        if (
            self.tof_process is not None
            and self.tof_process.exitcode is None
            and self.tof_state_queue is not None
        ):
            queues.append(self.tof_state_queue)
        if not queues:
            return

        with self.sim_lock:
            sim_time = float(self.data.time)
            qpos = np.array(self.data.qpos, dtype=np.float64, copy=True)
            qvel = np.array(self.data.qvel, dtype=np.float64, copy=True)

        # 每个传感器队列只保留最新一帧，避免计算慢时越积越多拖住控制。
        for state_queue in queues:
            try:
                while True:
                    state_queue.get_nowait()
            except queue.Empty:
                pass

            try:
                state_queue.put_nowait((sim_time, qpos, qvel))
            except queue.Full:
                pass

    def _lidar_state_loop(self):
        period = 1.0 / max(1.0, float(self.lidar_state_rate_hz))
        while rclpy.ok() and not self.stop_event.is_set():
            start = time.perf_counter()
            self._send_lidar_state()
            elapsed = time.perf_counter() - start
            sleep_time = period - elapsed
            if sleep_time > 0.0:
                self.stop_event.wait(sleep_time)
