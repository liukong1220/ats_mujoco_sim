"""Model asset resolution, viewer startup, pose reset, and dynamic obstacle mixin."""

from ats_mujoco_sim.sim_sensor_math import *  # noqa: F401,F403
from ats_mujoco_sim.sim_sensor_math import (
    _quat_wxyz_from_yaw,
    _refresh_viewer_data,
    _timeout_clock,
    _yaw_from_quat_wxyz,
)


class SimAssetsAndObstaclesMixin:
    """Mixin handling scene/model assets, viewer, pose resets, and dynamic obstacles."""

    def _get_bool_parameter(self, name):
        value = self.get_parameter(name).value
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes", "on")
        return bool(value)

    def _start_viewer(self):
        """Open the MuJoCo window without ever blocking start-up.

        launch_passive() returns only after the compositor has mapped the window. With a locked
        Wayland session that never happens (the pip GLFW has no X11 backend here), and the node
        used to sit in this call before it published /clock, so the whole stack timed out
        (2026-10-25). The window is opened from a helper thread: the simulation runs without it
        until it exists, and the window appears when the desktop is unlocked.
        """
        def open_viewer():
            try:
                import mujoco.viewer

                shadow = mujoco.MjData(self.model)
                mujoco.mj_resetData(self.model, shadow)   # reads the model only; the first sync brings the real state
                viewer = mujoco.viewer.launch_passive(
                    self.model,
                    shadow,
                    show_left_ui=True,
                    show_right_ui=True,
                )
                viewer.opt.geomgroup[3] = 1
                self.viewer_data = shadow
                self.viewer = viewer
                self.get_logger().info("【MuJoCo查看器】已启动")
            except Exception as exc:
                self.viewer = None
                self.get_logger().error(f"【MuJoCo查看器】启动失败：{exc}")

        self.viewer = None
        self.viewer_data = None
        self._viewer_open_thread = threading.Thread(
            target=open_viewer, name="mujoco-viewer-open", daemon=True)
        self._viewer_open_thread.start()
        self.get_logger().info(
            "【MuJoCo查看器】正在后台打开窗口；桌面锁屏时窗口要等到解锁才会出现，仿真不受影响")

    def _resolve_model_path(self):
        model_path = str(self.get_parameter("model_path").value).strip()
        if model_path:
            return Path(model_path).expanduser()

        scene_file = str(self.get_parameter("scene_file").value).strip()
        if scene_file:
            return Path(scene_file).expanduser()

        manifest_scene = self._read_manifest_value("scene_file")
        if manifest_scene:
            return Path(manifest_scene).expanduser()

        map_dir = str(self.get_parameter("map_dir").value).strip()
        if map_dir:
            return Path(map_dir).expanduser() / "model" / "swerve.xml"

        try:
            from ament_index_python.packages import get_package_share_directory

            share_dir = Path(get_package_share_directory("ats_mujoco_sim"))
            return share_dir / "models" / "swerve_chassis.xml"
        except Exception:
            package_dir = Path(__file__).resolve().parents[1]
            return package_dir / "models" / "swerve_chassis.xml"

    def _prepare_model_path(self, source_model_path):
        """按启动参数生成临时 MuJoCo 场景，避免修改受版本控制的模型。"""
        source_model_path = Path(source_model_path).expanduser()
        remove_walls = not self._get_bool_parameter("enable_static_walls")
        apply_mount = self.lidar_mount_preset == "chassis"
        if not remove_walls and not apply_mount:
            return source_model_path

        xml_text = source_model_path.read_text(encoding="utf-8")
        include_line = '<include file="rmuc_2025_wall_boxes.xml"/>'
        walls_present = remove_walls and include_line in xml_text
        if not walls_present and not apply_mount:
            return source_model_path
        if walls_present:
            xml_text = xml_text.replace(
                include_line,
                "<!-- 已按 enable_static_walls=false 移除二维地图静态墙体；保留 heightfield。 -->",
            )
        if apply_mount:
            xml_text = patch_lidar_mount(xml_text, self.lidar_mount_pose)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".xml",
            prefix=".ats_mujoco_no_walls_",
            dir=source_model_path.parent,
            delete=False,
        ) as generated:
            generated.write(xml_text)
            self._generated_model_path = Path(generated.name)
        self.get_logger().info(
            "【仿真场景】enable_static_walls=false：已移除二维地图静态墙体，"
            "保留 MuJoCo heightfield 地形；规划地图仍由 /map 独立提供。"
        )
        return self._generated_model_path

    def _check_model_assets(self, model_path):
        xml_text = model_path.read_text(encoding="utf-8")
        if MID360_MESH_RELATIVE_PATH not in xml_text:
            return

        relative_mesh_path = (
            model_path.parent / MID360_MESH_RELATIVE_PATH
        ).expanduser().resolve()
        if not relative_mesh_path.exists():
            raise FileNotFoundError(
                f"MuJoCo MID360 mesh is missing: {relative_mesh_path}"
            )

    def _manifest_path(self):
        manifest_path = str(self.get_parameter("map_manifest_path").value).strip()
        if manifest_path:
            return Path(manifest_path).expanduser()

        map_dir = str(self.get_parameter("map_dir").value).strip()
        if map_dir:
            return Path(map_dir).expanduser() / "map_manifest.yaml"
        return None

    def _read_manifest_value(self, key):
        manifest_path = self._manifest_path()
        if manifest_path is None or not manifest_path.exists():
            return ""

        prefix = f"{key}:"
        for raw_line in manifest_path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines():
            line = raw_line.strip()
            if not line.startswith(prefix):
                continue
            value = line.split(":", 1)[1].strip()
            if len(value) >= 2 and value[0] in ("'", '"') and value[-1] == value[0]:
                value = value[1:-1]
            return value
        return ""

    def _manifest_bool(self, key, default):
        value = self._read_manifest_value(key)
        if value == "":
            return default
        return bool_from_value(value)

    def _manifest_int(self, key, default):
        value = self._read_manifest_value(key)
        if value == "":
            return default
        try:
            return int(value)
        except ValueError:
            return default

    def _manifest_float(self, key, default):
        value = self._read_manifest_value(key)
        if value == "":
            return default
        try:
            return float(value)
        except ValueError:
            return default

    def _wait_for_map_assets(self):
        ready_file_text = str(self.get_parameter("map_ready_file").value).strip()
        if not ready_file_text:
            ready_file_text = self._read_manifest_value("ready_file")
        if not ready_file_text:
            map_dir = str(self.get_parameter("map_dir").value).strip()
            if map_dir:
                ready_file_text = str(Path(map_dir).expanduser() / "assets.ready")

        ready_token = str(self.get_parameter("map_ready_token").value).strip()
        if not ready_token:
            ready_token = self._read_manifest_value("ready_token")
        timeout_sec = float(self.get_parameter("map_wait_timeout_sec").value)
        if not ready_file_text or timeout_sec <= 0.0:
            return

        ready_file = Path(ready_file_text).expanduser()
        start_time = time.monotonic()
        while rclpy.ok():
            if ready_file.exists():
                text = ready_file.read_text(encoding="utf-8", errors="ignore")
                if not ready_token or f"ready_token: {ready_token}" in text:
                    return
            if time.monotonic() - start_time >= timeout_sec:
                raise TimeoutError(f"Timed out waiting for map assets: {ready_file}")
            time.sleep(0.05)

    def _name_ids(self, names, object_type):
        if object_type == "actuator":
            mj_type = mujoco.mjtObj.mjOBJ_ACTUATOR
        else:
            mj_type = mujoco.mjtObj.mjOBJ_JOINT

        ids = {}
        for key, name in names.items():
            object_id = mujoco.mj_name2id(self.model, mj_type, name)
            if object_id < 0:
                raise RuntimeError(f"MuJoCo {object_type} not found: {name}")
            ids[key] = object_id
        return ids

    def _body_id(self, name):
        body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            name,
        )
        if body_id < 0:
            raise RuntimeError(f"MuJoCo body not found: {name}")
        return body_id

    def _geom_id(self, name):
        geom_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            name,
        )
        if geom_id < 0:
            raise RuntimeError(f"MuJoCo geom not found: {name}")
        return geom_id

    def _descendant_body_ids(self, root_body_id):
        descendants = set()
        for body_id in range(self.model.nbody):
            current = body_id
            while current > 0 and current != root_body_id:
                current = int(self.model.body_parentid[current])
            if current == root_body_id:
                descendants.add(body_id)
        return descendants

    def _ground_geom_ids(self):
        ground_ids = set()
        for name in (
            "floor", "terrain", "rmuc_2025_field", "rmuc_2026_field", "rmuc_2027_field",
        ):
            geom_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                name,
            )
            if geom_id >= 0:
                ground_ids.add(geom_id)
        return ground_ids

    def _site_id(self, name):
        site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            name,
        )
        if site_id < 0:
            raise RuntimeError(f"MuJoCo site not found: {name}")
        return site_id

    def _base_free_joint_id(self):
        jntadr = self.model.body_jntadr[self.base_body_id]
        jntnum = self.model.body_jntnum[self.base_body_id]
        for joint_id in range(jntadr, jntadr + jntnum):
            if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                return joint_id
        raise RuntimeError(f"{self.base_frame_id} has no freejoint")

    def _dynamic_obstacle_scene_config(self):
        return DynamicObstacleSceneConfig(
            enabled=self._manifest_bool(
                "dynamic_obstacles_enabled",
                self._get_bool_parameter("dynamic_obstacles_enabled"),
            ),
            count=self._manifest_int(
                "dynamic_obstacle_count",
                int(self.get_parameter("dynamic_obstacle_count").value),
            ),
            shape=self._read_manifest_value("dynamic_obstacle_shape")
            or str(self.get_parameter("dynamic_obstacle_shape").value),
            radius=self._manifest_float(
                "dynamic_obstacle_radius",
                float(self.get_parameter("dynamic_obstacle_radius").value),
            ),
            height=self._manifest_float(
                "dynamic_obstacle_height",
                float(self.get_parameter("dynamic_obstacle_height").value),
            ),
            mass=self._manifest_float(
                "dynamic_obstacle_mass",
                float(self.get_parameter("dynamic_obstacle_mass").value),
            ),
            speed=self._manifest_float(
                "dynamic_obstacle_speed",
                float(self.get_parameter("dynamic_obstacle_speed").value),
            ),
            path_length=self._manifest_float(
                "dynamic_obstacle_path_length",
                float(self.get_parameter("dynamic_obstacle_path_length").value),
            ),
            min_robot_distance=self._manifest_float(
                "dynamic_obstacle_min_robot_distance",
                float(self.get_parameter("dynamic_obstacle_min_robot_distance").value),
            ),
            near_robot_radius=self._manifest_float(
                "dynamic_obstacle_near_robot_radius",
                float(self.get_parameter("dynamic_obstacle_near_robot_radius").value),
            ),
            map_clearance=self._manifest_float(
                "dynamic_obstacle_map_clearance",
                float(self.get_parameter("dynamic_obstacle_map_clearance").value),
            ),
            replan_rate_hz=self._manifest_float(
                "dynamic_obstacle_replan_rate_hz",
                float(self.get_parameter("dynamic_obstacle_replan_rate_hz").value),
            ),
        )

    def _map_bounds_from_manifest(self):
        x_lower = self._manifest_float("pcd_map_x_lower", -5.0)
        x_upper = self._manifest_float("pcd_map_x_upper", 5.0)
        y_lower = self._manifest_float("pcd_map_y_lower", -5.0)
        y_upper = self._manifest_float("pcd_map_y_upper", 5.0)
        if x_upper <= x_lower:
            x_lower, x_upper = -5.0, 5.0
        if y_upper <= y_lower:
            y_lower, y_upper = -5.0, 5.0
        map_half_x = 0.5 * (x_upper - x_lower)
        map_half_y = 0.5 * (y_upper - y_lower)
        map_center_x = x_lower + map_half_x
        map_center_y = y_lower + map_half_y
        return map_center_x, map_center_y, map_half_x, map_half_y

    def _resolve_dynamic_obstacles(self):
        config = self.dynamic_obstacle_config
        if not config.enabled or config.count <= 0:
            return []

        runtime_configs = build_runtime_configs(
            config,
            *self._map_bounds_from_manifest(),
        )
        obstacles = []
        for index, runtime in enumerate(runtime_configs):
            joint_name = f"{runtime.name}_joint"
            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            if joint_id < 0:
                self.get_logger().warn(
                    f"【动态障碍】模型中找不到关节：{joint_name}"
                )
                continue
            if self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
                self.get_logger().warn(
                    f"【动态障碍】关节不是自由关节：{joint_name}"
                )
                continue
            obstacles.append({
                "config": runtime,
                "qpos_addr": int(self.model.jnt_qposadr[joint_id]),
                "dof_addr": int(self.model.jnt_dofadr[joint_id]),
                "state": DynamicObstacleControllerState(
                    target=np.array(runtime.center[:2], dtype=np.float64),
                    last_replan_time=staggered_replan_time(
                        index,
                        len(runtime_configs),
                        config.replan_rate_hz,
                    ),
                ),
                "plan_lock": threading.Lock(),
            })

        if obstacles:
            self.get_logger().info(
                f"【动态障碍】已启用：{len(obstacles)} 个运行时驱动刚体"
            )
        return obstacles

    def _load_dynamic_obstacle_sampler(self, config):
        map_image = self._read_manifest_value("global_map_image")
        map_yaml = self._read_manifest_value("global_map_yaml")
        if not map_image:
            map_image = self._read_manifest_value("map_image")
        if not map_yaml:
            map_yaml = self._read_manifest_value("map_yaml")
        if not map_image or not map_yaml:
            return None

        try:
            resolution, origin = read_map_metadata(Path(map_yaml).expanduser())
            occupancy = load_occupancy_image(
                Path(map_image).expanduser(),
                self._manifest_int("occupied_pixel_threshold", 230),
            )
            return StaticMapSampler(
                occupancy=occupancy,
                resolution=resolution,
                origin=origin,
                clearance=config.map_clearance,
            )
        except Exception as exc:
            self.get_logger().warn(
                f"【动态障碍】地图采样器已停用：{exc}"
            )
            return None

    def _set_initial_pose_from_parameters(self):
        position = np.array(
            [
                float(self.get_parameter("start_x").value),
                float(self.get_parameter("start_y").value),
                float(self.get_parameter("start_z").value),
            ],
            dtype=np.float64,
        )
        yaw = float(self.get_parameter("start_yaw").value)
        self.reset_pose_position = position.copy()
        self.reset_pose_yaw = yaw
        self._reset_pose_locked(
            position, yaw, float(self.get_parameter("start_roll").value),
            float(self.get_parameter("start_pitch").value))

    def _reset_pose_locked(self, position, yaw, roll=0.0, pitch=0.0):
        """Reset only the chassis state while preserving time and diagnostics."""
        position = np.asarray(position, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("reset pose position must contain three finite values")
        if not np.isfinite(yaw):
            raise ValueError("reset pose yaw must be finite")

        orientation = quaternion_from_rpy(float(roll), float(pitch), float(yaw))
        self.data.qpos[self.free_qpos_addr:self.free_qpos_addr + 3] = position
        self.data.qpos[self.free_qpos_addr + 3:self.free_qpos_addr + 7] = orientation
        self.data.qvel[self.free_dof_addr:self.free_dof_addr + 6] = 0.0
        for name in WHEEL_ORDER:
            steer_dof = self.model.jnt_dofadr[self.steer_joint_ids[name]]
            wheel_dof = self.model.jnt_dofadr[self.wheel_joint_ids[name]]
            self.data.qvel[steer_dof] = 0.0
            self.data.qvel[wheel_dof] = 0.0
            self.data.ctrl[self.steer_actuator_ids[name]] = 0.0
            self.data.ctrl[self.wheel_actuator_ids[name]] = 0.0

        self.motion_cmd = ChassisCommand()
        self.effective_motion_cmd = ChassisCommand()
        self.direct_speeds = {name: 0.0 for name in WHEEL_ORDER}
        self.direct_steer_angles = {name: 0.0 for name in WHEEL_ORDER}
        self.steer_recenter_targets = {name: None for name in WHEEL_ORDER}
        self.last_wheel_speeds = {name: 0.0 for name in WHEEL_ORDER}
        self.last_motion_time = _timeout_clock(self)
        self.last_speed_time = 0.0
        self.hard_stop_requested = self.freeze_motion or self.emergency_stop_active
        mujoco.mj_forward(self.model, self.data)
        self.last_steer_angles = {
            name: float(
                self.data.qpos[self.model.jnt_qposadr[self.steer_joint_ids[name]]]
            )
            for name in WHEEL_ORDER
        }
        self.current_targets = [
            WheelTarget(name, self.last_steer_angles[name], 0.0)
            for name in WHEEL_ORDER
        ]

    def _reset_pose_callback(self, request, response):
        del request
        try:
            with self.sim_lock:
                self._reset_pose_locked(
                    self.reset_pose_position,
                    self.reset_pose_yaw,
                )
            response.success = True
            response.message = (
                "reset chassis to configured start pose; simulation time and "
                "contact diagnostics were preserved"
            )
        except Exception as exc:
            response.success = False
            response.message = f"failed to reset configured start pose: {exc}"
            self.get_logger().error(response.message)
        return response

    def _update_dynamic_obstacles_locked(self, sim_time):
        for obstacle in self.dynamic_obstacles:
            runtime = obstacle["config"]
            qpos_addr = obstacle["qpos_addr"]
            dof_addr = obstacle["dof_addr"]
            current_pos = np.array(self.data.qpos[qpos_addr:qpos_addr + 3], dtype=np.float64)
            if not np.any(np.isfinite(current_pos)) or np.linalg.norm(current_pos) < 1.0e-9:
                current_pos = np.array(runtime.center, dtype=np.float64)
            current_xy = current_pos[:2]
            target_xy = self._dynamic_obstacle_waypoint(obstacle, current_xy)
            delta = target_xy - current_xy
            distance = float(np.linalg.norm(delta))
            if distance > 1.0e-6:
                direction = delta / distance
            else:
                direction = np.zeros(2, dtype=np.float64)

            speed = float(runtime.speed) * behavior_speed_scale(runtime.behavior)
            step = min(distance, speed * self.model.opt.timestep)
            next_xy = current_xy + direction * step
            velocity_xy = direction * speed if distance > 1.0e-6 else np.zeros(2)
            yaw = float(np.arctan2(direction[1], direction[0])) if distance > 1.0e-6 else 0.0

            self.data.qpos[qpos_addr:qpos_addr + 3] = np.array(
                [next_xy[0], next_xy[1], runtime.center[2]],
                dtype=np.float64,
            )
            self.data.qpos[qpos_addr + 3:qpos_addr + 7] = _quat_wxyz_from_yaw(
                yaw
            )
            self.data.qvel[dof_addr:dof_addr + 3] = np.array(
                [velocity_xy[0], velocity_xy[1], 0.0],
                dtype=np.float64,
            )
            self.data.qvel[dof_addr + 3:dof_addr + 6] = np.array(
                [0.0, 0.0, 0.0],
                dtype=np.float64,
            )

    def _dynamic_obstacle_waypoint(self, obstacle, current_xy):
        state = obstacle["state"]
        with obstacle["plan_lock"]:
            if state.path and state.waypoint_index < len(state.path):
                waypoint = state.path[state.waypoint_index]
                while (
                    state.waypoint_index < len(state.path) - 1
                    and np.linalg.norm(waypoint - current_xy) < 0.35
                ):
                    state.waypoint_index += 1
                    waypoint = state.path[state.waypoint_index]
                return np.array(waypoint, dtype=np.float64)
            return np.array(state.target, dtype=np.float64)

    def _dynamic_obstacle_target(self, obstacle, current_xy, robot_xy, robot_yaw, sim_time):
        state = obstacle["state"]
        config = self.dynamic_obstacle_config
        replan_period = 1.0 / config.replan_rate_hz
        with obstacle["plan_lock"]:
            target = np.array(state.target, dtype=np.float64)
            path = [np.array(point, dtype=np.float64) for point in state.path]
            waypoint_index = int(state.waypoint_index)
            last_replan_time = float(state.last_replan_time)
            mode = state.mode

        need_replan = sim_time - last_replan_time >= replan_period
        if np.linalg.norm(target - current_xy) < 0.3:
            need_replan = True

        if path and waypoint_index < len(path):
            waypoint = path[waypoint_index]
            if self.dynamic_obstacle_sampler is not None:
                if not self.dynamic_obstacle_sampler.segment_is_free(current_xy, waypoint):
                    need_replan = True
            if obstacle["config"].behavior == "global_wander":
                if not segment_stays_clear_of_point(
                    current_xy,
                    waypoint,
                    robot_xy,
                    config.min_robot_distance,
                ):
                    need_replan = True
            if not need_replan:
                return waypoint
        elif self.dynamic_obstacle_sampler is not None:
            if not self.dynamic_obstacle_sampler.segment_is_free(current_xy, target):
                need_replan = True
            if obstacle["config"].behavior == "global_wander":
                if not segment_stays_clear_of_point(
                    current_xy,
                    target,
                    robot_xy,
                    config.min_robot_distance,
                ):
                    need_replan = True

        if not need_replan:
            return target

        sampler = self.dynamic_obstacle_sampler
        if sampler is None:
            sampler = self._fallback_dynamic_sampler()
        planner = self.dynamic_obstacle_planner
        if planner is None or self.dynamic_obstacle_sampler is None:
            planner = sampler.coarse_planner(0.25)
        target, mode = choose_behavior_target(
            behavior=obstacle["config"].behavior,
            current_xy=current_xy,
            robot_xy=robot_xy,
            sampler=sampler,
            rng=self.dynamic_obstacle_rng,
            state_mode=mode,
            min_robot_distance=config.min_robot_distance,
            near_robot_radius=config.near_robot_radius,
            robot_yaw=robot_yaw,
        )
        new_path = self._dynamic_obstacle_path(sampler, planner, current_xy, target)
        if obstacle["config"].behavior == "global_wander":
            new_path = self._dynamic_obstacle_safe_path(
                new_path,
                current_xy,
                robot_xy,
                config.min_robot_distance,
            )
            if len(new_path) <= 1 and np.linalg.norm(current_xy - robot_xy) >= config.min_robot_distance:
                new_path = [np.array(current_xy, dtype=np.float64)]
                target = np.array(current_xy, dtype=np.float64)
                mode = "global_hold"

        waypoint_index = 1 if len(new_path) > 1 else 0
        with obstacle["plan_lock"]:
            state.target = np.array(target, dtype=np.float64)
            state.path = [np.array(point, dtype=np.float64) for point in new_path]
            state.mode = mode
            state.waypoint_index = waypoint_index
            state.last_replan_time = sim_time
            return (
                np.array(state.path[state.waypoint_index], dtype=np.float64)
                if state.path
                else np.array(state.target, dtype=np.float64)
            )

    def _dynamic_obstacle_path(self, sampler, planner, current_xy, target_xy):
        if sampler.segment_is_free(current_xy, target_xy):
            return [
                np.array(current_xy, dtype=np.float64),
                np.array(target_xy, dtype=np.float64),
            ]
        path = planner.astar_path(current_xy, target_xy, max_expansions=8000)
        if not path:
            return [np.array(target_xy, dtype=np.float64)]
        return [np.array(point, dtype=np.float64) for point in path]

    def _dynamic_obstacle_safe_path(self, path, current_xy, robot_xy, min_robot_distance):
        safe_path = [np.array(current_xy, dtype=np.float64)]
        for point in path[1:]:
            point = np.array(point, dtype=np.float64)
            current_gap = float(np.linalg.norm(safe_path[-1] - robot_xy))
            if current_gap < min_robot_distance:
                # 已经太近时允许它沿远离机器人的方向退出安全圈。
                step = point - safe_path[-1]
                away = safe_path[-1] - robot_xy
                if np.dot(away, step) < -1.0e-6:
                    break
                if np.linalg.norm(point - robot_xy) <= current_gap:
                    break
                safe_path.append(point)
                continue

            if not segment_stays_clear_of_point(
                safe_path[-1],
                point,
                robot_xy,
                min_robot_distance,
            ):
                break
            safe_path.append(point)
        return safe_path

    def _fallback_dynamic_sampler(self):
        map_center_x, map_center_y, map_half_x, map_half_y = self._map_bounds_from_manifest()
        occupancy = np.zeros((20, 20), dtype=bool)
        resolution = max(map_half_x, map_half_y) * 2.0 / 20.0
        origin = (map_center_x - map_half_x, map_center_y - map_half_y, 0.0)
        return StaticMapSampler(occupancy, resolution, origin, clearance=0.0)


    def _start_dynamic_obstacle_planner_thread(self):
        if not self.dynamic_obstacles:
            return
        rate_hz = max(1.0, self.dynamic_obstacle_config.replan_rate_hz * 4.0)
        self.get_logger().info(
            "【动态障碍规划器】线程已启用："
            f"调度频率={rate_hz:.1f} Hz"
        )
        self.dynamic_obstacle_planner_thread = threading.Thread(
            target=self._dynamic_obstacle_planner_loop,
            name="swerve_dynamic_obstacle_planner",
            daemon=True,
        )
        self.dynamic_obstacle_planner_thread.start()

    def _dynamic_obstacle_planner_loop(self):
        rate_hz = max(1.0, self.dynamic_obstacle_config.replan_rate_hz * 4.0)
        period = 1.0 / rate_hz
        while rclpy.ok() and not self.stop_event.is_set():
            loop_start = time.perf_counter()
            snapshots = []
            with self.sim_lock:
                sim_time = float(self.data.time)
                robot_xy = np.array(
                    self.data.qpos[self.free_qpos_addr:self.free_qpos_addr + 2],
                    dtype=np.float64,
                )
                robot_quat = np.array(
                    self.data.qpos[self.free_qpos_addr + 3:self.free_qpos_addr + 7],
                    dtype=np.float64,
                )
                robot_yaw = _yaw_from_quat_wxyz(robot_quat)
                for obstacle in self.dynamic_obstacles:
                    qpos_addr = obstacle["qpos_addr"]
                    current_pos = np.array(
                        self.data.qpos[qpos_addr:qpos_addr + 3],
                        dtype=np.float64,
                    )
                    if (
                        not np.any(np.isfinite(current_pos))
                        or np.linalg.norm(current_pos) < 1.0e-9
                    ):
                        current_pos = np.array(
                            obstacle["config"].center,
                            dtype=np.float64,
                        )
                    snapshots.append((obstacle, current_pos[:2]))

            for obstacle, current_xy in snapshots:
                if self.stop_event.is_set():
                    break
                self._dynamic_obstacle_target(
                    obstacle,
                    current_xy,
                    robot_xy,
                    robot_yaw,
                    sim_time,
                )

            elapsed = time.perf_counter() - loop_start
            sleep_time = period - elapsed
            if sleep_time > 0.0:
                self.stop_event.wait(sleep_time)
