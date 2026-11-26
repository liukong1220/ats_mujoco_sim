"""Physics loop, ROS callbacks, swerve drive, contact audit, TF and telemetry mixin."""

from ats_mujoco_sim.sim_sensor_math import *  # noqa: F401,F403
from ats_mujoco_sim.sim_sensor_math import (
    _mat_to_xyzw,
    _quat_xyzw_from_rpy,
    _refresh_viewer_data,
    _sim_time_to_stamp,
    _yaw_from_quat_wxyz,
)


class SimDriveAndTelemetryMixin:
    """Mixin handling physics stepping, drive control, contact checks, and feedback."""

    def _start_simulation_thread(self):
        self.sim_thread = threading.Thread(
            target=self._simulation_loop,
            name="swerve_mujoco_physics",
            daemon=True,
        )
        self.sim_thread.start()


    def _simulation_loop(self):
        next_step_deadline = time.perf_counter()
        while rclpy.ok() and not self.stop_event.is_set():
            step_start = time.perf_counter()
            sync_viewer = None
            sync_shadow = None
            with self.sim_lock:
                if self.viewer is not None and not self.viewer.is_running():
                    self.viewer.close()
                    self.viewer = None
                    self.stop_event.set()
                    break

                self._step_simulation_locked()
                viewer, shadow = self.viewer, self.viewer_data
                if viewer is not None and shadow is not None:
                    now = time.perf_counter()
                    if now - self.last_viewer_sync_time >= 1.0 / self.viewer_rate_hz:
                        # Only copy the compact integration state under sim_lock; run the
                        # heavier mj_forward and viewer.sync() outside the lock so ROS
                        # callbacks and telemetry are not stalled during 3D rendering.
                        mujoco.mj_copyState(
                            self.model, self.data, shadow, mujoco.mjtState.mjSTATE_INTEGRATION
                        )
                        sync_viewer = viewer
                        sync_shadow = shadow
                        self.last_viewer_sync_time = now

            if sync_viewer is not None and sync_shadow is not None:
                mujoco.mj_forward(self.model, sync_shadow)
                sync_viewer.sync()

            if self.lio_mode:
                self._lio_flush_outbox()
                # Relative sleeps accumulate scheduling overshoot at 400 Hz.
                # Keep observation time in mj_step; only wall pacing changes.
                next_step_deadline += self.model.opt.timestep
                wall_now = time.perf_counter()
                if wall_now - next_step_deadline > self.model.opt.timestep:
                    # Sensor backpressure may exceed a step. Resume from now
                    # rather than running an unbounded burst to repay the lag.
                    next_step_deadline = wall_now
                sleep_time = next_step_deadline - wall_now
                if sleep_time > 0.0:
                    self.stop_event.wait(sleep_time)
            else:
                step_time = time.perf_counter() - step_start
                sleep_time = self.model.opt.timestep - step_time
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

# TAG 物理仿真核心步骤：计算控制目标、应用到 Mujoco、电动障碍物更新、推进仿真
    def _step_simulation_locked(self):
        self.current_targets = self._compute_targets()
        self._apply_targets(self.current_targets, self.hard_stop_requested)
        self._update_dynamic_obstacles_locked(float(self.data.time))
        self.obstacle_fixture.hold_position()
        lio_sample = (
            self._lio_capture_before_step_locked() if self.lio_mode else None
        )
        mujoco.mj_step(self.model, self.data)
        self.obstacle_fixture.hold_position()
        if lio_sample is not None:
            self._lio_collect_after_step_locked(lio_sample)
        if self.freeze_motion:
            # Remove any residual chassis momentum at the controlled fault
            # boundary while leaving sensor and localization publication alive.
            self.data.qvel[self.free_dof_addr:self.free_dof_addr + 6] = 0.0
        self._enforce_joint_velocity_limits()
        self._evaluate_contacts()

    def _motion_control_callback(self, msg):
        with self.sim_lock:
            self.motion_cmd = ChassisCommand(
                float(msg.linear_x),
                float(msg.linear_y),
                float(msg.angular_z),
            )
            self.last_motion_time = self._clock_now()

    def _on_set_parameters(self, parameters):
        # The transitions and their operator-visible wording live in
        # runtime_fault_state, which carries no ROS message imports so the
        # contract stays unit-testable without a built workspace.
        fixture_requests = [p for p in parameters if p.name == "test_obstacle_position"]
        try:
            for parameter in fixture_requests:
                self.obstacle_fixture.validate_position(parameter.value)
        except ValueError as error:
            return SetParametersResult(successful=False, reason=str(error))
        outcome = apply_runtime_fault_parameters(self, parameters)
        if outcome.successful:
            for parameter in fixture_requests:
                with self.sim_lock:
                    marker = self.obstacle_fixture.set_position(parameter.value)
                outcome.reason = json.dumps(marker)
                self.get_logger().warn("S3 obstacle_fixture " + outcome.reason)
        for message in outcome.messages:
            self.get_logger().warn(message)
        return SetParametersResult(
            successful=outcome.successful,
            reason=outcome.reason,
        )

    def _emergency_stop_callback(self, msg):
        with self.sim_lock:
            self.emergency_stop_active = bool(msg.data)

    def _yaw_authority_request_callback(self, msg):
        with self.sim_lock:
            if msg.request_sequence <= self.gimbal_request_sequence:
                return
            self.gimbal_request_sequence = int(msg.request_sequence)
            self.gimbal_yaw_authority = int(msg.yaw_authority)
            self.gimbal_locked = bool(msg.require_gimbal_lock)

    def _pose_cmd_callback(self, msg):
        with self.sim_lock:
            self.motion_cmd = ChassisCommand(
                float(msg.v),
                0.0,
                float(msg.omega),
            )
            self.last_motion_time = self._clock_now()

    def _speed_control_callback(self, msg):
        with self.sim_lock:
            self.direct_speeds["lf"] = float(msg.lf_speed)
            self.direct_speeds["lr"] = float(msg.lr_speed)
            self.direct_speeds["rf"] = float(msg.rf_speed)
            self.direct_speeds["rr"] = float(msg.rr_speed)
            self.last_speed_time = self._clock_now()

    def _steer_control_callback(self, msg):
        with self.sim_lock:
            self.direct_steer_angles["lf"] = radians(float(msg.lf_steer_angle))
            self.direct_steer_angles["lr"] = radians(float(msg.lr_steer_angle))
            self.direct_steer_angles["rf"] = radians(float(msg.rf_steer_angle))
            self.direct_steer_angles["rr"] = radians(float(msg.rr_steer_angle))

    def _motion_mode_callback(self, request, response):
        mode = int(request.cmd_ctl)
        if mode not in VALID_MOTION_MODES:
            self.get_logger().warn(f"【运动模式】拒绝未知模式：{mode}")
            response.cmd_ack = CMD_ACK_FAIL
            return response

        with self.sim_lock:
            self.mode = mode
            self.motion_cmd = ChassisCommand()
            self.last_motion_time = self._clock_now()
        self.get_logger().info(
            f"【运动模式】已切换为 {mode}（{MODE_NAMES[mode]}）"
        )
        response.cmd_ack = CMD_ACK_FINISH
        return response

    def _control_mode_callback(self, request, response):
        with self.sim_lock:
            self.control_mode = int(request.cmd_ctl)
        response.cmd_ack = CMD_ACK_FINISH
        return response

    def _compute_targets(self):
        now = self._clock_now()

        # if self.mode == MODE_USER_CTRL:
        #     if now - self.last_speed_time > self.command_timeout:
        #         speeds = {name: 0.0 for name in WHEEL_ORDER}
        #     else:
        #         speeds = self.direct_speeds

        #     targets = [
        #         WheelTarget(
        #             name,
        #             self.direct_steer_angles[name],
        #             speeds[name],
        #         )
        #         for name in WHEEL_ORDER
        #     ]
        #     self.effective_motion_cmd = estimate_chassis_command(targets)
        #     return targets

        command_stale = now - self.last_motion_time > self.command_timeout
        self.hard_stop_requested = (
            self.freeze_motion or self.emergency_stop_active or command_stale
        )
        if self.hard_stop_requested:
            command = ChassisCommand()
        else:
            command = self.motion_cmd

        previous_angles = {
            name: self._joint_position(self.steer_joint_ids[name])
            for name in WHEEL_ORDER
        }
        effective, targets = mode_to_wheel_targets(
            self.mode,
            command,
            previous_angles,
            self.wheel_positions,
        )
        self.effective_motion_cmd = effective
        return targets

    def _apply_targets(self, targets, hard_stop=False):
        dt = float(self.model.opt.timestep)
        for target in targets:
            steer_id = self.steer_actuator_ids[target.name]
            wheel_id = self.wheel_actuator_ids[target.name]
            limited_angle, steer_saturated = rate_limit_angle(
                self.last_steer_angles[target.name],
                target.steer_angle,
                self.max_steer_rate * dt,
            )
            desired_speed = 0.0 if hard_stop else target.wheel_speed
            speed_saturated = False
            if self.max_wheel_speed > 0.0:
                limited_speed = min(
                    max(desired_speed, -self.max_wheel_speed),
                    self.max_wheel_speed,
                )
                speed_saturated = limited_speed != desired_speed
                desired_speed = limited_speed

            actual_angle = self._joint_position(self.steer_joint_ids[target.name])
            alignment = max(0.0, cos(normalize_angle(target.steer_angle - actual_angle)))
            desired_speed *= alignment
            acceleration_saturated = False
            if hard_stop:
                limited_speed = 0.0
            else:
                max_speed_delta = self.max_wheel_acceleration * dt
                previous_speed = self.last_wheel_speeds[target.name]
                speed_delta = desired_speed - previous_speed
                limited_delta = min(max(speed_delta, -max_speed_delta), max_speed_delta)
                limited_speed = previous_speed + limited_delta
                acceleration_saturated = abs(limited_delta - speed_delta) > 1e-12
            wheel_ctrl = limited_speed / self.wheel_radius

            self.data.ctrl[steer_id] = self._clamp_actuator_ctrl(
                steer_id,
                limited_angle,
            )
            self.data.ctrl[wheel_id] = self._clamp_actuator_ctrl(
                wheel_id,
                wheel_ctrl,
            )
            self.last_steer_angles[target.name] = limited_angle
            self.last_wheel_speeds[target.name] = limited_speed
            self.drive_speed_saturated[target.name] = speed_saturated
            self.drive_acceleration_saturated[target.name] = acceleration_saturated
            self.steer_rate_saturated[target.name] = steer_saturated
            self.drive_speed_saturation_count += int(speed_saturated)
            self.drive_acceleration_saturation_count += int(acceleration_saturated)
            self.steer_rate_saturation_count += int(steer_saturated)

    def _enforce_joint_velocity_limits(self):
        max_wheel_rate = self.max_wheel_speed / self.wheel_radius
        for name in WHEEL_ORDER:
            steer_dof = self.model.jnt_dofadr[self.steer_joint_ids[name]]
            wheel_dof = self.model.jnt_dofadr[self.wheel_joint_ids[name]]
            self.data.qvel[steer_dof] = min(
                max(self.data.qvel[steer_dof], -self.max_steer_rate),
                self.max_steer_rate,
            )
            if self.hard_stop_requested:
                self.data.qvel[wheel_dof] = 0.0
            else:
                self.data.qvel[wheel_dof] = min(
                    max(self.data.qvel[wheel_dof], -max_wheel_rate),
                    max_wheel_rate,
                )

    def _evaluate_contacts(self):
        stats = self.contact_eval_stats
        force = self._contact_force_buf
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            g1_robot = geom1 in self.robot_geom_ids
            g2_robot = geom2 in self.robot_geom_ids
            if not g1_robot and not g2_robot:
                continue
            force.fill(0.0)
            mujoco.mj_contactForce(self.model, self.data, contact_index, force)
            force_norm = float(np.linalg.norm(force[:3]))
            robot_geom = geom1 if g1_robot else geom2
            wheel_center_z = (
                float(self.data.xpos[self.geom_body_ids[robot_geom]][2])
                if robot_geom in self.wheel_geom_ids
                else None
            )
            verdict = classify_mujoco_contact(
                geom1,
                geom2,
                self.geom_names[geom1],
                self.geom_names[geom2],
                self.robot_geom_ids,
                self.wheel_geom_ids,
                self.ground_geom_ids,
                contact.pos,
                contact.frame,
                float(contact.dist),
                force_norm,
                wheel_center_z=wheel_center_z,
                wheel_radius=self.wheel_radius,
            )
            if verdict is None:
                continue
            category = verdict["category"]
            if not verdict["is_illegal"]:
                if category == "legal_wheel_ground":
                    stats["legal_wheel_ground_count"] += 1
                elif category == "legal_wheel_drivable_deck":
                    stats["legal_wheel_drivable_deck_count"] += 1
                elif category == "legal_chassis_ramp_graze":
                    stats["legal_chassis_ramp_graze_count"] += 1
                elif category == "speculative_margin":
                    stats["speculative_margin_count"] += 1
                continue

            self.contact_violation_count += 1
            self.max_contact_force = max(self.max_contact_force, force_norm)
            stats["illegal_contact_count"] += 1
            stats["illegal_max_force_n"] = max(stats["illegal_max_force_n"], force_norm)
            penetration = max(0.0, -float(contact.dist))
            stats["illegal_max_penetration_m"] = max(
                stats["illegal_max_penetration_m"], penetration
            )
            if verdict["robot_role"] == "chassis":
                stats["illegal_chassis_count"] += 1
            elif category == "illegal_wheel_ceiling":
                stats["illegal_wheel_ceiling_count"] += 1
            elif category == "illegal_wheel_obstacle":
                stats["illegal_obstacle_count"] += 1
            else:
                stats["illegal_wheel_side_count"] += 1

            # Bounded trace of illegal physical contacts with full attitude & normal.
            if self.contact_diagnostic_count < 32:
                self.contact_diagnostic_count += 1
                record = {
                    "sim_time": float(self.data.time),
                    "violation_count": self.contact_violation_count,
                    "category": category,
                    "robot_role": verdict["robot_role"],
                    "env_category": verdict["env_category"],
                    "normal_env_to_robot_z": round(float(verdict["nz"]), 4),
                    "geom_ids": [geom1, geom2],
                    "geom_names": [self.geom_names[geom1], self.geom_names[geom2]],
                    "position": np.asarray(contact.pos).tolist(),
                    "distance_m": float(contact.dist),
                    "force_norm_n": force_norm,
                    "base_position": np.asarray(self.data.xpos[self.base_body_id]).tolist(),
                    "base_quaternion_wxyz": np.asarray(
                        self.data.xquat[self.base_body_id]).tolist(),
                }
                stats["illegal_samples"].append(record)
                self.get_logger().warning("【非法物理接触】" + json.dumps(record, allow_nan=False))

    def _clamp_actuator_ctrl(self, actuator_id, value):
        low = self.model.actuator_ctrlrange[actuator_id, 0]
        high = self.model.actuator_ctrlrange[actuator_id, 1]
        if low == 0.0 and high == 0.0:
            return value
        return min(max(value, low), high)

    def _joint_position(self, joint_id):
        qpos_addr = self.model.jnt_qposadr[joint_id]
        return float(self.data.qpos[qpos_addr])

    def _joint_velocity(self, joint_id):
        dof_addr = self.model.jnt_dofadr[joint_id]
        return float(self.data.qvel[dof_addr])

    def _set_transform(self, msg, translation, quat_xyzw):
        msg.transform.translation.x = float(translation[0])
        msg.transform.translation.y = float(translation[1])
        msg.transform.translation.z = float(translation[2])
        msg.transform.rotation.x = float(quat_xyzw[0])
        msg.transform.rotation.y = float(quat_xyzw[1])
        msg.transform.rotation.z = float(quat_xyzw[2])
        msg.transform.rotation.w = float(quat_xyzw[3])

    def _make_transform(self, stamp, parent, child, translation, quat_xyzw):
        msg = TransformStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = parent
        msg.child_frame_id = child
        self._set_transform(msg, translation, quat_xyzw)
        return msg

    def _publish_static_transforms(self):
        stamp = self.get_clock().now().to_msg()
        self.static_tf_broadcaster.sendTransform([
            self._make_transform(
                stamp,
                self.robot_base_frame_id,
                self.base_footprint_frame_id,
                # base_footprint lies on the ground below the base_link body.
                np.array([0.0, 0.0, -self.base_footprint_height]),
                (0.0, 0.0, 0.0, 1.0),
            ),
            self._make_transform(
                stamp,
                self.robot_base_frame_id,
                self.lidar_tf_frame_id,
                *self._lidar_static_pose(),
            )
        ])

    def _lidar_static_pose(self):
        """body -> front_mid360 translation and xyzw quaternion.

        The legacy preset keeps the original constants so older scenes publish
        the exact values they always did; the chassis preset derives them from
        the same MountPose that patched the model and set the gravity vector.
        """
        if self.lidar_mount_preset == "legacy_gimbal":
            return MID360_TRANSLATION, _quat_xyzw_from_rpy(*MID360_RPY)
        return (
            self.lidar_mount_pose.position_vector(),
            self.lidar_mount_pose.quat_xyzw(),
        )

    def _body_pose_locked(self):
        pos = np.array(self.data.xpos[self.base_body_id], dtype=np.float64)
        mat = np.array(
            self.data.xmat[self.base_body_id].reshape(3, 3),
            dtype=np.float64,
        )
        return pos, mat

    def _site_pose_locked(self, site_name):
        site = self.data.site(site_name)
        pos = np.array(site.xpos, dtype=np.float64)
        mat = np.array(site.xmat.reshape(3, 3), dtype=np.float64)
        return pos, mat

    def _publish_truth(self):
        stamp = self.get_clock().now().to_msg()
        with self.sim_lock:
            base_pos, base_mat = self._body_pose_locked()
            lidar_pos, lidar_mat = self._site_pose_locked(self.lidar_frame_site)
            left_tof_pos, left_tof_mat = self._site_pose_locked(
                self.left_tof_site
            )
            right_tof_pos, right_tof_mat = self._site_pose_locked(
                self.right_tof_site
            )
            base_linear_velocity, base_angular_velocity = (
                self._body_velocity_locked()
            )
            lidar_linear_velocity, lidar_angular_velocity = (
                self._site_velocity_locked(self.lidar_frame_site_id)
            )

        base_quat = _mat_to_xyzw(base_mat)
        lidar_quat = _mat_to_xyzw(lidar_mat)
        lidar_to_left_tof_mat = lidar_mat.T @ left_tof_mat
        lidar_to_left_tof_pos = lidar_mat.T @ (left_tof_pos - lidar_pos)
        lidar_to_left_tof_quat = _mat_to_xyzw(lidar_to_left_tof_mat)
        lidar_to_right_tof_mat = lidar_mat.T @ right_tof_mat
        lidar_to_right_tof_pos = lidar_mat.T @ (right_tof_pos - lidar_pos)
        lidar_to_right_tof_quat = _mat_to_xyzw(lidar_to_right_tof_mat)
        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = self.odom_frame_id
        odom_msg.child_frame_id = self.robot_base_frame_id
        odom_msg.pose.pose.position.x = float(base_pos[0])
        odom_msg.pose.pose.position.y = float(base_pos[1])
        odom_msg.pose.pose.position.z = float(base_pos[2])
        odom_msg.pose.pose.orientation.x = base_quat[0]
        odom_msg.pose.pose.orientation.y = base_quat[1]
        odom_msg.pose.pose.orientation.z = base_quat[2]
        odom_msg.pose.pose.orientation.w = base_quat[3]
        odom_msg.twist.twist.linear.x = float(base_linear_velocity[0])
        odom_msg.twist.twist.linear.y = float(base_linear_velocity[1])
        odom_msg.twist.twist.linear.z = float(base_linear_velocity[2])
        odom_msg.twist.twist.angular.x = float(base_angular_velocity[0])
        odom_msg.twist.twist.angular.y = float(base_angular_velocity[1])
        odom_msg.twist.twist.angular.z = float(base_angular_velocity[2])

        lidar_odom_msg = Odometry()
        lidar_odom_msg.header.stamp = stamp
        lidar_odom_msg.header.frame_id = self.odom_frame_id
        lidar_odom_msg.child_frame_id = self.lidar_tf_frame_id
        lidar_odom_msg.pose.pose.position.x = float(lidar_pos[0])
        lidar_odom_msg.pose.pose.position.y = float(lidar_pos[1])
        lidar_odom_msg.pose.pose.position.z = float(lidar_pos[2])
        lidar_odom_msg.pose.pose.orientation.x = lidar_quat[0]
        lidar_odom_msg.pose.pose.orientation.y = lidar_quat[1]
        lidar_odom_msg.pose.pose.orientation.z = lidar_quat[2]
        lidar_odom_msg.pose.pose.orientation.w = lidar_quat[3]
        lidar_odom_msg.twist.twist.linear.x = float(lidar_linear_velocity[0])
        lidar_odom_msg.twist.twist.linear.y = float(lidar_linear_velocity[1])
        lidar_odom_msg.twist.twist.linear.z = float(lidar_linear_velocity[2])
        lidar_odom_msg.twist.twist.angular.x = float(lidar_angular_velocity[0])
        lidar_odom_msg.twist.twist.angular.y = float(lidar_angular_velocity[1])
        lidar_odom_msg.twist.twist.angular.z = float(lidar_angular_velocity[2])

        transforms = []
        if self.publish_map_to_odom_tf:
            transforms.append(
                self._make_transform(
                    stamp,
                    self.map_frame_id,
                    self.odom_frame_id,
                    np.zeros(3),
                    (0.0, 0.0, 0.0, 1.0),
                )
            )
        if self.publish_robot_base_tf:
            transforms.append(
                self._make_transform(
                    stamp,
                    self.odom_frame_id,
                    self.robot_base_frame_id,
                    base_pos,
                    base_quat,
                )
            )
        transforms.extend([
            self._make_transform(
                stamp,
                self.lidar_tf_frame_id,
                self.left_tof_frame_id,
                lidar_to_left_tof_pos,
                lidar_to_left_tof_quat,
            ),
            self._make_transform(
                stamp,
                self.lidar_tf_frame_id,
                self.right_tof_frame_id,
                lidar_to_right_tof_pos,
                lidar_to_right_tof_quat,
            ),
        ])

        self.localization_pub.publish(odom_msg)
        if self.lidar_odometry_pub is not None:
            self.lidar_odometry_pub.publish(lidar_odom_msg)
        self.tf_broadcaster.sendTransform(transforms)

    def _publish_feedback(self):
        with self.sim_lock:
            self._publish_motion_feedback()
            self._publish_wheel_feedback()
            self._publish_gimbal_status()
            self._publish_swerve_telemetry()
            self._publish_system_feedback()
            self._publish_battery_feedback()

    def _publish_motion_feedback(self):
        msg = MotionFb()
        msg.linear_x = float(self.effective_motion_cmd.linear_x)
        msg.linear_y = float(self.effective_motion_cmd.linear_y)
        msg.angular_z = float(self.effective_motion_cmd.angular_z)
        msg.mode_type = int(self.mode)
        msg.mode_switch = 0
        self.motion_fb_pub.publish(msg)

    def _publish_wheel_feedback(self):
        speed_msg = SpeedFb()
        steer_msg = SteerFb()

        wheel_speeds = {}
        steer_angles = {}
        for name in WHEEL_ORDER:
            wheel_velocity = self._joint_velocity(self.wheel_joint_ids[name])
            wheel_speeds[name] = wheel_velocity * self.wheel_radius
            steer_angles[name] = degrees(
                self._joint_position(self.steer_joint_ids[name])
            )

        speed_msg.lf_speed = float(wheel_speeds["lf"])
        speed_msg.lr_speed = float(wheel_speeds["lr"])
        speed_msg.rf_speed = float(wheel_speeds["rf"])
        speed_msg.rr_speed = float(wheel_speeds["rr"])

        steer_msg.lf_steer_angle = float(steer_angles["lf"])
        steer_msg.lr_steer_angle = float(steer_angles["lr"])
        steer_msg.rf_steer_angle = float(steer_angles["rf"])
        steer_msg.rr_steer_angle = float(steer_angles["rr"])

        self.speed_fb_pub.publish(speed_msg)
        self.steer_fb_pub.publish(steer_msg)

    def _body_velocity_locked(self):
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.base_body_id,
            velocity,
            1,
        )
        return velocity[3:6], velocity[0:3]

    def _site_velocity_locked(self, site_id):
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_SITE,
            site_id,
            velocity,
            1,
        )
        return velocity[3:6], velocity[0:3]

    def _publish_swerve_telemetry(self):
        linear_velocity, angular_velocity = self._body_velocity_locked()
        drive_rpm = []
        wheel_speed_mps = []
        steer_angle = []
        steer_rate = []
        longitudinal_slip = []
        lateral_slip = []
        for name in WHEEL_ORDER:
            wheel_rate = self._joint_velocity(self.wheel_joint_ids[name])
            wheel_speed = wheel_rate * self.wheel_radius
            angle = self._joint_position(self.steer_joint_ids[name])
            angle_rate = self._joint_velocity(self.steer_joint_ids[name])
            x_pos, y_pos = self.wheel_positions[name]
            wheel_vx = linear_velocity[0] - angular_velocity[2] * y_pos
            wheel_vy = linear_velocity[1] + angular_velocity[2] * x_pos
            longitudinal = wheel_vx * cos(angle) + wheel_vy * sin(angle)
            lateral = -wheel_vx * sin(angle) + wheel_vy * cos(angle)
            drive_rpm.append(wheel_rate * 60.0 / (2.0 * pi))
            wheel_speed_mps.append(wheel_speed)
            steer_angle.append(angle)
            steer_rate.append(angle_rate)
            longitudinal_slip.append(wheel_speed - longitudinal)
            lateral_slip.append(lateral)

        message = SwerveTelemetry()
        message.header.stamp = self._stamp_now()
        message.header.frame_id = self.robot_base_frame_id
        self.telemetry_sequence += 1
        message.sequence = self.telemetry_sequence
        message.drive_rpm = drive_rpm
        message.wheel_speed_mps = wheel_speed_mps
        message.steer_angle = steer_angle
        message.steer_rate = steer_rate
        message.longitudinal_slip_mps = longitudinal_slip
        message.lateral_slip_mps = lateral_slip
        message.command_vx = float(self.effective_motion_cmd.linear_x)
        message.command_vy = float(self.effective_motion_cmd.linear_y)
        message.command_wz = float(self.effective_motion_cmd.angular_z)
        message.measured_vx = float(linear_velocity[0])
        message.measured_vy = float(linear_velocity[1])
        message.measured_wz = float(angular_velocity[2])
        message.drive_speed_saturated = [
            self.drive_speed_saturated[name] for name in WHEEL_ORDER
        ]
        message.drive_acceleration_saturated = [
            self.drive_acceleration_saturated[name] for name in WHEEL_ORDER
        ]
        message.steer_rate_saturated = [
            self.steer_rate_saturated[name] for name in WHEEL_ORDER
        ]
        message.drive_speed_saturation_count = self.drive_speed_saturation_count
        message.drive_acceleration_saturation_count = (
            self.drive_acceleration_saturation_count
        )
        message.steer_rate_saturation_count = self.steer_rate_saturation_count
        message.contact_violation_count = self.contact_violation_count
        message.max_contact_force = self.max_contact_force
        self.swerve_telemetry_pub.publish(message)
        self._contact_eval_publish_stride += 1
        if (
            self.contact_violation_count != self._last_published_illegal_count
            or self._contact_eval_publish_stride >= 10
        ):
            self._contact_eval_publish_stride = 0
            self._last_published_illegal_count = self.contact_violation_count
            eval_msg = String()
            eval_msg.data = json.dumps(
                {**self.contact_eval_stats, "sim_time": float(self.data.time)},
                allow_nan=False,
            )
            self.contact_eval_pub.publish(eval_msg)

    def _publish_gimbal_status(self):
        _, body_rotation = self._body_pose_locked()
        body_yaw = float(np.arctan2(body_rotation[1, 0], body_rotation[0, 0]))
        message = GimbalYawStatus()
        message.header.stamp = self._stamp_now()
        message.header.frame_id = self.robot_base_frame_id
        self.gimbal_status_sequence += 1
        message.sequence = self.gimbal_status_sequence
        message.request_sequence = self.gimbal_request_sequence
        message.yaw_authority = self.gimbal_yaw_authority
        message.locked = self.gimbal_locked
        # The present MuJoCo assets model a fixed lidar mounting.  This is a
        # simulator acknowledgement, not a claim of rotating-lidar map fidelity.
        message.tf_healthy = True
        message.gimbal_yaw = body_yaw
        message.body_yaw = body_yaw
        message.fake_yaw = self.initial_body_yaw
        self.gimbal_status_pub.publish(message)

    def _publish_system_feedback(self):
        msg = SystemstateFb()
        msg.system_mode = 0
        msg.control_mode = int(self.control_mode)
        msg.emergency_mode = 0
        msg.obstacle_mode = 0
        msg.proximity_switch_mode = 0
        msg.secure_edge_mode = 0
        self.system_fb_pub.publish(msg)

    def _publish_battery_feedback(self):
        msg = BatteryFb()
        msg.battery_soc = 100
        msg.battery_voltage = 480
        msg.battery_current = 0
        self.battery_fb_pub.publish(msg)
