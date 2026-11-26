"""Multiprocessing LiDAR and ToF worker processes for MuJoCo simulation."""

from ats_mujoco_sim.sim_sensor_math import *  # noqa: F401,F403
from ats_mujoco_sim.sim_sensor_math import (
    _cast_tof_rays,
    _import_mujoco_lidar,
    _import_multi_ray,
    _livox_pointcloud_message,
    _pointcloud_message,
    _publish_pointcloud,
    _relative_site_transform,
    _shutdown_rclpy_if_needed,
    _sim_time_to_stamp,
    _site_points_to_world,
    _timeout_clock,
    _tof_scan_pattern,
    _tof_side_targets,
    _transform_site_points,
)


def _lidar_process_main(config, state_queue, stop_event, occlusion_event):
    pin_lidar_process(config.get("lidar_cpu_ids", []))
    import mujoco
    import numpy as np
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2

    MjLidarWrapper, scan_gen = _import_mujoco_lidar()

    rclpy.init(args=[])
    node = Node("swerve_lidar_publisher")

    model = mujoco.MjModel.from_xml_path(config["model_path"])
    data = mujoco.MjData(model)
    bodyexclude = model.body(config["base_frame_id"]).id

    geomgroup = np.ones((mujoco.mjNGROUP,), dtype=np.ubyte)
    geomgroup[3] = 0
    lidar_args = {
        "bodyexclude": bodyexclude,
        "geomgroup": geomgroup,
        "ti_init_args": {"offline_cache": False},
    }
    lidar_model = str(config.get("lidar_model", "mid360")).lower()
    if lidar_model not in ("mid360", "livox_mid360"):
        raise ValueError(
            "ats_mujoco_sim only supports the MID360 LiDAR scan pattern; "
            f"got lidar_model={lidar_model!r}"
        )

    lidar_pattern = scan_gen.LivoxGenerator("mid360")
    lidar_downsample = max(1, int(config.get("lidar_downsample", 1)))
    lidar_min_range = float(lidar_pattern.laser_min_range)
    lidar_max_range = float(lidar_pattern.laser_max_range)

    try:
        lidar = MjLidarWrapper(
            model,
            site_name=config["lidar_site"],
            backend=config["lidar_backend"],
            cutoff_dist=lidar_max_range,
            args=lidar_args,
        )
    except ImportError as exc:
        requested_backend = str(config["lidar_backend"])
        if requested_backend.lower() == "cpu":
            raise
        node.get_logger().warning(
            "【LiDAR后端】'%s' 不可用（%s），已回退到 CPU；"
            "低功耗机器可显式设置 lidar_backend:=cpu。"
            % (requested_backend, exc)
        )
        config["lidar_backend"] = "cpu"
        lidar = MjLidarWrapper(
            model,
            site_name=config["lidar_site"],
            backend="cpu",
            cutoff_dist=lidar_max_range,
            args=lidar_args,
        )

    threads = int(config["lidar_threads"])
    tracer = None
    if threads > 1 and lidar.backend == "cpu":
        def make_cpu_lidar():
            return MjLidarWrapper(
                model, site_name=config["lidar_site"], backend="cpu",
                cutoff_dist=lidar_max_range, args=lidar_args,
            )
        tracer = ParallelRayTracer(model, make_cpu_lidar, threads)
    pattern = livox_scan.sensor_pattern(np.asarray(lidar_pattern.ray_angles, dtype=np.float32))
    ray_count = livox_scan.rays_per_frame(1.0 / config["lidar_rate_hz"], lidar_downsample)
    cursor = 0
    msg = _pointcloud_message(config["lidar_frame_id"])
    publisher = node.create_publisher(PointCloud2, config["lidar_topic"], 1)
    registered_scan_topic = str(config.get("registered_scan_topic", ""))
    registered_scan_msg = _pointcloud_message(config["registered_scan_frame_id"])
    registered_scan_publisher = None
    if registered_scan_topic:
        registered_scan_publisher = node.create_publisher(
            PointCloud2,
            registered_scan_topic,
            1,
        )

    node.get_logger().info(
        "【LiDAR进程】已启动：型号=MID360，"
        f"采样数={lidar_pattern.samples}，"
        f"downsample={lidar_downsample}, "
        f"backend={config['lidar_backend']}"
    )

    lidar_period = 1.0 / config["lidar_rate_hz"]
    rate_clock = config.get("lidar_rate_clock", "wall")
    next_lidar_time = None

    try:
        while rclpy.ok() and not stop_event.is_set():
            try:
                state = state_queue.get(timeout=0.1)
            except queue.Empty:
                rclpy.spin_once(node, timeout_sec=0.0)
                continue

            while True:
                try:
                    state = state_queue.get_nowait()
                except queue.Empty:
                    break

            sim_time, qpos, qvel = state
            rate_time = float(sim_time) if rate_clock == "sim" else time.monotonic()
            if next_lidar_time is None:
                next_lidar_time = rate_time
            if rate_time + 1.0e-9 < next_lidar_time:
                rclpy.spin_once(node, timeout_sec=0.0)
                continue

            data.time = float(sim_time)
            data.qpos[:] = qpos
            data.qvel[:] = qvel
            mujoco.mj_forward(model, data)

            stamp = node.get_clock().now().to_msg()

            frame_rays, cursor = livox_scan.take_rays(
                pattern, cursor, ray_count, stride=lidar_downsample
            )

            def sweep_rays():
                theta, phi = frame_rays[:, 0], frame_rays[:, 1]
                theta = np.ascontiguousarray(theta, dtype=np.float32)
                phi = np.ascontiguousarray(phi, dtype=np.float32)
                if tracer is not None:
                    tracer.set_state(sim_time, qpos, qvel)
                    ranges, points = tracer.trace(theta, phi)
                    return np.ascontiguousarray(points[np.asarray(ranges) >= lidar_min_range], dtype=np.float32)
                if hasattr(lidar, "trace_points"):
                    return lidar.trace_points(
                        data,
                        theta,
                        phi,
                        min_range=lidar_min_range,
                    )
                ranges = lidar.trace_rays(data, theta, phi)
                hit_points = lidar.get_hit_points()
                valid = np.asarray(ranges) >= lidar_min_range
                return np.ascontiguousarray(
                    np.asarray(hit_points, dtype=np.float32)[valid],
                    dtype=np.float32,
                )

            # While occluded, deliberately publish a valid, timestamped empty
            # return and skip the ray sweep entirely.  The sensor process stays
            # healthy, so this models a complete field-of-view occlusion rather
            # than a dropped input topic.
            raycast_points = resolve_lidar_return(
                occlusion_event.is_set(), sweep_rays
            )
            local_points = _transform_site_points(
                data,
                raycast_points,
                config["lidar_site"],
                config["lidar_frame_site"],
            )
            _publish_pointcloud(publisher, msg, stamp, local_points)
            if registered_scan_publisher is not None:
                registered_points = _site_points_to_world(
                    data,
                    raycast_points,
                    config["lidar_site"],
                )
                _publish_pointcloud(
                    registered_scan_publisher,
                    registered_scan_msg,
                    stamp,
                    registered_points,
                )
            next_lidar_time += lidar_period
            if rate_time - next_lidar_time >= lidar_period:
                next_lidar_time = rate_time + lidar_period

            rclpy.spin_once(node, timeout_sec=0.0)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RCLError:
        if node.context.ok():
            raise
    finally:
        if tracer is not None:
            tracer.close()
        try:
            node.destroy_node()
        finally:
            _shutdown_rclpy_if_needed()


def _lidar_process_lio_main(config, state_queue, stop_event, occlusion_event):
    """Time-resolved Mid360 for ``localization_source:=lio``.

    A frame is split into segments of at most ``lidar_segment_s``.  Each segment
    is traced against the buffered simulation state closest to the segment's
    mid time, and every hit carries the time its beam fired (``offset_time``)
    and its scan line.  Everything is stamped with simulation time, so a
    simulation that cannot keep up with real time just runs slower.
    """
    pin_lidar_process(config.get("lidar_cpu_ids", []))
    import mujoco
    import numpy as np
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile as _QoSProfile
    from sensor_msgs.msg import PointCloud2
    from std_msgs.msg import Float64MultiArray

    MjLidarWrapper, scan_gen = _import_mujoco_lidar()

    rclpy.init(args=[])
    node = Node("swerve_lidar_publisher")

    model = mujoco.MjModel.from_xml_path(config["model_path"])
    data = mujoco.MjData(model)
    # Item 8 of the S0 brief: the self-exclusion can be switched off to see
    # which rays the chassis itself would return.
    if config["lidar_self_exclusion"]:
        bodyexclude = model.body(config["base_frame_id"]).id
    else:
        bodyexclude = -1
        node.get_logger().warning(
            "【LiDAR自身排除】已关闭：车体自身几何会产生回波（仅用于检查）。"
        )

    geomgroup = np.ones((mujoco.mjNGROUP,), dtype=np.ubyte)
    geomgroup[3] = 0
    lidar_args = {
        "bodyexclude": bodyexclude,
        "geomgroup": geomgroup,
        "ti_init_args": {"offline_cache": False},
    }
    if config["lidar_backend"] == "cpu":
        from mujoco_lidar.core_cpu.mjlidar_cpu import StaticFieldMesh

        lidar_args["field_mesh"] = StaticFieldMesh.from_model(model, geomgroup, bodyexclude)
        node.get_logger().info("【LiDAR场地网格】%s" % (
            "Embree" if lidar_args["field_mesh"] is not None else "MuJoCo"
        ))
    lidar_pattern = scan_gen.LivoxGenerator("mid360")
    min_range = float(lidar_pattern.laser_min_range)
    max_range = float(config["lidar_max_range"])
    pattern = livox_scan.sensor_pattern(np.asarray(lidar_pattern.ray_angles, dtype=np.float32))

    try:
        lidar = MjLidarWrapper(
            model,
            site_name=config["lidar_site"],
            backend=config["lidar_backend"],
            cutoff_dist=max_range,
            args=lidar_args,
        )
    except ImportError as exc:
        if str(config["lidar_backend"]).lower() == "cpu":
            raise
        node.get_logger().warning(
            "【LiDAR后端】'%s' 不可用（%s），已回退到 CPU。"
            % (config["lidar_backend"], exc)
        )
        lidar = MjLidarWrapper(
            model,
            site_name=config["lidar_site"],
            backend="cpu",
            cutoff_dist=max_range,
            args=lidar_args,
        )

    # Share every segment's rays across threads (CPU backend only): the simulated Mid360
    # then keeps up with its real 200 k points/s instead of running at a tenth of real time.
    threads = resolve_thread_count(config.get("lidar_threads", 1))
    tracer = None
    if threads > 1 and lidar.backend == "cpu":
        def make_cpu_lidar():
            return MjLidarWrapper(
                model,
                site_name=config["lidar_site"],
                backend="cpu",
                cutoff_dist=max_range,
                args=lidar_args,
            )

        tracer = ParallelRayTracer(model, make_cpu_lidar, threads)
    elif threads > 1:
        node.get_logger().warning(
            "【LiDAR并行】后端 '%s' 不是 CPU，lidar_threads=%d 被忽略，单线程追踪。"
            % (lidar.backend, threads)
        )
        threads = 1

    period = 1.0 / float(config["lidar_rate_hz"])
    ray_count = livox_scan.rays_per_frame(period, config["lidar_downsample"])
    segments = livox_scan.segment_count(period, config["lidar_segment_s"])
    bounds = livox_scan.segment_bounds(ray_count, segments)
    range_noise = RangeNoiseModel(RangeNoiseConfig(
        enable=bool(config["enable_range_noise"]),
        sigma_m=float(config["range_noise_sigma_m"]),
        seed=int(config["noise_seed"]) + 1,
    ))

    qos = _QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
    publisher = node.create_publisher(PointCloud2, config["livox_points_topic"], qos)
    stats_publisher = node.create_publisher(
        Float64MultiArray, config["lidar_stats_topic"], 10
    )
    msg = _livox_pointcloud_message(config["lidar_frame_id"])

    node.get_logger().info(
        "【LiDAR进程·LIO】已启动：每帧射线=%d，分段=%d（每段<=%.1f ms），"
        "量程=[%.2f, %.1f] m，自身排除=%s，距离噪声=%s，后端=%s，追踪线程=%d"
        % (
            ray_count, segments, 1e3 * float(config["lidar_segment_s"]),
            min_range, max_range, bool(config["lidar_self_exclusion"]),
            range_noise.active, config["lidar_backend"], threads,
        )
    )

    states = []  # (sim_time, qpos, qvel), oldest first

    def fetch_until(sim_time):
        """Block until a buffered state at or after ``sim_time`` exists."""
        while not states or states[-1][0] < sim_time - 1.0e-12:
            if stop_event.is_set() or not rclpy.ok():
                return False
            try:
                states.append(state_queue.get(timeout=0.1))
            except queue.Empty:
                rclpy.spin_once(node, timeout_sec=0.0)
        return True

    relative = None
    frame_start = None
    cursor = 0
    frame_index = 0
    trace_sum = 0.0
    wait_sum = 0.0

    try:
        while rclpy.ok() and not stop_event.is_set():
            if frame_start is None:
                if not fetch_until(float("-inf")):
                    break
                frame_start = float(states[0][0])
            frame_rays, cursor = livox_scan.take_rays(
                pattern, cursor, ray_count, stride=config["lidar_downsample"]
            )
            theta_all = np.ascontiguousarray(frame_rays[:, 0], dtype=np.float32)
            phi_all = np.ascontiguousarray(frame_rays[:, 1], dtype=np.float32)

            chunks_xyz = []
            chunks_index = []
            wait_s = 0.0
            trace_s = 0.0
            complete = True
            occluded = occlusion_event.is_set()
            for segment in range(segments):
                segment_end = frame_start + period * float(bounds[segment + 1]) / ray_count
                wait_start = time.perf_counter()
                if not fetch_until(segment_end):
                    complete = False
                    break
                wait_s += time.perf_counter() - wait_start

                trace_start = time.perf_counter()
                mid = livox_scan.segment_midtime(frame_start, period, bounds, segment)
                chosen = livox_scan.nearest_state_index([s[0] for s in states], mid)
                state_time, qpos, qvel = states[chosen]
                del states[:chosen]
                first, last = int(bounds[segment]), int(bounds[segment + 1])
                if not occluded and last > first:
                    data.time = float(state_time)
                    data.qpos[:] = qpos
                    data.qvel[:] = qvel
                    mujoco.mj_kinematics(model, data)
                    if relative is None:
                        relative = _relative_site_transform(
                            data, config["lidar_site"], config["lidar_frame_site"]
                        )
                    if tracer is not None:
                        tracer.set_state(state_time, qpos, qvel)
                        distances, hit_points = tracer.trace(
                            theta_all[first:last], phi_all[first:last]
                        )
                        distances = np.asarray(distances)
                    else:
                        distances = np.asarray(
                            lidar.trace_rays(data, theta_all[first:last], phi_all[first:last])
                        )
                        hit_points = None
                    hits = np.nonzero(distances >= min_range)[0]
                    if hits.size:
                        if hit_points is None:
                            hit_points = lidar.get_hit_points()
                        points = np.asarray(hit_points, dtype=np.float64)[hits]
                        points = range_noise.apply(points)
                        rot, trans = relative
                        chunks_xyz.append(points @ rot.T + trans)
                        chunks_index.append(hits + first)
                trace_s += time.perf_counter() - trace_start

            if not complete:
                break

            if chunks_xyz:
                xyz = np.ascontiguousarray(np.vstack(chunks_xyz), dtype=np.float32)
                ray_index = np.concatenate(chunks_index)
            else:
                xyz = np.zeros((0, 3), dtype=np.float32)
                ray_index = np.zeros(0, dtype=np.int64)
            payload = livox_scan.pack_points(
                xyz,
                livox_scan.offset_time_ns(ray_index, ray_count, period),
                (ray_index % livox_scan.SCAN_LINES).astype(np.uint8),
            )
            msg.header.stamp = _sim_time_to_stamp(frame_start)
            msg.width = int(len(xyz))
            msg.row_step = livox_scan.POINT_STEP * msg.width
            msg.data = payload
            publisher.publish(msg)

            stats = Float64MultiArray()
            stats.data = [
                float(frame_start), 1e3 * trace_s, 1e3 * wait_s,
                float(len(xyz)), float(ray_count), float(segments),
            ]
            stats_publisher.publish(stats)

            trace_sum += trace_s
            wait_sum += wait_s
            frame_index += 1
            if frame_index % 100 == 0:
                node.get_logger().info(
                    "【LiDAR进程·LIO】近100帧：平均追踪 %.1f ms/帧，平均等待状态 %.1f ms/帧，"
                    "点数 %d" % (1e1 * trace_sum, 1e1 * wait_sum, len(xyz))
                )
                trace_sum = 0.0
                wait_sum = 0.0
            frame_start += period
            rclpy.spin_once(node, timeout_sec=0.0)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RCLError:
        if node.context.ok():
            raise
    finally:
        if tracer is not None:
            tracer.close()
        try:
            node.destroy_node()
        finally:
            _shutdown_rclpy_if_needed()



def _tof_process_main(config, state_queue, stop_event):
    import mujoco
    import numpy as np
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2

    rclpy.init(args=[])
    node = Node("swerve_tof_publisher")

    model = mujoco.MjModel.from_xml_path(config["model_path"])
    data = mujoco.MjData(model)
    base_body_id = model.body(config["base_frame_id"]).id
    bodyexclude = model.body(config["base_frame_id"]).id

    geomgroup = np.ones((mujoco.mjNGROUP,), dtype=np.ubyte)
    geomgroup[3] = 0

    merged_tof_publisher = node.create_publisher(
        PointCloud2,
        config["merged_tof_topic"],
        1,
    )
    merged_tof_msg = _pointcloud_message(config["base_frame_id"])
    tof_sensors = []
    for tof_config in config["tof_sensors"]:
        side_sign = 1.0 if tof_config["side"] == "left" else -1.0
        tof_sensors.append({
            "config": tof_config,
            "targets_base": _tof_side_targets(
                side_sign,
                config["tof_footprint_length"],
                config["tof_footprint_width"],
                config["tof_footprint_expand"],
                config["tof_footprint_resolution"],
                config["tof_footprint_z_min"],
                config["tof_footprint_z_max"],
            ),
            "publisher": node.create_publisher(
                PointCloud2,
                tof_config["topic"],
                1,
            ),
            "msg": _pointcloud_message(tof_config["frame_id"]),
        })

    point_count = sum(tof["targets_base"].shape[0] for tof in tof_sensors)
    node.get_logger().info(
        "【侧向ToF进程】已启动："
        f"射线数={point_count}，"
        f"足迹扩展=+{config['tof_footprint_expand']} m "
        f"@ {config['tof_footprint_resolution']} m, "
        f"z={config['tof_footprint_z_min']}-"
        f"{config['tof_footprint_z_max']} m"
    )

    tof_period = 1.0 / config["tof_rate_hz"]
    rate_clock = config.get("lidar_rate_clock", "wall")
    next_tof_time = None

    try:
        while rclpy.ok() and not stop_event.is_set():
            try:
                state = state_queue.get(timeout=0.1)
            except queue.Empty:
                rclpy.spin_once(node, timeout_sec=0.0)
                continue

            while True:
                try:
                    state = state_queue.get_nowait()
                except queue.Empty:
                    break

            sim_time, qpos, qvel = state
            rate_time = float(sim_time) if rate_clock == "sim" else time.monotonic()
            if next_tof_time is None:
                next_tof_time = rate_time
            if rate_time + 1.0e-9 < next_tof_time:
                rclpy.spin_once(node, timeout_sec=0.0)
                continue

            data.time = float(sim_time)
            data.qpos[:] = qpos
            data.qvel[:] = qvel
            mujoco.mj_forward(model, data)

            stamp = node.get_clock().now().to_msg()
            base_pos = np.asarray(data.xpos[base_body_id], dtype=np.float32)
            base_mat = np.asarray(
                data.xmat[base_body_id].reshape(3, 3),
                dtype=np.float32,
            )
            merged_points = []
            for tof in tof_sensors:
                site = data.site(tof["config"]["site"])
                points, points_base = _cast_tof_rays(
                    model,
                    data,
                    np.asarray(site.xpos, dtype=np.float32),
                    np.asarray(site.xmat.reshape(3, 3), dtype=np.float32),
                    base_pos,
                    base_mat,
                    tof["targets_base"],
                    geomgroup,
                    bodyexclude,
                    config["tof_min_range"],
                )
                if points_base.size:
                    merged_points.append(points_base)
                _publish_pointcloud(
                    tof["publisher"],
                    tof["msg"],
                    stamp,
                    points,
                )
            if merged_points:
                merged_points = np.ascontiguousarray(
                    np.vstack(merged_points),
                    dtype=np.float32,
                )
            else:
                merged_points = np.zeros((0, 3), dtype=np.float32)
            _publish_pointcloud(
                merged_tof_publisher,
                merged_tof_msg,
                stamp,
                merged_points,
            )
            next_tof_time += tof_period
            if rate_time - next_tof_time >= tof_period:
                next_tof_time = rate_time + tof_period
            rclpy.spin_once(node, timeout_sec=0.0)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RCLError:
        if node.context.ok():
            raise
    finally:
        try:
            node.destroy_node()
        finally:
            _shutdown_rclpy_if_needed()
