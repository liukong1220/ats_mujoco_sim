"""ROS2 interface that drives the MuJoCo four-swerve chassis model."""

from math import cos, degrees, pi, radians, sin
import multiprocessing as mp
from pathlib import Path
import queue
import sys
import threading
import time

from carstatemsgs.msg import CarState
from ats_navigation_interfaces.msg import SwerveTelemetry
from ats_navigation_interfaces.msg import GimbalYawStatus
from ats_navigation_interfaces.msg import YawAuthorityRequest
from geometry_msgs.msg import TransformStamped
import mujoco
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import Bool
from tf2_ros import StaticTransformBroadcaster
from tf2_ros import TransformBroadcaster

from manda_can_control.msg import BatteryFb
from manda_can_control.msg import MotionCtrl
from manda_can_control.msg import MotionFb
from manda_can_control.msg import SpeedCtrl
from manda_can_control.msg import SpeedFb
from manda_can_control.msg import SteerCtrl
from manda_can_control.msg import SteerFb
from manda_can_control.msg import SystemstateFb
from manda_can_control.srv import ControlMode
from manda_can_control.srv import MotionMode

from ats_mujoco_sim.dynamic_obstacles import DynamicObstacleSceneConfig
from ats_mujoco_sim.dynamic_obstacles import DynamicObstacleControllerState
from ats_mujoco_sim.dynamic_obstacles import StaticMapSampler
from ats_mujoco_sim.dynamic_obstacles import behavior_speed_scale
from ats_mujoco_sim.dynamic_obstacles import bool_from_value
from ats_mujoco_sim.dynamic_obstacles import build_runtime_configs
from ats_mujoco_sim.dynamic_obstacles import choose_behavior_target
from ats_mujoco_sim.dynamic_obstacles import segment_stays_clear_of_point
from ats_mujoco_sim.dynamic_obstacles import staggered_replan_time
from ats_mujoco_sim.kinematics import ChassisCommand
from ats_mujoco_sim.kinematics import MAX_STEER_RATE_RADPS
from ats_mujoco_sim.kinematics import MAX_WHEEL_SPEED_MPS
from ats_mujoco_sim.kinematics import MODE_NAMES
from ats_mujoco_sim.kinematics import MODE_SWERVE
from ats_mujoco_sim.kinematics import MODE_USER_CTRL
from ats_mujoco_sim.kinematics import WHEEL_ORDER
from ats_mujoco_sim.kinematics import WHEEL_POSITIONS
from ats_mujoco_sim.kinematics import WHEEL_RADIUS_M
from ats_mujoco_sim.kinematics import WheelTarget
from ats_mujoco_sim.kinematics import contact_is_violation
from ats_mujoco_sim.kinematics import estimate_chassis_command
from ats_mujoco_sim.kinematics import mode_to_wheel_targets
from ats_mujoco_sim.kinematics import normalize_angle
from ats_mujoco_sim.kinematics import rate_limit_angle
from ats_mujoco_sim.map_metadata import load_occupancy_image
from ats_mujoco_sim.map_metadata import read_map_metadata
from ats_mujoco_sim.mid360_model import MID360_MESH_RELATIVE_PATH


CMD_ACK_FINISH = 0
CMD_ACK_FAIL = 1
VALID_MOTION_MODES = set(MODE_NAMES)

STEER_ACTUATORS = {
    "lf": "front_left_steer_pos",
    "lr": "rear_left_steer_pos",
    "rf": "front_right_steer_pos",
    "rr": "rear_right_steer_pos",
}

WHEEL_ACTUATORS = {
    "lf": "front_left_wheel_vel",
    "lr": "rear_left_wheel_vel",
    "rf": "front_right_wheel_vel",
    "rr": "rear_right_wheel_vel",
}

STEER_JOINTS = {
    "lf": "front_left_steer_joint",
    "lr": "rear_left_steer_joint",
    "rf": "front_right_steer_joint",
    "rr": "rear_right_steer_joint",
}

WHEEL_JOINTS = {
    "lf": "front_left_wheel_joint",
    "lr": "rear_left_wheel_joint",
    "rf": "front_right_wheel_joint",
    "rr": "rear_right_wheel_joint",
}

# Keep the simulated MID360 mounting pose aligned with
# ats_sentry_robot.sdf.xmacro:
#   parent=gimbal_yaw_odom
#   pose="-0.2 -0.0 0.0 0.0 0 -${61*pi/180}"
MID360_PARENT_FRAME_ID = "gimbal_yaw_odom"
MID360_FRAME_ID = "front_mid360"
MID360_TRANSLATION = np.array([-0.2, 0.0, 0.0], dtype=np.float64)
MID360_RPY = (0.0, 0.0, radians(-61.0))


def _shutdown_rclpy_if_needed():
    """Shutdown rclpy quietly; Ctrl-C can make launch call shutdown first."""
    if not rclpy.ok():
        return
    try:
        rclpy.shutdown()
    except RCLError:
        # Treat double shutdown during SIGINT as a normal exit path.
        pass


def _import_mujoco_lidar():
    try:
        from mujoco_lidar import MjLidarWrapper
        from mujoco_lidar import scan_gen

        return MjLidarWrapper, scan_gen
    except ModuleNotFoundError as exc:
        if exc.name != "mujoco_lidar":
            raise

    for parent in Path(__file__).resolve().parents:
        candidates = (
            parent / "MuJoCo-LiDAR" / "src",
            parent / "src" / "MuJoCo-LiDAR" / "src",
        )
        for candidate in candidates:
            if (candidate / "mujoco_lidar" / "__init__.py").exists():
                sys.path.insert(0, str(candidate))
                from mujoco_lidar import MjLidarWrapper
                from mujoco_lidar import scan_gen

                return MjLidarWrapper, scan_gen

    raise ModuleNotFoundError(
        "mujoco_lidar is not installed and src/MuJoCo-LiDAR/src was not found"
    )


def _mat_to_xyzw(mat):
    trace = float(mat[0, 0] + mat[1, 1] + mat[2, 2])
    if trace > 0.0:
        s = (trace + 1.0) ** 0.5 * 2.0
        w = 0.25 * s
        x = (mat[2, 1] - mat[1, 2]) / s
        y = (mat[0, 2] - mat[2, 0]) / s
        z = (mat[1, 0] - mat[0, 1]) / s
    elif mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
        s = (1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]) ** 0.5 * 2.0
        w = (mat[2, 1] - mat[1, 2]) / s
        x = 0.25 * s
        y = (mat[0, 1] + mat[1, 0]) / s
        z = (mat[0, 2] + mat[2, 0]) / s
    elif mat[1, 1] > mat[2, 2]:
        s = (1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2]) ** 0.5 * 2.0
        w = (mat[0, 2] - mat[2, 0]) / s
        x = (mat[0, 1] + mat[1, 0]) / s
        y = 0.25 * s
        z = (mat[1, 2] + mat[2, 1]) / s
    else:
        s = (1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1]) ** 0.5 * 2.0
        w = (mat[1, 0] - mat[0, 1]) / s
        x = (mat[0, 2] + mat[2, 0]) / s
        y = (mat[1, 2] + mat[2, 1]) / s
        z = 0.25 * s
    return float(x), float(y), float(z), float(w)


def _quat_wxyz_from_yaw(yaw):
    half_yaw = 0.5 * float(yaw)
    return np.array(
        [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)],
        dtype=np.float64,
    )


def _quat_xyzw_from_rpy(roll, pitch, yaw):
    half_roll = 0.5 * float(roll)
    half_pitch = 0.5 * float(pitch)
    half_yaw = 0.5 * float(yaw)
    cr = np.cos(half_roll)
    sr = np.sin(half_roll)
    cp = np.cos(half_pitch)
    sp = np.sin(half_pitch)
    cy = np.cos(half_yaw)
    sy = np.sin(half_yaw)
    return (
        float(sr * cp * cy - cr * sp * sy),
        float(cr * sp * cy + sr * cp * sy),
        float(cr * cp * sy - sr * sp * cy),
        float(cr * cp * cy + sr * sp * sy),
    )


def _yaw_from_quat_wxyz(quat):
    w, x, y, z = [float(value) for value in quat]
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _pointcloud_message(frame_id):
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs.msg import PointField

    msg = PointCloud2()
    msg.header.frame_id = frame_id
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 16
    msg.height = 1
    msg.is_dense = True
    return msg


def _tof_scan_pattern(horizontal_fov_deg, vertical_fov_deg, width, height):
    horizontal = np.deg2rad(
        np.linspace(
            -0.5 * float(horizontal_fov_deg),
            0.5 * float(horizontal_fov_deg),
            max(1, int(width)),
            dtype=np.float32,
        )
    )
    vertical = np.deg2rad(
        np.linspace(
            -0.5 * float(vertical_fov_deg),
            0.5 * float(vertical_fov_deg),
            max(1, int(height)),
            dtype=np.float32,
        )
    )
    theta, phi = np.meshgrid(horizontal, vertical)
    return (
        np.ascontiguousarray(theta.ravel(), dtype=np.float32),
        np.ascontiguousarray(phi.ravel(), dtype=np.float32),
    )


def _publish_pointcloud(publisher, msg, stamp, points, intensity=1.0):
    points = np.ascontiguousarray(points, dtype=np.float32)
    if points.size == 0:
        points = np.zeros((0, 4), dtype=np.float32)
    elif points.ndim != 2 or points.shape[1] not in (3, 4):
        raise ValueError("Point cloud must have shape (N, 3) or (N, 4)")
    elif points.shape[1] == 3:
        intensities = np.full(
            (points.shape[0], 1),
            float(intensity),
            dtype=np.float32,
        )
        points = np.ascontiguousarray(
            np.hstack((points, intensities)),
            dtype=np.float32,
        )
    msg.header.stamp = stamp
    msg.width = int(points.shape[0])
    msg.row_step = msg.point_step * msg.width
    msg.data = points.tobytes()
    publisher.publish(msg)


def _transform_site_points(data, points, source_site_name, target_site_name):
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] not in (3, 4):
        raise ValueError("Point cloud must have shape (N, 3) or (N, 4)")
    source_site = data.site(source_site_name)
    target_site = data.site(target_site_name)
    source_pos = np.asarray(source_site.xpos, dtype=np.float32)
    source_rot = np.asarray(source_site.xmat, dtype=np.float32).reshape(3, 3)
    target_pos = np.asarray(target_site.xpos, dtype=np.float32)
    target_rot = np.asarray(target_site.xmat, dtype=np.float32).reshape(3, 3)

    world_points = points[:, :3] @ source_rot.T + source_pos
    target_points = (world_points - target_pos) @ target_rot
    if points.shape[1] == 4:
        target_points = np.hstack((target_points, points[:, 3:4]))
    return np.ascontiguousarray(target_points, dtype=np.float32)


def _site_points_to_world(data, points, source_site_name):
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] not in (3, 4):
        raise ValueError("Point cloud must have shape (N, 3) or (N, 4)")
    source_site = data.site(source_site_name)
    source_pos = np.asarray(source_site.xpos, dtype=np.float32)
    source_rot = np.asarray(source_site.xmat, dtype=np.float32).reshape(3, 3)
    world_points = points[:, :3] @ source_rot.T + source_pos
    if points.shape[1] == 4:
        world_points = np.hstack((world_points, points[:, 3:4]))
    return np.ascontiguousarray(world_points, dtype=np.float32)


def _tof_side_targets(
    side_sign,
    footprint_length,
    footprint_width,
    footprint_expand,
    resolution,
    z_min,
    z_max,
):
    half_length = 0.5 * footprint_length
    half_width = 0.5 * footprint_width
    outer_x = half_length + footprint_expand
    side_y = side_sign * (half_width + footprint_expand)

    xs = np.arange(-outer_x, outer_x + 0.5 * resolution, resolution)
    zs = np.arange(z_min, z_max + 0.5 * resolution, resolution)
    grid_x, grid_z = np.meshgrid(xs, zs)

    # 每个 ToF 朝车体侧面外扩 1m 的矩形平面打 ray，命中点才发布。
    points = np.column_stack((
        grid_x.ravel(),
        np.full(grid_x.size, side_y),
        grid_z.ravel(),
    ))
    return np.ascontiguousarray(points, dtype=np.float32)


def _cast_tof_rays(
    model,
    data,
    origin_world,
    site_mat,
    base_pos,
    base_mat,
    target_points_base,
    geomgroup,
    bodyexclude,
    min_range,
):
    origin_base = (origin_world - base_pos) @ base_mat
    ray_base = target_points_base - origin_base
    max_dist = np.linalg.norm(ray_base, axis=1)
    usable = max_dist > min_range
    if not np.any(usable):
        empty = np.zeros((0, 3), dtype=np.float32)
        return empty, empty

    ray_world = ray_base[usable] @ base_mat.T
    ray_world /= np.linalg.norm(ray_world, axis=1, keepdims=True)
    ray_max_dist = max_dist[usable]
    cutoff = float(np.max(ray_max_dist))

    dist = np.full(ray_world.shape[0], cutoff, dtype=np.float64)
    geomid = np.full(ray_world.shape[0], -1, dtype=np.int32)
    mujoco.mj_multiRay(
        m=model,
        d=data,
        pnt=np.array([origin_world], dtype=np.float64).T,
        vec=ray_world.astype(np.float64).ravel(),
        geomgroup=geomgroup,
        flg_static=1,
        bodyexclude=bodyexclude,
        geomid=geomid,
        dist=dist,
        normal=None,
        nray=ray_world.shape[0],
        cutoff=cutoff,
    )

    hit = (
        (geomid >= 0)
        & (dist >= min_range)
        & (dist <= ray_max_dist + 1e-3)
    )
    if not np.any(hit):
        empty = np.zeros((0, 3), dtype=np.float32)
        return empty, empty

    hit_world = origin_world + ray_world[hit] * dist[hit, np.newaxis]
    points_site = (hit_world - origin_world) @ site_mat
    points_base = (hit_world - base_pos) @ base_mat
    return (
        np.ascontiguousarray(points_site, dtype=np.float32),
        np.ascontiguousarray(points_base, dtype=np.float32),
    )


def _lidar_process_main(config, state_queue, stop_event):
    import mujoco
    import numpy as np
    import rclpy
    from rclpy._rclpy_pybind11 import RCLError
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
            "LiDAR backend '%s' is unavailable (%s); falling back to CPU. "
            "Use lidar_backend:=cpu on low-power machines to avoid this warning."
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
        "LiDAR process started: "
        f"model=MID360, "
        f"samples={lidar_pattern.samples}, "
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
            theta, phi = lidar_pattern.sample_ray_angles(
                downsample=lidar_downsample
            )
            theta = np.ascontiguousarray(theta, dtype=np.float32)
            phi = np.ascontiguousarray(phi, dtype=np.float32)
            if hasattr(lidar, "trace_points"):
                raycast_points = lidar.trace_points(
                    data,
                    theta,
                    phi,
                    min_range=lidar_min_range,
                )
            else:
                ranges = lidar.trace_rays(data, theta, phi)
                raycast_points = lidar.get_hit_points()
                valid = np.asarray(ranges) >= lidar_min_range
                raycast_points = np.ascontiguousarray(
                    np.asarray(raycast_points, dtype=np.float32)[valid],
                    dtype=np.float32,
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
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except RCLError:
                pass


def _tof_process_main(config, state_queue, stop_event):
    import mujoco
    import numpy as np
    import rclpy
    from rclpy._rclpy_pybind11 import RCLError
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
        "Side ToF process started: "
        f"{point_count} rays, "
        f"footprint +{config['tof_footprint_expand']} m "
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
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except RCLError:
                pass


class SwerveMujocoSim(Node):
    """Simulate the real WL100 control interface on top of MuJoCo."""

    def __init__(self):
        super().__init__("ats_mujoco_sim")

        self.declare_parameter("model_path", "")
        self.declare_parameter("scene_file", "")
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
        self.declare_parameter("viewer_rate_hz", 30.0)
        self.declare_parameter("truth_rate_hz", 10.0)
        self.declare_parameter("enable_lidar", True)
        self.declare_parameter("lidar_backend", "gpu")
        self.declare_parameter("lidar_model", "mid360")
        self.declare_parameter("lidar_downsample", 1)
        self.declare_parameter("lidar_rate_hz", 10.0)
        self.declare_parameter("lidar_rate_clock", "wall")
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
        self.declare_parameter("tof_footprint_length", 0.60)
        self.declare_parameter("tof_footprint_width", 0.50)
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
        self.declare_parameter("base_frame_id", "base_link")
        self.declare_parameter("pose_cmd_topic", "/simulation/PoseSub")
        self.declare_parameter("start_x", 0.0)
        self.declare_parameter("start_y", 0.0)
        self.declare_parameter("start_z", 0.18)
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

        self.model_path = self._resolve_model_path()
        self._wait_for_map_assets()
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
        if self.freeze_motion:
            self.get_logger().warn(
                "freeze_motion=true: sensors remain active while chassis motion is held"
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
        if self.viewer_rate_hz <= 0.0:
            raise ValueError("viewer_rate_hz must be positive")
        if self.truth_rate_hz <= 0.0:
            raise ValueError("truth_rate_hz must be positive")

        self.mode = MODE_SWERVE
        self.control_mode = 1
        self.motion_cmd = ChassisCommand()
        self.effective_motion_cmd = ChassisCommand()
        self.last_motion_time = time.monotonic()
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
        self.max_contact_force = 0.0
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
        self.sim_lock = threading.RLock()
        self.add_on_set_parameters_callback(self._on_set_parameters)
        self.stop_event = threading.Event()
        self.sim_thread = None
        self.dynamic_obstacle_planner_thread = None
        self.lidar_process = None
        self.lidar_state_queue = None
        self.lidar_stop_event = None
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
        self.steer_joint_ids = self._name_ids(STEER_JOINTS, "joint")
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
        self.localization_pub = self.create_publisher(
            Odometry,
            self.odom_topic,
            10,
        )
        self.lidar_odometry_topic = str(
            self.get_parameter("lidar_odometry_topic").value
        )
        self.lidar_odometry_pub = None
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
        self.truth_timer = self.create_timer(
            1.0 / self.truth_rate_hz,
            self._publish_truth,
        )
        if self.show_viewer:
            self._start_viewer()
        self._start_dynamic_obstacle_planner_thread()
        self._start_simulation_thread()

        self.get_logger().info(
            f"Loaded MuJoCo swerve model: {self.model_path}"
        )
        self.get_logger().info(
            "Motion modes: 0=swerve, 2=crab, 4=spin, 8=user_ctrl, 16=park"
        )

    def destroy_node(self):
        """Close the optional MuJoCo viewer before shutting down ROS."""
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
        if self.lidar_process is not None:
            self.lidar_process.join(timeout=2.0)
            if self.lidar_process.is_alive():
                self.lidar_process.terminate()
                self.lidar_process.join(timeout=1.0)
        if self.tof_process is not None:
            self.tof_process.join(timeout=2.0)
            if self.tof_process.is_alive():
                self.tof_process.terminate()
                self.tof_process.join(timeout=1.0)
        if self.lidar_state_queue is not None:
            self.lidar_state_queue.cancel_join_thread()
            self.lidar_state_queue.close()
        if self.tof_state_queue is not None:
            self.tof_state_queue.cancel_join_thread()
            self.tof_state_queue.close()
        if self.sim_thread is not None and self.sim_thread.is_alive():
            self.sim_thread.join(timeout=1.0)
        with self.sim_lock:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
        return super().destroy_node()

    def _get_bool_parameter(self, name):
        value = self.get_parameter(name).value
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes", "on")
        return bool(value)

    def _start_viewer(self):
        try:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=True,
                show_right_ui=True,
            )
            self.viewer.opt.geomgroup[3] = 1
            self.get_logger().info("MuJoCo viewer started")
        except Exception as exc:
            self.viewer = None
            self.get_logger().error(f"Failed to start MuJoCo viewer: {exc}")

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
        for name in ("floor", "terrain", "rmuc_2026_field"):
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
                    f"Dynamic obstacle joint not found in model: {joint_name}"
                )
                continue
            if self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
                self.get_logger().warn(
                    f"Dynamic obstacle joint is not freejoint: {joint_name}"
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
                f"Dynamic obstacles enabled: {len(obstacles)} runtime-driven bodies"
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
                f"Dynamic obstacle map sampler disabled: {exc}"
            )
            return None

    def _set_initial_pose_from_parameters(self):
        qpos_addr = self.free_qpos_addr
        qvel_addr = self.free_dof_addr
        self.data.qpos[qpos_addr:qpos_addr + 3] = np.array(
            [
                float(self.get_parameter("start_x").value),
                float(self.get_parameter("start_y").value),
                float(self.get_parameter("start_z").value),
            ],
            dtype=np.float64,
        )
        self.data.qpos[qpos_addr + 3:qpos_addr + 7] = _quat_wxyz_from_yaw(
            float(self.get_parameter("start_yaw").value)
        )
        self.data.qvel[qvel_addr:qvel_addr + 6] = 0.0

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
        if self.lidar_downsample <= 0:
            raise ValueError("lidar_downsample must be positive")
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
            self.lidar_state_queue = ctx.Queue(maxsize=1)
            self.lidar_stop_event = ctx.Event()
            self.lidar_process = ctx.Process(
                target=_lidar_process_main,
                args=(config, self.lidar_state_queue, self.lidar_stop_event),
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
        self.lidar_state_thread = threading.Thread(
            target=self._lidar_state_loop,
            name="swerve_lidar_state_sender",
            daemon=True,
        )
        self.lidar_state_thread.start()
        if self.lidar_enabled:
            self.get_logger().info(
                f"MID360 LiDAR process enabled: "
                f"downsample={self.lidar_downsample}, "
                f"{self.lidar_rate_hz} Hz, "
                f"clock={self.lidar_rate_clock}, "
                f"state={self.lidar_state_rate_hz} Hz, "
                f"site={self.lidar_site}, frame_site={self.lidar_frame_site}, "
                f"topic={self.lidar_topic}, "
                f"frame={self.lidar_frame_id}, backend={self.lidar_backend}"
            )
        if self.tof_enabled:
            left_topic = str(self.get_parameter("left_tof_topic").value)
            right_topic = str(self.get_parameter("right_tof_topic").value)
            self.get_logger().info(
                "Side ToF enabled: "
                f"range={tof_min_range}-{tof_range} m, "
                f"resolution={tof_width}x{tof_height}, "
                f"footprint={tof_footprint_length}x{tof_footprint_width}"
                f" + {tof_footprint_expand} m"
                f" @ {tof_footprint_resolution} m, "
                f"z={tof_footprint_z_min}-{tof_footprint_z_max} m, "
                f"left={left_topic}, right={right_topic}"
            )

    def _send_lidar_state(self):
        if (
            self.lidar_process is not None
            and self.lidar_process.exitcode is not None
        ):
            if not self.lidar_process_dead_warned:
                self.get_logger().error(
                    "LiDAR process exited with code "
                    f"{self.lidar_process.exitcode}"
                )
                self.lidar_process_dead_warned = True
        if (
            self.tof_process is not None
            and self.tof_process.exitcode is not None
        ):
            if not self.tof_process_dead_warned:
                self.get_logger().error(
                    "ToF process exited with code "
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

    def _start_simulation_thread(self):
        self.sim_thread = threading.Thread(
            target=self._simulation_loop,
            name="swerve_mujoco_physics",
            daemon=True,
        )
        self.sim_thread.start()

    def _start_dynamic_obstacle_planner_thread(self):
        if not self.dynamic_obstacles:
            return
        rate_hz = max(1.0, self.dynamic_obstacle_config.replan_rate_hz * 4.0)
        self.get_logger().info(
            "Dynamic obstacle planner thread enabled: "
            f"{rate_hz:.1f} Hz scheduler"
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

    def _simulation_loop(self):
        while rclpy.ok() and not self.stop_event.is_set():
            step_start = time.perf_counter()
            with self.sim_lock:
                if self.viewer is not None and not self.viewer.is_running():
                    self.viewer.close()
                    self.viewer = None
                    self.stop_event.set()
                    break

                if self.viewer is None:
                    self._step_simulation_locked()
                else:
                    with self.viewer.lock():
                        self._step_simulation_locked()
                    now = time.perf_counter()
                    if (
                        now - self.last_viewer_sync_time
                        >= 1.0 / self.viewer_rate_hz
                    ):
                        self.viewer.sync()
                        self.last_viewer_sync_time = now

            step_time = time.perf_counter() - step_start
            sleep_time = self.model.opt.timestep - step_time
            if sleep_time > 0.0:
                time.sleep(sleep_time)

# TAG 物理仿真核心步骤：计算控制目标、应用到 Mujoco、电动障碍物更新、推进仿真
    def _step_simulation_locked(self):
        self.current_targets = self._compute_targets()
        self._apply_targets(self.current_targets, self.hard_stop_requested)
        self._update_dynamic_obstacles_locked(float(self.data.time))
        mujoco.mj_step(self.model, self.data)
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
            self.last_motion_time = time.monotonic()

    def _on_set_parameters(self, parameters):
        for parameter in parameters:
            if parameter.name != "freeze_motion":
                continue
            if not isinstance(parameter.value, bool):
                return SetParametersResult(
                    successful=False,
                    reason="freeze_motion must be a boolean",
                )
            with self.sim_lock:
                self.freeze_motion = parameter.value
            self.get_logger().warn(
                "freeze_motion=%s: chassis execution %s while sensors remain active",
                parameter.value,
                "held" if parameter.value else "enabled",
            )
        return SetParametersResult(successful=True)

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
            self.last_motion_time = time.monotonic()

    def _speed_control_callback(self, msg):
        with self.sim_lock:
            self.direct_speeds["lf"] = float(msg.lf_speed)
            self.direct_speeds["lr"] = float(msg.lr_speed)
            self.direct_speeds["rf"] = float(msg.rf_speed)
            self.direct_speeds["rr"] = float(msg.rr_speed)
            self.last_speed_time = time.monotonic()

    def _steer_control_callback(self, msg):
        with self.sim_lock:
            self.direct_steer_angles["lf"] = radians(float(msg.lf_steer_angle))
            self.direct_steer_angles["lr"] = radians(float(msg.lr_steer_angle))
            self.direct_steer_angles["rf"] = radians(float(msg.rf_steer_angle))
            self.direct_steer_angles["rr"] = radians(float(msg.rr_steer_angle))

    def _motion_mode_callback(self, request, response):
        mode = int(request.cmd_ctl)
        if mode not in VALID_MOTION_MODES:
            self.get_logger().warn(f"Reject unknown motion mode: {mode}")
            response.cmd_ack = CMD_ACK_FAIL
            return response

        with self.sim_lock:
            self.mode = mode
            self.motion_cmd = ChassisCommand()
            self.last_motion_time = time.monotonic()
        self.get_logger().info(
            f"Motion mode set to {mode} ({MODE_NAMES[mode]})"
        )
        response.cmd_ack = CMD_ACK_FINISH
        return response

    def _control_mode_callback(self, request, response):
        with self.sim_lock:
            self.control_mode = int(request.cmd_ctl)
        response.cmd_ack = CMD_ACK_FINISH
        return response

    def _compute_targets(self):
        now = time.monotonic()

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
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if not contact_is_violation(
                geom1,
                geom2,
                self.robot_geom_ids,
                self.wheel_geom_ids,
                self.ground_geom_ids,
            ):
                continue
            self.contact_violation_count += 1
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, contact_index, force)
            self.max_contact_force = max(
                self.max_contact_force,
                float(np.linalg.norm(force[:3])),
            )

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
                np.zeros(3),
                (0.0, 0.0, 0.0, 1.0),
            ),
            self._make_transform(
                stamp,
                self.robot_base_frame_id,
                self.lidar_tf_frame_id,
                MID360_TRANSLATION,
                _quat_xyzw_from_rpy(*MID360_RPY),
            )
        ])

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
            x_pos, y_pos = WHEEL_POSITIONS[name]
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
        message.header.stamp = self.get_clock().now().to_msg()
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

    def _publish_gimbal_status(self):
        _, body_rotation = self._body_pose_locked()
        body_yaw = float(np.arctan2(body_rotation[1, 0], body_rotation[0, 0]))
        message = GimbalYawStatus()
        message.header.stamp = self.get_clock().now().to_msg()
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


def main(args=None):
    """Run the ROS2 MuJoCo simulation node."""
    rclpy.init(args=args)
    node = SwerveMujocoSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        _shutdown_rclpy_if_needed()


if __name__ == "__main__":
    main()
