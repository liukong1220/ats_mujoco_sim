"""Math, quaternion, PointCloud2, and ToF ray-casting helpers for MuJoCo simulation."""

from math import cos, degrees, pi, radians, sin
import json
import multiprocessing as mp
from pathlib import Path
import queue
import sys
import tempfile
import threading
import time

from builtin_interfaces.msg import Time
from carstatemsgs.msg import CarState
from ats_navigation_interfaces.msg import SwerveTelemetry
from ats_mujoco_sim.spawn_pose import quaternion_from_rpy
from ats_navigation_interfaces.msg import GimbalYawStatus
from ats_navigation_interfaces.msg import YawAuthorityRequest
from geometry_msgs.msg import TransformStamped
import mujoco
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool
from std_msgs.msg import String
from std_srvs.srv import Trigger
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
from ats_mujoco_sim import livox_scan
from ats_mujoco_sim.parallel_lidar import ParallelRayTracer
from ats_mujoco_sim.parallel_lidar import available_cpu_ids
from ats_mujoco_sim.parallel_lidar import pin_lidar_process
from ats_mujoco_sim.parallel_lidar import resolve_downsample
from ats_mujoco_sim.parallel_lidar import resolve_thread_count
from ats_mujoco_sim.parallel_lidar import select_lidar_cpu_ids
from ats_mujoco_sim.kinematics import ChassisCommand
from ats_mujoco_sim.kinematics import MAX_STEER_RATE_RADPS
from ats_mujoco_sim.kinematics import MAX_WHEEL_SPEED_MPS
from ats_mujoco_sim.kinematics import MODE_NAMES
from ats_mujoco_sim.kinematics import MODE_SWERVE
from ats_mujoco_sim.kinematics import MODE_USER_CTRL
from ats_mujoco_sim.kinematics import WHEEL_ORDER
from ats_mujoco_sim.kinematics import WHEEL_RADIUS_M
from ats_mujoco_sim.kinematics import WheelTarget
from ats_mujoco_sim.kinematics import classify_mujoco_contact
from ats_mujoco_sim.kinematics import contact_is_violation
from ats_mujoco_sim.kinematics import estimate_chassis_command
from ats_mujoco_sim.kinematics import mode_to_wheel_targets
from ats_mujoco_sim.kinematics import normalize_angle
from ats_mujoco_sim.kinematics import rate_limit_angle
from ats_mujoco_sim.kinematics import wheel_positions_from_model
from ats_mujoco_sim.lidar_mount import MID360_IMU_IN_LIDAR
from ats_mujoco_sim.lidar_mount import TILT_DIRECTIONS
from ats_mujoco_sim.lidar_mount import chassis_mount
from ats_mujoco_sim.lidar_mount import legacy_gimbal_mount
from ats_mujoco_sim.map_metadata import load_occupancy_image
from ats_mujoco_sim.map_metadata import read_map_metadata
from ats_mujoco_sim.mid360_model import MID360_MESH_RELATIVE_PATH
from ats_mujoco_sim.model_patch import IMU_ACCEL_SENSOR
from ats_mujoco_sim.model_patch import IMU_GYRO_SENSOR
from ats_mujoco_sim.model_patch import patch_lidar_mount
from ats_mujoco_sim.model_patch import configure_wheel_speed_limit
from ats_mujoco_sim.runtime_fault_state import apply_runtime_fault_parameters
from ats_mujoco_sim.runtime_fault_state import pointcloud_payload
from ats_mujoco_sim.runtime_fault_state import resolve_lidar_return
from ats_mujoco_sim.obstacle_fixture import ObstacleFixture
from ats_mujoco_sim.sensor_noise import ImuNoiseConfig
from ats_mujoco_sim.sensor_noise import ImuNoiseModel
from ats_mujoco_sim.sensor_noise import RangeNoiseConfig
from ats_mujoco_sim.sensor_noise import RangeNoiseModel


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
# Must match the lidar_link body in models/*.xml and the xmacro livox pose.
MID360_TRANSLATION = np.array([-0.1, 0.245, 0.325], dtype=np.float64)
MID360_RPY = (radians(75.0), 0.0, radians(-161.0))


def _refresh_viewer_data(model, source, shadow):
    """Bring the window's private MjData to the simulation's current state.

    The Python bindings of MuJoCo 3.4 have no ``mj_copyData`` (the first version of this fix used
    it and the physics thread died with AttributeError the moment a window was requested,
    2026-10-29), so the integration state is copied and the derived quantities the renderer reads
    (body/geom poses, contacts) are recomputed on the private copy only.
    """
    mujoco.mj_copyState(model, source, shadow, mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model, shadow)


def _shutdown_rclpy_if_needed():
    """Shut down the default context only if it is still valid."""
    rclpy.get_default_context().try_shutdown()


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


def _import_multi_ray():
    """取 `mj_multiRay` 的 ABI 兼容包装。

    `mujoco_lidar` 与本节点同为可选安装，因此沿用 `_import_mujoco_lidar` 的
    路径修复再导入，避免模块加载期就硬依赖它。
    """
    try:
        from mujoco_lidar.multiray_compat import multi_ray
    except ModuleNotFoundError as exc:
        if exc.name != "mujoco_lidar":
            raise
        _import_mujoco_lidar()
        from mujoco_lidar.multiray_compat import multi_ray
    return multi_ray


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


def _timeout_clock(node):
    """Command-freshness clock: simulation time in LIO mode, monotonic wall time otherwise.

    Takes the node explicitly (and tolerates objects without ``lio_mode``) so
    code paths that unit tests drive through a stand-in namespace keep working.
    """
    if getattr(node, "lio_mode", False):
        return float(node.data.time)
    return time.monotonic()


def _sim_time_to_stamp(sim_time):
    """builtin_interfaces/Time for a simulation time in seconds."""
    total_ns = int(round(float(sim_time) * 1.0e9))
    stamp = Time()
    stamp.sec = total_ns // 1_000_000_000
    stamp.nanosec = total_ns % 1_000_000_000
    return stamp


def _livox_pointcloud_message(frame_id):
    """PointCloud2 skeleton for the time-resolved Mid360 cloud.

    The layout is the contract with ``ats_livox_adapter``: x, y, z, intensity
    as float32, ``offset_time`` as uint32 nanoseconds from the header stamp,
    ``line`` and ``tag`` as uint8, 24 bytes per point.
    """
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs.msg import PointField

    msg = PointCloud2()
    msg.header.frame_id = frame_id
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name="offset_time", offset=16, datatype=PointField.UINT32, count=1),
        PointField(name="line", offset=20, datatype=PointField.UINT8, count=1),
        PointField(name="tag", offset=21, datatype=PointField.UINT8, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = livox_scan.POINT_STEP
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
    # An empty cloud still publishes with a legal header and stamp so an
    # occluded sensor keeps its cadence instead of looking like a dead topic.
    payload = pointcloud_payload(points, msg.point_step, intensity)
    msg.header.stamp = stamp
    msg.width = payload.width
    msg.row_step = payload.row_step
    msg.data = payload.data
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
    # 与 LiDAR 后端共用同一个 ABI 兼容层：`normal` 槽位只在部分 MuJoCo
    # 版本存在，按版本写死会让第一次 raycast 抛 TypeError。
    multi_ray = _import_multi_ray()
    multi_ray(
        model,
        data,
        np.array([origin_world], dtype=np.float64).T,
        ray_world.astype(np.float64).ravel(),
        geomgroup,
        1,
        bodyexclude,
        geomid,
        dist,
        ray_world.shape[0],
        cutoff,
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



def _relative_site_transform(data, source_site_name, target_site_name):
    """Constant rotation and translation taking source-site points to the target site."""
    source = data.site(source_site_name)
    target = data.site(target_site_name)
    source_rot = np.asarray(source.xmat, dtype=np.float64).reshape(3, 3)
    target_rot = np.asarray(target.xmat, dtype=np.float64).reshape(3, 3)
    rot = target_rot.T @ source_rot
    trans = target_rot.T @ (
        np.asarray(source.xpos, dtype=np.float64) - np.asarray(target.xpos, dtype=np.float64)
    )
    return rot, trans
