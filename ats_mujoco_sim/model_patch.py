"""Text-level MuJoCo model patches driven by launch parameters.

The lidar process reloads the model from a file path, so the mount override
must exist on disk rather than as an in-memory spec edit.  The versioned XML
is never modified; callers write the patched text next to it.
"""

import re
import math

from ats_mujoco_sim.lidar_mount import MountPose


def configure_wheel_speed_limit(model, actuator_names, speed_mps, radius_m):
    """Match velocity actuator ctrlrange to the configured wheel linear limit.

    XML's old 39.27 rad/s range must not silently clamp a 5 m/s request.
    Torque/force, acceleration and steering limits are deliberately preserved.
    """
    if not math.isfinite(speed_mps) or not math.isfinite(radius_m):
        raise ValueError("wheel speed/radius must be finite")
    if speed_mps <= 0.0 or radius_m <= 0.0:
        raise ValueError("wheel speed/radius must be positive")
    rate = float(speed_mps) / float(radius_m)
    for name in actuator_names:
        actuator_id = model.actuator(name).id
        model.actuator_ctrlrange[actuator_id] = (-rate, rate)
        model.actuator_ctrllimited[actuator_id] = True
    return rate

_LIDAR_BODY_RE = re.compile(
    r'(<body\s+name="lidar_link"\s+)pos="[^"]*"(\s+)quat="[^"]*"'
)

# Names the node looks up when localization_source is "lio".
IMU_SITE_NAME = "mid360_imu_site"
IMU_GYRO_SENSOR = "mid360_imu_gyro"
IMU_ACCEL_SENSOR = "mid360_imu_accel"


def _fmt(values):
    return " ".join(f"{float(v):.12g}" for v in values)


def patch_lidar_mount(xml_text, pose: MountPose):
    """Replace the pos/quat of the ``lidar_link`` body with ``pose``."""
    replacement = (
        r'\1pos="' + _fmt(pose.position) + r'"\2quat="' + _fmt(pose.quat_wxyz()) + '"'
    )
    patched, count = _LIDAR_BODY_RE.subn(replacement, xml_text, count=1)
    if count != 1:
        raise ValueError(
            'model has no <body name="lidar_link" pos=... quat=...>; '
            "the lidar mount parameters cannot be applied"
        )
    return patched
