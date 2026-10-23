"""Text-level MuJoCo model patches driven by launch parameters.

The lidar process reloads the model from a file path, so the mount override
must exist on disk rather than as an in-memory spec edit.  The versioned XML
is never modified; callers write the patched text next to it.
"""

import re

from ats_mujoco_sim.lidar_mount import MountPose

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
