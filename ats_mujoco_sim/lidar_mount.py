"""Mid360 mount geometry on the chassis body.

Pure numpy, no ROS or MuJoCo import, so the single source of truth for the
mount can be unit-tested and reused by the model patcher, the static TF and
the Point-LIO gravity override.  One function produces all three, so the
gravity vector, the TF roll/pitch and the MuJoCo quaternion cannot drift apart
(report appendix A.21, consistency item 1).

Frames: body = the MuJoCo ``base_link`` body (published as
``gimbal_yaw_odom``).  Sensor = the ``front_mid360`` frame.
"""

from dataclasses import dataclass
import math

import numpy as np

GRAVITY_M_S2 = 9.81

# Point-LIO ``mapping.extrinsic_T`` is the lidar origin expressed in the IMU
# frame and ``extrinsic_R`` is identity, so the IMU origin in the lidar frame
# is the negated vector (node_params.yaml: [-0.011, -0.02329, 0.04412]).
MID360_IMU_IN_LIDAR = (0.011, 0.02329, -0.04412)

# Chassis half extents after the 2026-10-23 axis correction (report 2.1.8):
# x (heading) 0.581 m, y 0.571 m.
BODY_SIZE_X = 0.581
BODY_SIZE_Y = 0.571
# The report mounts the sensor 0.05 m inside the midpoint of the front (+x) edge.
EDGE_INSET = 0.05
# Height of the MuJoCo base_link origin above the wheel contact plane.
BASE_LINK_HEIGHT = 0.18

TILT_DIRECTIONS = ("front", "left")


@dataclass(frozen=True)
class MountPose:
    """Sensor pose in the body frame."""

    position: tuple
    rotation: tuple  # row-major 3x3

    def rotation_matrix(self):
        return np.array(self.rotation, dtype=np.float64).reshape(3, 3)

    def position_vector(self):
        return np.array(self.position, dtype=np.float64)

    def quat_wxyz(self):
        """MuJoCo ``quat`` attribute order."""
        x, y, z, w = _matrix_to_quat_xyzw(self.rotation_matrix())
        return (w, x, y, z)

    def quat_xyzw(self):
        return _matrix_to_quat_xyzw(self.rotation_matrix())

    def rpy(self):
        """REP-103 roll, pitch, yaw (R = Rz(yaw) Ry(pitch) Rx(roll))."""
        rot = self.rotation_matrix()
        pitch = -math.asin(max(-1.0, min(1.0, rot[2, 0])))
        if abs(rot[2, 0]) < 1.0 - 1e-9:
            roll = math.atan2(rot[2, 1], rot[2, 2])
            yaw = math.atan2(rot[1, 0], rot[0, 0])
        else:  # gimbal lock, fold roll into yaw
            roll = 0.0
            yaw = math.atan2(-rot[0, 1], rot[1, 1])
        return (roll, pitch, yaw)

    def gravity_in_sensor(self, g=GRAVITY_M_S2):
        """Gravity vector a stationary IMU would see in the sensor frame.

        This is what Point-LIO's ``mapping.gravity`` must equal for the
        extrinsic_R = I Mid360 (initialisation aligns the measured gravity
        direction to this parameter).
        """
        world_down_in_body = np.array([0.0, 0.0, -float(g)])
        return tuple(float(v) for v in self.rotation_matrix().T @ world_down_in_body)

    def imu_position_in_body(self, imu_in_lidar=MID360_IMU_IN_LIDAR):
        return tuple(
            float(v)
            for v in self.position_vector()
            + self.rotation_matrix() @ np.array(imu_in_lidar, dtype=np.float64)
        )


def _matrix_to_quat_xyzw(mat):
    trace = float(mat[0, 0] + mat[1, 1] + mat[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (mat[2, 1] - mat[1, 2]) / s
        y = (mat[0, 2] - mat[2, 0]) / s
        z = (mat[1, 0] - mat[0, 1]) / s
    elif mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
        s = math.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]) * 2.0
        w = (mat[2, 1] - mat[1, 2]) / s
        x = 0.25 * s
        y = (mat[0, 1] + mat[1, 0]) / s
        z = (mat[0, 2] + mat[2, 0]) / s
    elif mat[1, 1] > mat[2, 2]:
        s = math.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2]) * 2.0
        w = (mat[0, 2] - mat[2, 0]) / s
        x = (mat[0, 1] + mat[1, 0]) / s
        y = 0.25 * s
        z = (mat[1, 2] + mat[2, 1]) / s
    else:
        s = math.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1]) * 2.0
        w = (mat[1, 0] - mat[0, 1]) / s
        x = (mat[0, 2] + mat[2, 0]) / s
        y = (mat[1, 2] + mat[2, 1]) / s
        z = 0.25 * s
    return (float(x), float(y), float(z), float(w))


def _rotation_about_axis(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    kx, ky, kz = axis
    cross = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
    return (
        np.eye(3) * math.cos(angle)
        + math.sin(angle) * cross
        + (1.0 - math.cos(angle)) * np.outer(axis, axis)
    )


def chassis_mount(
    x=BODY_SIZE_X / 2.0 - EDGE_INSET,
    y=0.0,
    height_above_ground=0.20,
    tilt_direction="front",
    tilt_deg=35.0,
    body_height=BASE_LINK_HEIGHT,
):
    """Mount whose sensor +z axis leans ``tilt_deg`` towards ``tilt_direction``.

    The default position is the midpoint of the front (+x) edge pulled 0.05 m
    inwards: x = 0.2905 - 0.05 = 0.2405 m, y = 0 (report 2.1.8).

    ``front`` leans towards body +x (Mid360 +x points forward and down, the
    report's "pitch +30 deg"); ``left`` leans towards body +y.  The sensor
    azimuth zero stays on body +x in both cases: the Mid360 pattern is a
    360 degree sweep, so only the lean matters.  ``height_above_ground`` is
    the optical centre (the ``front_mid360`` frame origin) above the ground.
    """
    if tilt_direction not in TILT_DIRECTIONS:
        raise ValueError(
            f"tilt_direction must be one of {TILT_DIRECTIONS}, got {tilt_direction!r}"
        )
    lean = np.array([1.0, 0.0, 0.0]) if tilt_direction == "front" else np.array([0.0, 1.0, 0.0])
    axis = np.cross(np.array([0.0, 0.0, 1.0]), lean)
    rotation = _rotation_about_axis(axis, math.radians(float(tilt_deg)))
    # Remove sub-ulp noise so exact zeros stay zero in generated XML/TF text.
    rotation = np.where(np.abs(rotation) < 1e-15, 0.0, rotation)
    return MountPose(
        position=(float(x), float(y), float(height_above_ground) - float(body_height)),
        rotation=tuple(float(v) for v in rotation.reshape(-1)),
    )


def legacy_gimbal_mount():
    """The pre-2026-10-23 gimbal mount, kept to reproduce old truth output."""
    roll, pitch, yaw = math.radians(75.0), 0.0, math.radians(-161.0)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rot = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    return MountPose(
        position=(-0.1, 0.245, 0.325),
        rotation=tuple(float(v) for v in rot.reshape(-1)),
    )
