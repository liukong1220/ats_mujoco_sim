"""Initial parking attitude only; no change to subsequent dynamics or controls."""
import math


def quaternion_from_rpy(roll, pitch, yaw):
    """Return MuJoCo wxyz for Rz(yaw) Ry(pitch) Rx(roll), radians."""
    if not all(math.isfinite(value) for value in (roll, pitch, yaw)):
        raise ValueError("spawn attitude must be finite")
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)
