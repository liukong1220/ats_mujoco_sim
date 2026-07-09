"""Simple four-swerve-drive kinematics for the WL100-style chassis."""

from dataclasses import dataclass
from math import atan2, cos, degrees, hypot, pi, radians, sin


MODE_SWERVE = 0
MODE_CRAB = 2
MODE_SPIN = 4
MODE_USER_CTRL = 8
MODE_PARK = 16

MODE_NAMES = {
    MODE_SWERVE: "swerve",
    MODE_CRAB: "crab",
    MODE_SPIN: "spin",
    MODE_USER_CTRL: "user_ctrl",
    MODE_PARK: "park",
}

WHEEL_ORDER = ("lf", "lr", "rf", "rr")

# Keep these positions in sync with the MuJoCo chassis XML.
WHEEL_POSITIONS = {
    "lf": (0.225, 0.225),
    "lr": (-0.225, 0.225),
    "rf": (0.225, -0.225),
    "rr": (-0.225, -0.225),
}

PARK_ANGLES_RAD = {
    "lf": radians(45.0),
    "lr": radians(-45.0),
    "rf": radians(-45.0),
    "rr": radians(45.0),
}


@dataclass
class ChassisCommand:
    """Body-frame chassis command, using meters and radians."""

    linear_x: float = 0.0
    linear_y: float = 0.0
    angular_z: float = 0.0


@dataclass
class WheelTarget:
    """Target for one swerve module."""

    name: str
    steer_angle: float
    wheel_speed: float

    @property
    def steer_angle_deg(self):
        """Return the steering target in degrees."""
        return degrees(self.steer_angle)


def normalize_angle(angle):
    """Wrap an angle to [-pi, pi]."""
    while angle > pi:
        angle -= 2.0 * pi
    while angle < -pi:
        angle += 2.0 * pi
    return angle


def optimize_steer_angle(angle, speed, previous_angle):
    """Flip wheel speed if that avoids a long steering rotation."""
    angle = normalize_angle(angle)
    previous_angle = normalize_angle(previous_angle)
    delta = normalize_angle(angle - previous_angle)

    if delta > pi / 2.0:
        return normalize_angle(angle - pi), -speed
    if delta < -pi / 2.0:
        return normalize_angle(angle + pi), -speed
    return angle, speed


def chassis_to_wheel_targets(command, previous_angles=None):
    """Convert a chassis velocity command to four wheel targets."""
    previous_angles = previous_angles or {}
    targets = []

    for name in WHEEL_ORDER:
        x_pos, y_pos = WHEEL_POSITIONS[name]
        wheel_vx = command.linear_x - command.angular_z * y_pos
        wheel_vy = command.linear_y + command.angular_z * x_pos
        speed = hypot(wheel_vx, wheel_vy)

        if speed < 1e-6:
            angle = previous_angles.get(name, 0.0)
            speed = 0.0
        else:
            angle = atan2(wheel_vy, wheel_vx)
            angle, speed = optimize_steer_angle(
                angle,
                speed,
                previous_angles.get(name, 0.0),
            )

        targets.append(WheelTarget(name, angle, speed))

    return targets


def park_wheel_targets():
    """Return a simple X-lock wheel pattern for parking."""
    return [
        WheelTarget(name, PARK_ANGLES_RAD[name], 0.0)
        for name in WHEEL_ORDER
    ]

def mode_to_wheel_targets(mode, command, previous_angles=None):
    """Apply WL100 motion-mode rules and return wheel targets."""
    previous_angles = previous_angles or {}

    if mode == MODE_SWERVE:
        effective = ChassisCommand(
            command.linear_x,
            command.linear_y,
            command.angular_z,
        )
        return effective, chassis_to_wheel_targets(effective, previous_angles)

    if mode == MODE_CRAB:
        effective = ChassisCommand(command.linear_x, command.linear_y, 0.0)
        return effective, chassis_to_wheel_targets(effective, previous_angles)

    if mode == MODE_SPIN:
        effective = ChassisCommand(0.0, 0.0, command.angular_z)
        return effective, chassis_to_wheel_targets(effective, previous_angles)

    if mode == MODE_PARK:
        effective = ChassisCommand()
        return effective, park_wheel_targets()

    effective = ChassisCommand()
    zero_targets = [
        WheelTarget(name, previous_angles.get(name, 0.0), 0.0)
        for name in WHEEL_ORDER
    ]
    return effective, zero_targets


def estimate_chassis_command(targets):
    """Estimate chassis motion from four wheel speed and steering targets."""
    if not targets:
        return ChassisCommand()

    linear_x = 0.0
    linear_y = 0.0
    angular_z = 0.0

    for target in targets:
        x_pos, y_pos = WHEEL_POSITIONS[target.name]
        wheel_vx = target.wheel_speed * cos(target.steer_angle)
        wheel_vy = target.wheel_speed * sin(target.steer_angle)
        linear_x += wheel_vx
        linear_y += wheel_vy

        radius_sq = x_pos * x_pos + y_pos * y_pos
        if radius_sq > 1e-9:
            angular_z += (-wheel_vx * y_pos + wheel_vy * x_pos) / radius_sq

    count = float(len(targets))
    return ChassisCommand(
        linear_x / count,
        linear_y / count,
        angular_z / count,
    )
