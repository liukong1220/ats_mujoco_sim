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

# Real vehicle: wheel centres at +-0.270 m from the chassis centre (`wheel_base_x/y` of
# ats_swerve_mpc_reality.yaml).  The simulator reads the wheel positions of the MuJoCo model it
# loads (wheel_positions_from_model); these are the defaults when no model is at hand.
WHEEL_OFFSET_X_M = 0.270
WHEEL_OFFSET_Y_M = 0.270
WHEEL_RADIUS_M = 0.0425
RUBBER_THICKNESS_M = 0.010
DRIVE_GEAR_RATIO = 1.0
STEER_GEAR_RATIO = 1.0
MOTOR_MAX_RPM = 1348.1359885  # User's provisional 5 m/s after the 1.2 redundancy; not a measured RPM.
STEER_MAX_RPM = 120.0
ACTUATOR_REDUNDANCY = 1.2
MAX_WHEEL_SPEED_MPS = (
    2.0 * pi * WHEEL_RADIUS_M * MOTOR_MAX_RPM
    / (60.0 * DRIVE_GEAR_RATIO * ACTUATOR_REDUNDANCY)
)
MAX_STEER_RATE_RADPS = (
    2.0 * pi * STEER_MAX_RPM
    / (60.0 * STEER_GEAR_RATIO * ACTUATOR_REDUNDANCY)
)

WHEEL_POSITIONS = {
    "lf": (WHEEL_OFFSET_X_M, WHEEL_OFFSET_Y_M),
    "lr": (-WHEEL_OFFSET_X_M, WHEEL_OFFSET_Y_M),
    "rf": (WHEEL_OFFSET_X_M, -WHEEL_OFFSET_Y_M),
    "rr": (-WHEEL_OFFSET_X_M, -WHEEL_OFFSET_Y_M),
}

STEER_BODY_PREFIX = {"lf": "front_left", "lr": "rear_left", "rf": "front_right", "rr": "rear_right"}


def wheel_positions_from_model(model):
    """Wheel centres (x, y) in the base_link frame, read from the steer-module bodies of ``model``."""
    import mujoco  # local import: this module is otherwise MuJoCo-free

    positions = {}
    for name, prefix in STEER_BODY_PREFIX.items():
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_steer_link")
        if body < 0:
            raise ValueError(f"model has no body {prefix}_steer_link")
        positions[name] = (float(model.body_pos[body][0]), float(model.body_pos[body][1]))
    return positions

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


def scale_planar_speed(vx, vy, limit):
    """Scale (vx, vy) so that its norm is at most ``limit``; ``limit <= 0`` disables the cap.

    The real exit limit (chassis_vel_transform ``max_linear_speed``) acts on the planar vector,
    so the commanded direction is kept; clamping x and y separately would bend it (report 5.1.9 C).
    """
    speed = hypot(vx, vy)
    if limit <= 0.0 or speed <= limit:
        return vx, vy
    scale = limit / speed
    return vx * scale, vy * scale


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


def rate_limit_angle(current_angle, target_angle, max_delta):
    """Move an angle toward a target by at most max_delta radians."""
    delta = normalize_angle(target_angle - current_angle)
    limited_delta = min(max(delta, -max_delta), max_delta)
    return normalize_angle(current_angle + limited_delta), abs(delta) > max_delta


def classify_mujoco_contact(
    geom1,
    geom2,
    geom1_name,
    geom2_name,
    robot_geom_ids,
    wheel_geom_ids,
    ground_geom_ids,
    contact_pos,
    contact_frame,
    contact_dist,
    force_norm,
    wheel_center_z=None,
    wheel_radius=WHEEL_RADIUS_M,
    support_normal_min_z=0.78,
):
    """Classify a MuJoCo contact pair by geometry role, normal orientation, and force.

    MuJoCo's ``contact.frame[:3]`` is the unit normal directed from ``geom1`` to
    ``geom2``.  We orient it as ``n_env_to_robot`` so upward floor support has
    ``nz ~ +1``, vertical side walls have ``nz ~ 0``, and tunnel ceilings have
    ``nz ~ -1``.

    Returns ``None`` for environment-only pairs, otherwise a dict with
    ``category``, ``is_illegal``, ``robot_role``, ``env_category``, ``nz``,
    ``robot_geom``, ``env_geom``, ``robot_geom_name``, and ``env_geom_name``.
    """
    g1_robot = geom1 in robot_geom_ids
    g2_robot = geom2 in robot_geom_ids
    if not g1_robot and not g2_robot:
        return None
    if g1_robot and g2_robot:
        return {
            "category": "illegal_self_contact",
            "is_illegal": True,
            "robot_role": "self",
            "env_category": "robot",
            "nz": float(contact_frame[2]),
            "robot_geom": geom1,
            "env_geom": geom2,
            "robot_geom_name": geom1_name,
            "env_geom_name": geom2_name,
        }

    if g1_robot:
        robot_geom, env_geom = geom1, geom2
        robot_name, env_name = geom1_name, geom2_name
        nz = -float(contact_frame[2])
    else:
        robot_geom, env_geom = geom2, geom1
        robot_name, env_name = geom2_name, geom1_name
        nz = float(contact_frame[2])

    robot_role = "wheel" if robot_geom in wheel_geom_ids else "chassis"
    env_lower = (env_name or "").lower()
    if env_geom in ground_geom_ids:
        env_category = "ground"
    elif (env_name or "").startswith("audited_roof_"):
        env_category = "audited_roof_box"
    elif any(token in env_lower for token in ("dynamic", "obstacle", "fixture", "test_box")):
        env_category = "dynamic_obstacle"
    else:
        env_category = "other_env"

    result = {
        "robot_role": robot_role,
        "env_category": env_category,
        "nz": nz,
        "robot_geom": robot_geom,
        "env_geom": env_geom,
        "robot_geom_name": robot_name,
        "env_geom_name": env_name,
    }

    # MuJoCo speculative margin pair with positive separation and zero reaction force.
    if contact_dist > 0.0 and force_norm <= 1e-6:
        result["category"] = "speculative_margin"
        result["is_illegal"] = False
        return result

    if robot_role == "wheel":
        below_axle = (
            wheel_center_z is None
            or float(contact_pos[2]) <= float(wheel_center_z) - 0.15 * float(wheel_radius)
        )
        if env_category == "ground":
            # Heightfield terrain has no overhangs/ceilings, and MuJoCo's hfield
            # prism-seam contact solver occasionally flips bottom-tread normals
            # to nz ~ -1 or horizontal at micro-penetrations (-1e-6..-2e-5 m).
            # Flag wheel-ground contacts only on real above-tread side wall hits.
            if (
                not below_axle
                and abs(nz) < 0.70
                and float(contact_dist) < -0.001
            ):
                result["category"] = "illegal_wheel_side_contact"
                result["is_illegal"] = True
                return result
            result["category"] = "legal_wheel_ground"
            result["is_illegal"] = False
            return result
        if abs(nz) >= support_normal_min_z and below_axle:
            if env_category == "audited_roof_box":
                # Bottom-tread vertical support on a drivable deck/bridge tile;
                # MuJoCo box-seam contacts can flip the vertical normal sign on a
                # single step, whereas a real ceiling contact occurs above the axle.
                result["category"] = "legal_wheel_drivable_deck"
                result["is_illegal"] = False
                return result
        if nz <= -0.5:
            result["category"] = "illegal_wheel_ceiling"
            result["is_illegal"] = True
            return result
        if env_category in ("dynamic_obstacle", "other_env"):
            result["category"] = "illegal_wheel_obstacle"
            result["is_illegal"] = True
            return result
        result["category"] = "illegal_wheel_side_contact"
        result["is_illegal"] = True
        return result

    # Chassis collision geom: heightfield ground has no ceiling; vertical/seam
    # terrain graze on ramps is legal support, while horizontal wall impact or
    # any contact with audited_roof_*/obstacles is illegal.
    if env_category == "ground":
        if abs(nz) >= support_normal_min_z:
            result["category"] = "legal_chassis_ramp_graze"
            result["is_illegal"] = False
            return result
        result["category"] = "illegal_chassis_wall"
        result["is_illegal"] = True
        return result
    if nz <= -0.5:
        result["category"] = "illegal_chassis_ceiling"
    else:
        result["category"] = "illegal_chassis_obstacle"
    result["is_illegal"] = True
    return result


def contact_is_violation(
    geom1,
    geom2,
    robot_geom_ids,
    wheel_geom_ids,
    ground_geom_ids,
):
    """Return true for robot contacts other than normal terrain support.

    A robot driving on a field whose terrain is a height field (e.g. the RMUC
    highland ramp) legitimately touches the ground with more than just the
    wheels: the chassis or a steer post can graze the ramp surface while
    climbing.  Those contacts are normal terrain support, not a collision
    with an obstacle.  Only contacts between robot geoms and non-terrain
    geoms are violations.
    """
    robot_contact = geom1 in robot_geom_ids or geom2 in robot_geom_ids
    if not robot_contact:
        return False
    robot_ground = (
        geom1 in robot_geom_ids and geom2 in ground_geom_ids
    ) or (
        geom2 in robot_geom_ids and geom1 in ground_geom_ids
    )
    if robot_ground:
        # Wheel-ground is the normal case; chassis-on-terrain happens on
        # ramps and is also terrain support, not a violation.
        return False
    # Robot geom touching another robot geom or a non-terrain geom.
    return True


def chassis_to_wheel_targets(command, previous_angles=None, wheel_positions=None):
    """Convert a chassis velocity command to four wheel targets."""
    previous_angles = previous_angles or {}
    wheel_positions = wheel_positions or WHEEL_POSITIONS
    targets = []

    for name in WHEEL_ORDER:
        x_pos, y_pos = wheel_positions[name]
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


def mode_to_wheel_targets(mode, command, previous_angles=None, wheel_positions=None):
    """Apply WL100 motion-mode rules and return wheel targets."""
    previous_angles = previous_angles or {}

    if mode == MODE_SWERVE:
        effective = ChassisCommand(
            command.linear_x,
            command.linear_y,
            command.angular_z,
        )
        return effective, chassis_to_wheel_targets(effective, previous_angles, wheel_positions)

    if mode == MODE_CRAB:
        effective = ChassisCommand(command.linear_x, command.linear_y, 0.0)
        return effective, chassis_to_wheel_targets(effective, previous_angles, wheel_positions)

    if mode == MODE_SPIN:
        effective = ChassisCommand(0.0, 0.0, command.angular_z)
        return effective, chassis_to_wheel_targets(effective, previous_angles, wheel_positions)

    if mode == MODE_PARK:
        effective = ChassisCommand()
        return effective, park_wheel_targets()

    effective = ChassisCommand()
    zero_targets = [
        WheelTarget(name, previous_angles.get(name, 0.0), 0.0)
        for name in WHEEL_ORDER
    ]
    return effective, zero_targets


def estimate_chassis_command(targets, wheel_positions=None):
    """Estimate chassis motion from four wheel speed and steering targets."""
    if not targets:
        return ChassisCommand()
    wheel_positions = wheel_positions or WHEEL_POSITIONS

    linear_x = 0.0
    linear_y = 0.0
    angular_z = 0.0

    for target in targets:
        x_pos, y_pos = wheel_positions[target.name]
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
