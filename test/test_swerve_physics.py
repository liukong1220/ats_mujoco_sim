from math import hypot, pi
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from ats_mujoco_sim.kinematics import ACTUATOR_REDUNDANCY
from ats_mujoco_sim.kinematics import ChassisCommand
from ats_mujoco_sim.kinematics import MAX_STEER_RATE_RADPS
from ats_mujoco_sim.kinematics import MAX_WHEEL_SPEED_MPS
from ats_mujoco_sim.kinematics import MOTOR_MAX_RPM
from ats_mujoco_sim.kinematics import STEER_MAX_RPM
from ats_mujoco_sim.kinematics import WHEEL_OFFSET_X_M
from ats_mujoco_sim.kinematics import WHEEL_OFFSET_Y_M
from ats_mujoco_sim.kinematics import WHEEL_RADIUS_M
from ats_mujoco_sim.kinematics import chassis_to_wheel_targets
from ats_mujoco_sim.kinematics import contact_is_violation
from ats_mujoco_sim.kinematics import rate_limit_angle
from ats_mujoco_sim.sim_node import SwerveMujocoSim


MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_NAMES = ("swerve_chassis.xml", "swerve.xml", "rmuc_2026_swerve.xml")


def _name_id(model, object_type, name):
    object_id = mujoco.mj_name2id(model, object_type, name)
    assert object_id >= 0, name
    return object_id


def test_physical_limits_match_motor_and_final_rolling_radius() -> None:
    raw_wheel_speed = 2.0 * pi * WHEEL_RADIUS_M * MOTOR_MAX_RPM / 60.0
    raw_steer_rate = 2.0 * pi * STEER_MAX_RPM / 60.0

    assert np.isclose(raw_wheel_speed, 2.0027653167)
    assert np.isclose(MAX_WHEEL_SPEED_MPS, raw_wheel_speed / ACTUATOR_REDUNDANCY)
    assert np.isclose(raw_steer_rate, 12.5663706144)
    assert np.isclose(MAX_STEER_RATE_RADPS, raw_steer_rate / ACTUATOR_REDUNDANCY)


def test_kinematics_uses_center_to_wheel_offsets_and_true_lateral_motion() -> None:
    lateral = chassis_to_wheel_targets(ChassisCommand(linear_y=1.0))
    assert all(np.isclose(abs(target.steer_angle), pi / 2.0) for target in lateral)
    assert all(np.isclose(abs(target.wheel_speed), 1.0) for target in lateral)

    spin = chassis_to_wheel_targets(ChassisCommand(angular_z=1.0))
    expected_speed = hypot(WHEEL_OFFSET_X_M, WHEEL_OFFSET_Y_M)
    assert all(np.isclose(abs(target.wheel_speed), expected_speed) for target in spin)


def test_steer_target_rate_limiter_wraps_and_saturates() -> None:
    limited, saturated = rate_limit_angle(0.0, pi / 2.0, 0.1)
    assert saturated
    assert np.isclose(limited, 0.1)

    wrapped, wrapped_saturated = rate_limit_angle(pi - 0.02, -pi + 0.02, 0.1)
    assert not wrapped_saturated
    assert np.isclose(wrapped, -pi + 0.02)


def test_all_active_models_compile_to_25kg_and_100mm_total_cog() -> None:
    for model_name in MODEL_NAMES:
        model = mujoco.MjModel.from_xml_path(str(MODEL_DIR / model_name))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        base_id = _name_id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        assert np.isclose(np.sum(model.body_mass), 25.0)
        assert np.isclose(model.body_mass[base_id], 23.0)
        assert np.isclose(data.subtree_com[base_id, 2], 0.10, atol=1e-7)

        for prefix, expected_xy in (
            ("front_left", (0.270, 0.270)),
            ("front_right", (0.270, -0.270)),
            ("rear_left", (-0.270, 0.270)),
            ("rear_right", (-0.270, -0.270)),
        ):
            steer_body = _name_id(
                model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_steer_link"
            )
            wheel_geom = _name_id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}_wheel")
            wheel_actuator = _name_id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}_wheel_vel"
            )
            assert np.allclose(model.body_pos[steer_body, :2], expected_xy)
            assert np.isclose(model.geom_size[wheel_geom, 0], WHEEL_RADIUS_M)
            assert np.allclose(model.geom_friction[wheel_geom], (0.8, 0.02, 0.005))
            assert np.allclose(
                model.actuator_ctrlrange[wheel_actuator],
                (
                    -MAX_WHEEL_SPEED_MPS / WHEEL_RADIUS_M,
                    MAX_WHEEL_SPEED_MPS / WHEEL_RADIUS_M,
                ),
                atol=1e-5,
            )
            assert np.allclose(model.actuator_forcerange[wheel_actuator], (-2.1, 2.1))


def test_odometry_twist_is_expressed_in_child_body_frame() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_DIR / "swerve_chassis.xml"))
    data = mujoco.MjData(model)
    base_id = _name_id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    free_joint = int(model.body_jntadr[base_id])
    qpos_address = int(model.jnt_qposadr[free_joint])
    dof_address = int(model.jnt_dofadr[free_joint])
    quaternion_slice = slice(qpos_address + 3, qpos_address + 7)
    linear_velocity_slice = slice(dof_address, dof_address + 3)

    data.qpos[quaternion_slice] = (
        np.cos(pi / 4.0),
        0.0,
        0.0,
        np.sin(pi / 4.0),
    )
    data.qvel[linear_velocity_slice] = (1.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)

    simulator = SimpleNamespace(model=model, data=data, base_body_id=base_id)
    linear, angular = SwerveMujocoSim._body_velocity_locked(simulator)
    assert np.allclose(linear, (0.0, -1.0, 0.0), atol=1e-12)
    assert np.allclose(angular, (0.0, 0.0, 0.0), atol=1e-12)


def test_contact_evaluator_detects_base_ground_penetration() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_DIR / "swerve_chassis.xml"))
    data = mujoco.MjData(model)
    base_id = _name_id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    free_joint = int(model.body_jntadr[base_id])
    qpos_address = int(model.jnt_qposadr[free_joint])
    data.qpos[qpos_address + 2] = 0.05
    mujoco.mj_forward(model, data)

    robot_bodies = set()
    for body_id in range(model.nbody):
        current = body_id
        while current > 0 and current != base_id:
            current = int(model.body_parentid[current])
        if current == base_id:
            robot_bodies.add(body_id)
    robot_geoms = {
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in robot_bodies
    }
    wheel_geoms = {
        _name_id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}_wheel")
        for prefix in ("front_left", "rear_left", "front_right", "rear_right")
    }
    ground_geoms = {_name_id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")}
    violations = [
        contact
        for contact in data.contact[: data.ncon]
        if contact_is_violation(
            int(contact.geom1),
            int(contact.geom2),
            robot_geoms,
            wheel_geoms,
            ground_geoms,
        )
    ]
    assert violations
