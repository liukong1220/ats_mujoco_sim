from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from ats_mujoco_sim.kinematics import ChassisCommand
from ats_mujoco_sim.kinematics import WHEEL_ORDER
from ats_mujoco_sim.kinematics import WheelTarget
from ats_mujoco_sim.sim_node import SwerveMujocoSim


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "swerve_chassis.xml"


def _name_id(model, object_type, name):
    object_id = mujoco.mj_name2id(model, object_type, name)
    assert object_id >= 0, name
    return object_id


def _make_simulator_state():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    base_body_id = _name_id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    free_joint_id = next(
        joint_id
        for joint_id in range(
            model.body_jntadr[base_body_id],
            model.body_jntadr[base_body_id] + model.body_jntnum[base_body_id],
        )
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
    )
    steering_names = {
        "lf": "front_left_steer_pos",
        "lr": "rear_left_steer_pos",
        "rf": "front_right_steer_pos",
        "rr": "rear_right_steer_pos",
    }
    wheel_names = {
        "lf": "front_left_wheel_vel",
        "lr": "rear_left_wheel_vel",
        "rf": "front_right_wheel_vel",
        "rr": "rear_right_wheel_vel",
    }
    steer_joint_names = {
        "lf": "front_left_steer_joint",
        "lr": "rear_left_steer_joint",
        "rf": "front_right_steer_joint",
        "rr": "rear_right_steer_joint",
    }
    wheel_joint_names = {
        "lf": "front_left_wheel_joint",
        "lr": "rear_left_wheel_joint",
        "rf": "front_right_wheel_joint",
        "rr": "rear_right_wheel_joint",
    }
    simulator = SimpleNamespace(
        model=model,
        data=data,
        free_qpos_addr=model.jnt_qposadr[free_joint_id],
        free_dof_addr=model.jnt_dofadr[free_joint_id],
        steer_actuator_ids={
            name: _name_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
            for name, actuator in steering_names.items()
        },
        wheel_actuator_ids={
            name: _name_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
            for name, actuator in wheel_names.items()
        },
        steer_joint_ids={
            name: _name_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            for name, joint in steer_joint_names.items()
        },
        wheel_joint_ids={
            name: _name_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            for name, joint in wheel_joint_names.items()
        },
        motion_cmd=ChassisCommand(0.7, -0.4, 0.3),
        effective_motion_cmd=ChassisCommand(0.5, -0.3, 0.2),
        direct_speeds={name: 0.8 for name in WHEEL_ORDER},
        direct_steer_angles={name: 0.2 for name in WHEEL_ORDER},
        last_wheel_speeds={name: 0.9 for name in WHEEL_ORDER},
        last_steer_angles={name: 0.4 for name in WHEEL_ORDER},
        current_targets=[WheelTarget(name, 0.4, 0.9) for name in WHEEL_ORDER],
        last_motion_time=0.0,
        last_speed_time=1.0,
        freeze_motion=False,
        emergency_stop_active=False,
        hard_stop_requested=True,
        contact_violation_count=7,
        max_contact_force=12.5,
    )
    simulator.data.qpos[simulator.free_qpos_addr:simulator.free_qpos_addr + 7] = (
        np.array([2.0, -1.0, 0.4, 1.0, 0.0, 0.0, 0.0])
    )
    simulator.data.qvel[:] = 0.7
    simulator.data.ctrl[:] = 0.9
    mujoco.mj_forward(model, data)
    return simulator


def test_reset_pose_restores_chassis_and_clears_execution_state() -> None:
    simulator = _make_simulator_state()

    SwerveMujocoSim._reset_pose_locked(simulator, (1.25, -0.75, 0.22), 0.6)

    qpos = simulator.data.qpos
    assert np.allclose(
        qpos[simulator.free_qpos_addr:simulator.free_qpos_addr + 3],
        (1.25, -0.75, 0.22),
    )
    assert np.allclose(
        qpos[simulator.free_qpos_addr + 3:simulator.free_qpos_addr + 7],
        (np.cos(0.3), 0.0, 0.0, np.sin(0.3)),
    )
    assert np.allclose(
        simulator.data.qvel[simulator.free_dof_addr:simulator.free_dof_addr + 6],
        0.0,
    )
    assert simulator.motion_cmd == ChassisCommand()
    assert simulator.effective_motion_cmd == ChassisCommand()
    assert all(value == 0.0 for value in simulator.direct_speeds.values())
    assert all(value == 0.0 for value in simulator.last_wheel_speeds.values())
    assert all(target.wheel_speed == 0.0 for target in simulator.current_targets)
    assert not simulator.hard_stop_requested
    assert simulator.contact_violation_count == 7
    assert simulator.max_contact_force == 12.5

    for name in WHEEL_ORDER:
        steer_dof = simulator.model.jnt_dofadr[simulator.steer_joint_ids[name]]
        wheel_dof = simulator.model.jnt_dofadr[simulator.wheel_joint_ids[name]]
        assert simulator.data.qvel[steer_dof] == 0.0
        assert simulator.data.qvel[wheel_dof] == 0.0
        assert simulator.data.ctrl[simulator.steer_actuator_ids[name]] == 0.0
        assert simulator.data.ctrl[simulator.wheel_actuator_ids[name]] == 0.0


def test_reset_pose_rejects_non_finite_pose() -> None:
    simulator = _make_simulator_state()

    with pytest.raises(ValueError, match="three finite values"):
        SwerveMujocoSim._reset_pose_locked(simulator, (np.nan, 0.0, 0.2), 0.0)
    with pytest.raises(ValueError, match="yaw must be finite"):
        SwerveMujocoSim._reset_pose_locked(simulator, (0.0, 0.0, 0.2), np.inf)
