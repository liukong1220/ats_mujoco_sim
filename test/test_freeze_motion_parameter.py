"""Runtime chassis-freeze and LiDAR-occlusion parameter contract.

These tests import only ``ats_mujoco_sim.runtime_fault_state``, which carries no
ROS message imports, so they collect and run against a plain source checkout as
well as inside a sourced workspace.  Nothing is skipped and no assertion is
relaxed to accommodate the ``carstatemsgs`` / ``manda_can_control`` message
packages that ``sim_node`` itself needs.
"""

import threading
from types import SimpleNamespace

from ats_mujoco_sim.runtime_fault_state import apply_runtime_fault_parameters


def make_simulator_state(**overrides):
    state = SimpleNamespace(
        freeze_motion=False,
        lidar_occlusion_enabled=False,
        lidar_occlusion_event=None,
        sim_lock=threading.Lock(),
    )
    for name, value in overrides.items():
        setattr(state, name, value)
    return state


def test_runtime_freeze_parameter_uses_single_formatted_logger_message() -> None:
    state = make_simulator_state()

    for value, execution_state in ((True, "held"), (False, "enabled")):
        outcome = apply_runtime_fault_parameters(
            state, [SimpleNamespace(name="freeze_motion", value=value)]
        )

        assert outcome.successful
        assert state.freeze_motion is value
        assert outcome.messages == [
            f"freeze_motion={value}: chassis execution {execution_state} "
            "while sensors remain active"
        ]


def test_runtime_lidar_occlusion_keeps_the_worker_control_event_in_sync() -> None:
    occlusion_event = threading.Event()
    state = make_simulator_state(lidar_occlusion_event=occlusion_event)

    for value, sensor_state in (
        (True, "empty returns enabled"),
        (False, "raycast enabled"),
    ):
        outcome = apply_runtime_fault_parameters(
            state, [SimpleNamespace(name="lidar_occlusion_enabled", value=value)]
        )

        assert outcome.successful
        assert state.lidar_occlusion_enabled is value
        assert occlusion_event.is_set() is value
        assert outcome.messages == [
            f"lidar_occlusion_enabled={value}: {sensor_state}; "
            "LiDAR headers and publication cadence remain active"
        ]


def test_runtime_fault_parameters_default_to_off() -> None:
    state = make_simulator_state(lidar_occlusion_event=threading.Event())

    assert state.freeze_motion is False
    assert state.lidar_occlusion_enabled is False
    assert state.lidar_occlusion_event.is_set() is False


def test_non_boolean_lidar_occlusion_request_is_rejected_without_applying() -> None:
    occlusion_event = threading.Event()
    state = make_simulator_state(lidar_occlusion_event=occlusion_event)

    for value in (1, "true", 0.0, None):
        outcome = apply_runtime_fault_parameters(
            state, [SimpleNamespace(name="lidar_occlusion_enabled", value=value)]
        )

        assert not outcome.successful
        assert outcome.reason == "lidar_occlusion_enabled must be a boolean"
        assert outcome.messages == []
        assert state.lidar_occlusion_enabled is False
        assert occlusion_event.is_set() is False


def test_non_boolean_freeze_motion_request_is_rejected_without_applying() -> None:
    state = make_simulator_state()

    outcome = apply_runtime_fault_parameters(
        state, [SimpleNamespace(name="freeze_motion", value="yes")]
    )

    assert not outcome.successful
    assert outcome.reason == "freeze_motion must be a boolean"
    assert state.freeze_motion is False


def test_mixed_request_is_rejected_without_partially_applying_a_fault() -> None:
    occlusion_event = threading.Event()
    state = make_simulator_state(lidar_occlusion_event=occlusion_event)

    outcome = apply_runtime_fault_parameters(
        state,
        [
            SimpleNamespace(name="freeze_motion", value=True),
            SimpleNamespace(name="lidar_occlusion_enabled", value="true"),
        ],
    )

    assert not outcome.successful
    assert outcome.reason == "lidar_occlusion_enabled must be a boolean"
    assert outcome.messages == []
    assert state.freeze_motion is False
    assert state.lidar_occlusion_enabled is False
    assert occlusion_event.is_set() is False


def test_unrelated_parameters_leave_runtime_fault_state_untouched() -> None:
    occlusion_event = threading.Event()
    state = make_simulator_state(lidar_occlusion_event=occlusion_event)

    outcome = apply_runtime_fault_parameters(
        state,
        [
            SimpleNamespace(name="sim_rate_hz", value=300.0),
            SimpleNamespace(name="publish_map_to_odom_tf", value=False),
        ],
    )

    assert outcome.successful
    assert outcome.messages == []
    assert state.freeze_motion is False
    assert state.lidar_occlusion_enabled is False
    assert occlusion_event.is_set() is False


def test_lidar_occlusion_without_a_worker_event_still_records_the_transition() -> None:
    state = make_simulator_state(lidar_occlusion_event=None)

    outcome = apply_runtime_fault_parameters(
        state, [SimpleNamespace(name="lidar_occlusion_enabled", value=True)]
    )

    assert outcome.successful
    assert state.lidar_occlusion_enabled is True
    assert len(outcome.messages) == 1
