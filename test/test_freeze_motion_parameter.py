import threading
from types import SimpleNamespace

from ats_mujoco_sim.sim_node import SwerveMujocoSim


class RecordingLogger:
    def __init__(self) -> None:
        self.messages = []

    def warn(self, message: str) -> None:
        self.messages.append(message)


def test_runtime_freeze_parameter_uses_single_formatted_logger_message() -> None:
    logger = RecordingLogger()
    simulator = SimpleNamespace(
        freeze_motion=False,
        sim_lock=threading.Lock(),
        get_logger=lambda: logger,
    )

    for value, execution_state in ((True, "held"), (False, "enabled")):
        result = SwerveMujocoSim._on_set_parameters(
            simulator, [SimpleNamespace(name="freeze_motion", value=value)]
        )

        assert result.successful
        assert simulator.freeze_motion is value
        assert logger.messages[-1] == (
            f"freeze_motion={value}: chassis execution {execution_state} "
            "while sensors remain active"
        )


def test_runtime_lidar_occlusion_keeps_the_worker_control_event_in_sync() -> None:
    logger = RecordingLogger()
    occlusion_event = threading.Event()
    simulator = SimpleNamespace(
        lidar_occlusion_enabled=False,
        lidar_occlusion_event=occlusion_event,
        sim_lock=threading.Lock(),
        get_logger=lambda: logger,
    )

    for value, state in ((True, "empty returns enabled"), (False, "raycast enabled")):
        result = SwerveMujocoSim._on_set_parameters(
            simulator,
            [SimpleNamespace(name="lidar_occlusion_enabled", value=value)],
        )

        assert result.successful
        assert simulator.lidar_occlusion_enabled is value
        assert occlusion_event.is_set() is value
        assert logger.messages[-1] == (
            f"lidar_occlusion_enabled={value}: {state}; "
            "LiDAR headers and publication cadence remain active"
        )
