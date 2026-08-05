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
