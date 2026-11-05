"""Physics must wait for a full LIO buffer without discarding sensor states."""
import queue
import threading
from types import SimpleNamespace

from ats_mujoco_sim.sim_node import SwerveMujocoSim


def test_two_state_buffer_applies_backpressure_and_preserves_every_state_in_order():
    state = SimpleNamespace(lidar_state_queue=queue.Queue(maxsize=2),
                            lidar_process=SimpleNamespace(exitcode=None),
                            stop_event=threading.Event(), lio_backpressure_s=0.0)
    filled = threading.Event()
    completed = threading.Event()
    states = [(i * .0025, [i], [i + 1]) for i in range(20)]

    def produce():
        for i, sample in enumerate(states):
            SwerveMujocoSim._lio_push_lidar_state(state, sample)
            if i == 1:
                filled.set()
        completed.set()

    producer = threading.Thread(target=produce, daemon=True)
    producer.start()
    try:
        assert filled.wait(1.0)
        assert not completed.wait(.08), "physics advanced past the full sensor buffer"
        observed = [state.lidar_state_queue.get(timeout=1.0) for _ in states]
        assert completed.wait(1.0)
        assert observed == states
        assert state.lio_backpressure_s > 0.0
    finally:
        state.stop_event.set()
        producer.join(timeout=1.0)
