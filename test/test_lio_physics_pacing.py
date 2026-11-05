"""LIO wall scheduling must not accumulate wakeup error or catch up without bound."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from ats_mujoco_sim import sim_node


def run_loop(monkeypatch, lio, slow_step=None):
    clock = SimpleNamespace(now=0.0)
    starts = []
    timestep, wakeup_error = .0025, .0003

    def sleep(duration):
        assert duration >= 0.0
        clock.now += duration + wakeup_error

    def step():
        starts.append(clock.now)
        clock.now += .020 if len(starts) == slow_step else .0004

    monkeypatch.setattr(sim_node.time, "perf_counter", lambda: clock.now)
    monkeypatch.setattr(sim_node.time, "sleep", sleep)
    monkeypatch.setattr(sim_node.rclpy, "ok", lambda: True)
    node = SimpleNamespace(
        sim_lock=nullcontext(), viewer=None, viewer_data=None, lio_mode=lio,
        stop_event=SimpleNamespace(is_set=lambda: len(starts) >= 40, wait=sleep),
        model=SimpleNamespace(opt=SimpleNamespace(timestep=timestep)),
        _step_simulation_locked=step, _lio_flush_outbox=lambda: None)
    sim_node.SwerveMujocoSim._simulation_loop(node)
    return starts, timestep, wakeup_error


def test_lio_wakeup_error_is_bounded_instead_of_accumulating(monkeypatch):
    starts, timestep, error = run_loop(monkeypatch, True)
    assert len(starts) == 40
    assert starts[-1] >= 39 * timestep
    assert starts[-1] <= 39 * timestep + error + 1e-12


def test_lio_overload_restarts_the_deadline_without_a_burst_of_catchup_steps(monkeypatch):
    starts, timestep, error = run_loop(monkeypatch, True, slow_step=8)
    assert starts[8] - starts[7] >= .020 - 1e-12
    assert all(b - a >= timestep - error for a, b in zip(starts[8:], starts[9:]))


def test_truth_mode_preserves_its_existing_relative_schedule(monkeypatch):
    starts, timestep, error = run_loop(monkeypatch, False)
    assert starts[-1] == pytest.approx(39 * (timestep + error), abs=1e-12)
