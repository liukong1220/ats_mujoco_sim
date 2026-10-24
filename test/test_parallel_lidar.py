"""Threaded ray tracing must be a pure speed-up: same rays, same state, same result."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from mujoco_lidar import MjLidarWrapper

from ats_mujoco_sim.parallel_lidar import MAX_AUTO_THREADS
from ats_mujoco_sim.parallel_lidar import ParallelRayTracer
from ats_mujoco_sim.parallel_lidar import resolve_thread_count
from ats_mujoco_sim.parallel_lidar import split_bounds

MODEL_XML = Path(__file__).resolve().parents[1] / "models/rmuc_2027_swerve.xml"
LIDAR_SITE = "lidar_raycast_site"


def test_auto_thread_count_is_a_quarter_of_the_cpus_capped_at_eight():
    assert resolve_thread_count(0, cpu_count=32) == MAX_AUTO_THREADS
    assert resolve_thread_count(0, cpu_count=16) == 4
    assert resolve_thread_count(0, cpu_count=2) == 1
    assert resolve_thread_count(0, cpu_count=1) == 1


def test_explicit_thread_count_wins_and_negative_is_rejected():
    assert resolve_thread_count(3, cpu_count=32) == 3
    assert resolve_thread_count(16, cpu_count=4) == 16
    with pytest.raises(ValueError):
        resolve_thread_count(-1)


@pytest.mark.parametrize("count,parts", [(10, 3), (1200, 8), (5, 8), (1, 4), (7, 1)])
def test_split_bounds_cover_every_ray_exactly_once(count, parts):
    bounds = split_bounds(count, parts)
    assert bounds[0][0] == 0 and bounds[-1][1] == count
    assert all(first < last for first, last in bounds)
    assert all(a[1] == b[0] for a, b in zip(bounds, bounds[1:]))
    assert len(bounds) <= parts


def test_split_bounds_of_nothing_is_empty():
    assert split_bounds(0, 4) == []


@pytest.fixture(scope="module")
def scene():
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    groups = np.ones(mujoco.mjNGROUP, dtype=np.ubyte)
    groups[3] = 0
    exclude = model.body("base_link").id

    def make_lidar():
        return MjLidarWrapper(
            model,
            site_name=LIDAR_SITE,
            backend="cpu",
            cutoff_dist=40.0,
            args={"bodyexclude": exclude, "geomgroup": groups},
        )

    return model, make_lidar


def _rays(count, seed):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-np.pi, np.pi, count).astype(np.float32)
    phi = np.radians(rng.uniform(-7.0, 52.0, count)).astype(np.float32)
    return theta, phi


def _state(model, x, y, z):
    data = mujoco.MjData(model)
    qpos = np.array(data.qpos, copy=True)
    qpos[:3] = (x, y, z)
    return qpos, np.zeros(model.nv)


@pytest.mark.parametrize("threads", [2, 4, 8])
def test_threaded_trace_is_identical_to_the_serial_trace(scene, threads):
    model, make_lidar = scene
    theta, phi = _rays(3000, seed=1)
    qpos, qvel = _state(model, 12.9, 2.043, 0.55)

    serial = ParallelRayTracer(model, make_lidar, 1)
    parallel = ParallelRayTracer(model, make_lidar, threads)
    try:
        for tracer in (serial, parallel):
            tracer.set_state(1.0, qpos, qvel)
        serial_distances, serial_points = serial.trace(theta, phi)
        distances, points = parallel.trace(theta, phi)
    finally:
        parallel.close()
        serial.close()

    assert int(np.count_nonzero(serial_distances > 0.3)) > 500      # not a vacuous comparison
    np.testing.assert_array_equal(distances, serial_distances)
    np.testing.assert_array_equal(points, serial_points)


def test_every_worker_follows_a_new_state(scene):
    model, make_lidar = scene
    theta, phi = _rays(2000, seed=2)
    states = [_state(model, 12.9, 2.043, 0.55), _state(model, 5.0, 8.0, 0.5)]

    serial = ParallelRayTracer(model, make_lidar, 1)
    parallel = ParallelRayTracer(model, make_lidar, 4)
    results = []
    try:
        for state in states:
            serial.set_state(0.0, *state)
            parallel.set_state(0.0, *state)
            expected, _ = serial.trace(theta, phi)
            actual, _ = parallel.trace(theta, phi)
            np.testing.assert_array_equal(actual, expected)
            results.append(actual)
    finally:
        parallel.close()
        serial.close()
    # The two poses see different surroundings, so a worker stuck on the old state would show here.
    assert not np.array_equal(results[0], results[1])
