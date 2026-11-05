"""IMU publication must preserve the impulse of every physics step."""
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from ats_mujoco_sim.sim_node import SwerveMujocoSim


def sampler(stride):
    return SimpleNamespace(
        imu_stride=stride, imu_gyro_adr=0, imu_accel_adr=3,
        imu_sample_sum=np.zeros(6), imu_sample_count=0, imu_sample_time_sum=0.0,
        lio_step_index=0, lio_outbox=[], ground_truth_stride=100000,
        data=SimpleNamespace(time=0.0, sensordata=np.zeros(6)),
        _body_pose_locked=lambda: (np.zeros(3), np.eye(3)),
        _body_velocity_locked=lambda: (np.zeros(3), np.zeros(3)))


def collect(state, observed_time, readings, dt):
    state.data.sensordata[:] = readings
    state.data.time = observed_time + dt
    sample = {"time": observed_time, "index": state.lio_step_index}
    SwerveMujocoSim._lio_collect_after_step_locked(state, sample)
    return sample


def test_decimation_keeps_impulses_from_both_physics_steps():
    state = sampler(2)
    samples = []
    for index in range(400):
        # A 400 Hz alternating contact impulse aliases to a constant 2 g
        # under the former 200 Hz point sampling. No noise/filter tuning.
        sample = collect(state, index * .0025,
                         [0, 0, 4.0 if index % 2 == 0 else 0,
                          0, 0, 19.62 if index % 2 == 0 else 0], .0025)
        if "imu" in sample:
            samples.append(sample)
    assert len(samples) == 200
    np.testing.assert_allclose([s["imu"][1][2] for s in samples], 1.0, atol=1e-15)
    np.testing.assert_allclose([s["imu"][0][2] for s in samples], 2.0, atol=1e-15)
    np.testing.assert_allclose([s["imu_time"] for s in samples],
                               np.arange(200) * .005 + .00125, atol=1e-14)
    assert "imu" not in state.lio_outbox[0]


def test_equal_sensor_and_physics_rate_preserves_each_observation_time():
    state = sampler(1)
    for index in range(10):
        time = index * .0025
        sample = collect(state, time, [1, 2, 3, 4, 5, 6], .0025)
        assert sample["imu_time"] == pytest.approx(time, abs=1e-15)
        np.testing.assert_allclose(sample["imu"][0], [1, 2, 3])
        np.testing.assert_allclose(sample["imu"][1], np.array([4, 5, 6]) / 9.81)


def test_real_contact_acceleration_integrates_to_the_physical_velocity_change():
    model = mujoco.MjModel.from_xml_string("""
        <mujoco>
          <option timestep="0.0025" integrator="Euler"/>
          <worldbody>
            <geom type="plane" size="2 2 .1"/>
            <body pos="0 0 .12">
              <freejoint/>
              <geom type="sphere" size=".1" mass="1"/>
              <site name="imu"/>
            </body>
          </worldbody>
          <sensor><gyro site="imu"/><accelerometer site="imu"/></sensor>
        </mujoco>""")
    data = mujoco.MjData(model)
    state = sampler(2)
    initial_velocity = data.qvel[2]
    measured_delta_velocity = 0.0
    for _ in range(400):
        observed_time = float(data.time)
        mujoco.mj_step(model, data)
        sample = collect(state, observed_time, data.sensordata, model.opt.timestep)
        if "imu" in sample:
            measured_delta_velocity += (sample["imu"][1][2] * 9.81 - 9.81) * .005
    assert measured_delta_velocity == pytest.approx(data.qvel[2] - initial_velocity, abs=1e-12)


def test_outbox_publishes_the_observation_time_instead_of_the_last_physics_time():
    state = sampler(2)
    collect(state, 1.0, [0, 0, 0, 0, 0, 9.81], .0025)
    collect(state, 1.0025, [0, 0, 0, 0, 0, 9.81], .0025)
    published = []
    state._publish_imu = lambda stamp, *values: published.append(stamp)
    state._publish_ground_truth = lambda *args: None
    state.clock_pub = SimpleNamespace(publish=lambda msg: None)
    state._lio_report = lambda stamp: None
    SwerveMujocoSim._lio_flush_outbox(state)
    assert published == [pytest.approx(1.00125, abs=1e-14)]
