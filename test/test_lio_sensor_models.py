"""Pure-python checks for the LIO-mode sensor helpers (no ROS needed)."""

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from ats_mujoco_sim import livox_scan
from ats_mujoco_sim.lidar_mount import chassis_mount
from ats_mujoco_sim.lidar_mount import legacy_gimbal_mount
from ats_mujoco_sim.model_patch import patch_lidar_mount
from ats_mujoco_sim.sensor_noise import ImuNoiseConfig
from ats_mujoco_sim.sensor_noise import ImuNoiseModel
from ats_mujoco_sim.sensor_noise import RangeNoiseConfig
from ats_mujoco_sim.sensor_noise import RangeNoiseModel

MODELS = Path(__file__).resolve().parents[1] / "models"


def test_front_30_deg_matches_report_gravity():
    # Report 2.1.8: pitch +30 deg gives gravity (4.905, 0, -8.496) in the IMU frame.
    pose = chassis_mount(tilt_direction="front", tilt_deg=30.0)
    gx, gy, gz = pose.gravity_in_sensor()
    assert gx == pytest.approx(4.905, abs=1e-3)
    assert gy == pytest.approx(0.0, abs=1e-9)
    assert gz == pytest.approx(-8.4957, abs=1e-3)
    roll, pitch, yaw = pose.rpy()
    assert (roll, yaw) == (pytest.approx(0.0), pytest.approx(0.0))
    assert pitch == pytest.approx(math.radians(30.0))


def test_left_tilt_leans_towards_left_and_keeps_azimuth():
    pose = chassis_mount(tilt_direction="left", tilt_deg=35.0)
    z_axis = pose.rotation_matrix()[:, 2]
    assert z_axis[1] == pytest.approx(math.sin(math.radians(35.0)))
    assert z_axis[0] == pytest.approx(0.0, abs=1e-12)
    roll, pitch, yaw = pose.rpy()
    assert roll == pytest.approx(-math.radians(35.0))
    assert (pitch, yaw) == (pytest.approx(0.0, abs=1e-12), pytest.approx(0.0))


def test_default_mount_is_the_front_edge_midpoint():
    pose = chassis_mount()
    # Chassis is 0.581 m long (x): 0.2905 - 0.05 inset, on the centre line.
    assert pose.position[0] == pytest.approx(0.2405)
    assert pose.position[1] == pytest.approx(0.0)


def test_default_mount_is_inside_the_height_envelope():
    pose = chassis_mount()
    # Optical centre 0.20 m above ground, base_link origin 0.18 m up.
    assert pose.position[2] == pytest.approx(0.02)
    assert 0.20 < 0.242


def test_rpy_quaternion_round_trip():
    for direction in ("front", "left"):
        pose = chassis_mount(tilt_direction=direction, tilt_deg=35.0)
        w, x, y, z = pose.quat_wxyz()
        assert w * w + x * x + y * y + z * z == pytest.approx(1.0)


def test_patched_model_places_lidar_and_imu_consistently():
    """MuJoCo pose, static TF source and Point-LIO gravity share one mount."""
    pose = chassis_mount(tilt_direction="front", tilt_deg=35.0)
    xml = patch_lidar_mount((MODELS / "rmuc_2027_swerve.xml").read_text("utf-8"), pose)
    patched = MODELS / ".test_lio_patched_model.xml"
    patched.write_text(xml, "utf-8")
    try:
        model = mujoco.MjModel.from_xml_path(str(patched))
    finally:
        patched.unlink()
    data = mujoco.MjData(model)
    data.qpos[:7] = [0.0, 0.0, 0.18, 1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    site = data.site("lidar_site")
    rotation = np.array(site.xmat).reshape(3, 3)
    np.testing.assert_allclose(rotation, pose.rotation_matrix(), atol=1e-9)
    # Default mount: front (+x) edge midpoint pulled in 0.05 m, optical centre 0.20 m high.
    np.testing.assert_allclose(site.xpos, [0.2405, 0.0, 0.20], atol=1e-9)

    gravity = rotation.T @ np.array([0.0, 0.0, -9.81])
    np.testing.assert_allclose(gravity, pose.gravity_in_sensor(), atol=1e-9)

    imu = data.site("mid360_imu_site")
    expected_imu = np.array(pose.imu_position_in_body()) + np.array([0.0, 0.0, 0.18])
    np.testing.assert_allclose(imu.xpos, expected_imu, atol=1e-9)


def test_legacy_preset_equals_the_versioned_model_pose():
    pose = legacy_gimbal_mount()
    model = mujoco.MjModel.from_xml_path(str(MODELS / "rmuc_2027_swerve.xml"))
    body_id = model.body("lidar_link").id
    np.testing.assert_allclose(model.body_pos[body_id], pose.position, atol=1e-9)
    quat = np.array(pose.quat_wxyz())
    got = model.body_quat[body_id]
    # q and -q are the same rotation.
    assert min(np.abs(got - quat).max(), np.abs(got + quat).max()) < 1e-6


def test_patch_requires_a_lidar_body():
    with pytest.raises(ValueError):
        patch_lidar_mount("<mujoco/>", chassis_mount())


def test_frame_rays_match_real_point_rate():
    # 20 Hz at 200 kpts/s class: 12000 rays per 50 ms frame.
    assert livox_scan.rays_per_frame(0.05) == 12000
    assert livox_scan.rays_per_frame(0.1) == 24000
    assert livox_scan.segment_count(0.05) == 10


def test_segments_cover_every_ray_once_and_offsets_are_monotonic():
    count = 12000
    bounds = livox_scan.segment_bounds(count, 10)
    assert bounds[0] == 0 and bounds[-1] == count
    assert np.all(np.diff(bounds) == 1200)
    offsets = livox_scan.offset_time_ns(np.arange(count), count, 0.05)
    assert offsets[0] == 0
    assert np.all(np.diff(offsets.astype(np.int64)) >= 0)
    assert offsets[-1] < 50_000_000


def test_take_rays_wraps_the_cyclic_pattern():
    pattern = np.arange(10, dtype=np.float32).reshape(5, 2)
    rays, cursor = livox_scan.take_rays(pattern, 3, 4)
    np.testing.assert_array_equal(rays[:, 0], [6, 8, 0, 2])
    assert cursor == 2


def test_nearest_state_and_midtime():
    times = [0.0, 0.0025, 0.005, 0.0075, 0.010]
    assert livox_scan.nearest_state_index(times, 0.0049) == 2
    assert livox_scan.nearest_state_index(times, -1.0) == 0
    assert livox_scan.nearest_state_index(times, 9.0) == 4
    bounds = livox_scan.segment_bounds(12000, 10)
    assert livox_scan.segment_midtime(1.0, 0.05, bounds, 0) == pytest.approx(1.0025)


def test_pack_points_layout():
    xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    payload = livox_scan.pack_points(
        xyz, np.array([7, 9], dtype=np.uint32), np.array([1, 3], dtype=np.uint8)
    )
    assert len(payload) == 2 * livox_scan.POINT_STEP
    rows = np.frombuffer(payload, dtype=livox_scan.LIVOX_POINT_DTYPE)
    assert rows["offset_time"].tolist() == [7, 9]
    assert rows["line"].tolist() == [1, 3]
    assert rows["tag"].tolist() == [0, 0]
    assert rows["z"].tolist() == [3.0, 6.0]


def test_noise_is_off_by_default_and_exact_passthrough():
    imu = ImuNoiseModel(ImuNoiseConfig())
    assert not imu.active
    gyro, accel = imu.apply([0.1, 0.2, 0.3], [0.0, 0.0, 1.0], 0.005)
    np.testing.assert_array_equal(gyro, [0.1, 0.2, 0.3])
    np.testing.assert_array_equal(accel, [0.0, 0.0, 1.0])
    rng = RangeNoiseModel(RangeNoiseConfig())
    points = np.ones((5, 3), dtype=np.float32)
    assert not rng.active
    assert rng.apply(points) is points


def test_noise_is_seeded_and_saturation_clips():
    config = ImuNoiseConfig(enable_noise=True, enable_saturation=True, seed=7)
    first = ImuNoiseModel(config)
    second = ImuNoiseModel(config)
    a = [first.apply([0, 0, 0], [0, 0, 1], 0.005) for _ in range(20)]
    b = [second.apply([0, 0, 0], [0, 0, 1], 0.005) for _ in range(20)]
    for (g1, a1), (g2, a2) in zip(a, b):
        np.testing.assert_array_equal(g1, g2)
        np.testing.assert_array_equal(a1, a2)
    clipped_gyro, clipped_acc = ImuNoiseModel(
        ImuNoiseConfig(enable_saturation=True)
    ).apply([100.0, -100.0, 0.0], [10.0, -10.0, 0.0], 0.005)
    assert clipped_gyro.tolist() == [35.0, -35.0, 0.0]
    assert clipped_acc.tolist() == [6.0, -6.0, 0.0]


def test_range_noise_scales_along_the_ray():
    model = RangeNoiseModel(RangeNoiseConfig(enable=True, sigma_m=0.02, seed=3))
    points = np.array([[10.0, 0.0, 0.0], [0.0, 5.0, 0.0]], dtype=np.float32)
    noisy = model.apply(points)
    # Direction is preserved, only the range changes.
    assert noisy[0, 1] == 0.0 and noisy[0, 2] == 0.0
    assert noisy[1, 0] == 0.0 and noisy[1, 2] == 0.0
    assert abs(noisy[0, 0] - 10.0) < 0.2
