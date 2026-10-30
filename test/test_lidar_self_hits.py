"""Mid360 rays that hit the robot itself beyond Point-LIO's 0.3 m blind ball.

sim_node excludes only base_link's own geoms from the LiDAR rays; the steer posts and wheels are
child bodies and stay visible.  Hits inside the blind ball are dropped by Point-LIO; hits outside it
become terrain points around the robot.  With the legacy gimbal pose of rmuc_2025_swerve.xml about
33 rays per frame landed on the rear-left steer post and wheel (0.37-0.49 m), and terrain_analysis_ext
turned them into 0.2 m hard cells on the south ramp of the 2025 red-box route (report 5.1.8 B).
"""
import importlib.util
from pathlib import Path

import mujoco
import numpy as np
import pytest

from ats_mujoco_sim.lidar_mount import chassis_mount
from ats_mujoco_sim.lidar_mount import legacy_gimbal_mount
from ats_mujoco_sim.model_patch import patch_lidar_mount
from mujoco_lidar.multiray_compat import multi_ray

PACKAGE = Path(__file__).resolve().parents[1]
MODELS = PACKAGE / "models"
BLIND_M = 0.3            # node_params.yaml point_lio preprocess.blind
RAYS = 60000             # five 12000-ray LIO frames of the Mid360 pattern


def _launch_lio_arguments():
    spec = importlib.util.spec_from_file_location(
        "rmuc_2025_mujoco_launch", PACKAGE / "launch" / "rmuc_2025_mujoco.launch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LIO_ARGUMENTS_2025


def _mid360_rays():
    scan_gen = pytest.importorskip("mujoco_lidar.scan_gen")
    angles = np.asarray(scan_gen.LivoxGenerator("mid360").ray_angles, dtype=np.float64)
    return angles[:RAYS, 0], angles[:RAYS, 1]


def _self_hits_beyond_blind(model_file, mount, tmp_path):
    xml = (MODELS / model_file).read_text()
    # Same directory as the model so the relative mesh/hfield paths resolve.
    patched = MODELS / f".test_self_hits_{tmp_path.name}.xml"
    patched.write_text(patch_lidar_mount(xml, mount))
    try:
        model = mujoco.MjModel.from_xml_path(str(patched))
    finally:
        patched.unlink()
    data = mujoco.MjData(model)
    data.qpos[:3] = [-0.18, 0.06, 0.381]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    base = model.body("base_link").id

    def on_robot(body):
        while body != 0:
            if body == base:
                return True
            body = model.body_parentid[body]
        return False

    theta, phi = _mid360_rays()
    site = data.site("lidar_raycast_site")
    vec = np.stack([np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), np.sin(phi)], -1)
    vec = vec @ site.xmat.reshape(3, 3).T
    groups = np.ones(mujoco.mjNGROUP, dtype=np.uint8)
    groups[3] = 0  # sim_node's LiDAR geomgroup
    geomid = np.full(len(vec), -1, dtype=np.int32)
    dist = np.full(len(vec), -1.0)
    multi_ray(model, data, site.xpos.astype(np.float64).reshape(3, 1), np.ascontiguousarray(vec.flatten()),
              groups, 1, base, geomid, dist, len(vec), 40.0)
    own = np.array([g >= 0 and on_robot(model.geom_bodyid[g]) for g in geomid])
    return int(np.count_nonzero(own & (dist > BLIND_M)))


def _hits_per_frame(count):
    return count / (RAYS / 12000)


def test_rmuc_2025_launch_mounts_the_lidar_on_the_chassis():
    assert _launch_lio_arguments()["lidar_mount_preset"] == "chassis"


def test_chassis_mount_puts_at_most_one_self_hit_per_frame_past_the_blind_ball(tmp_path):
    mount = chassis_mount(x=0.2405, y=0.0, height_above_ground=0.20, tilt_direction="front",
                          tilt_deg=30.0, body_height=0.18)
    assert _hits_per_frame(_self_hits_beyond_blind("rmuc_2025_swerve.xml", mount, tmp_path)) <= 1.0


def test_legacy_gimbal_mount_sees_its_own_rear_wheel_past_the_blind_ball(tmp_path):
    # Negative control: the pose the 2025 scene used before 2026-10-31.
    count = _self_hits_beyond_blind("rmuc_2025_swerve.xml", legacy_gimbal_mount(), tmp_path)
    assert _hits_per_frame(count) > 10.0
