"""The RMUL 2027 3V3 scene: field assets, planning map and tunnel lintel must agree and be drivable."""

from pathlib import Path
import re

import mujoco
import numpy as np
import pytest
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[2]
MODEL_XML = PACKAGE_ROOT / "models/rmul_2027_swerve.xml"
TUNNELS_YAML = PACKAGE_ROOT / "config/tunnels_rmul_2027.yaml"
NAV_YAML = PACKAGE_ROOT / "config/rmul_2027_navigation.yaml"
MAP_YAML = REPO_ROOT / "src/ats_sentry_bringup/map/rmul_2027.yaml"
MAP_PGM = REPO_ROOT / "src/ats_sentry_bringup/map/rmul_2027.pgm"
PCD = REPO_ROOT / "src/ats_sentry_bringup/pcd/rmul_2027.pcd"
WIDTH_M, HEIGHT_M = 12.6, 8.6
VEHICLE_HEIGHT = 0.242
BASE_TO_WHEEL_BOTTOM = 0.180


@pytest.fixture(scope="module")
def scene():
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


@pytest.fixture(scope="module")
def tunnels():
    return yaml.safe_load(TUNNELS_YAML.read_text(encoding="utf-8"))["tunnels"]


def _occupied():
    raw = MAP_PGM.read_bytes()
    header = re.match(rb"P5\s+(\d+)\s+(\d+)\s+(\d+)\s", raw)
    w, h = int(header.group(1)), int(header.group(2))
    pixels = np.frombuffer(raw[header.end():], dtype=np.uint8)[:w * h].reshape(h, w)
    return ((255 - pixels) / 255.0 > 0.65)[::-1]       # row 0 = y 0


def _launch_default(name):
    source = (PACKAGE_ROOT / "launch/rmul_2027_mujoco.launch.py").read_text()
    return float(re.search(rf'DeclareLaunchArgument\("{name}", default_value="([0-9.]+)"\)', source).group(1))


def test_map_yaml_matches_the_field():
    meta = yaml.safe_load(MAP_YAML.read_text())
    assert meta["image"] == "rmul_2027.pgm" and meta["resolution"] == 0.05 and meta["origin"] == [0.0, 0.0, 0.0]
    occupied = _occupied()
    assert occupied.shape == (round(HEIGHT_M / 0.05), round(WIDTH_M / 0.05))


def test_field_mesh_fills_the_map_frame(scene):
    model, data = scene
    geom = model.geom("rmuc_2027_field_visual")
    mesh_id = int(model.geom_dataid[geom.id])
    first = int(model.mesh_vertadr[mesh_id])
    vertices = model.mesh_vert[first:first + int(model.mesh_vertnum[mesh_id])]
    world = data.geom_xpos[geom.id] + vertices @ data.geom_xmat[geom.id].reshape(3, 3).T
    assert world.min(axis=0)[:2] == pytest.approx([0.0, 0.0], abs=0.01)
    assert world.max(axis=0)[:2] == pytest.approx([WIDTH_M, HEIGHT_M], abs=0.01)
    assert world.max(axis=0)[2] == pytest.approx(0.42, abs=0.01)


def test_hfield_spans_the_map_and_walls_are_solid(scene):
    model, data = scene
    assert (int(model.hfield_ncol[0]), int(model.hfield_nrow[0])) == (1260, 860)
    assert list(data.geom_xpos[model.geom("rmuc_2027_field").id][:2]) == pytest.approx([6.3, 4.3])
    rows, cols = 860, 1260
    heights = model.hfield_data[:rows * cols].reshape(rows, cols) * float(model.hfield_size[0][2])
    assert heights[430, 5] == pytest.approx(0.40, abs=0.01)          # west wall
    assert heights[5, 630] == pytest.approx(0.40, abs=0.01)          # south wall
    assert heights[430, 630] < 0.03                                   # open floor


def test_start_zone_is_free_lower_left_and_spawn_has_no_penetration(scene):
    model, _ = scene
    x, y, z = (_launch_default(n) for n in ("start_x", "start_y", "start_z"))
    assert x < WIDTH_M / 2 and y < HEIGHT_M / 2                       # lower-left quadrant
    occupied = _occupied()
    half = int(0.35 / 0.05)
    cell = occupied[int(y / 0.05) - half:int(y / 0.05) + half + 1, int(x / 0.05) - half:int(x / 0.05) + half + 1]
    assert not cell.any()
    data = mujoco.MjData(model)
    data.qpos[:3] = [x, y, z]
    mujoco.mj_forward(model, data)
    assert not [c for c in data.contact if c.dist < -1e-6]


def test_lintel_leaves_the_vehicle_room_to_pass(scene, tunnels):
    model, _ = scene
    assert len(tunnels) == 2
    for tunnel in tunnels:
        assert tunnel["roof_bottom_z"] - VEHICLE_HEIGHT >= 0.05       # official lintel left 0.008 m
        geom = model.geom("roof_" + tunnel["name"])
        bottom = float(geom.pos[2] - geom.size[2])
        assert bottom == pytest.approx(tunnel["roof_bottom_z"], abs=1e-3)
        cx = float(geom.pos[0])
        y = float(geom.pos[1])
        # Slide the vehicle through the lintel slot at floor level: it must never touch anything.
        for x in np.arange(cx - 0.8, cx + 0.8, 0.02):
            data = mujoco.MjData(model)
            data.qpos[:3] = [x, y, 0.2]
            mujoco.mj_forward(model, data)
            assert not [c for c in data.contact if c.dist < -1e-6], (tunnel["name"], x)


def test_tunnel_corridors_are_free_in_the_planning_map(tunnels):
    occupied = _occupied()
    for tunnel in tunnels:
        xs, ys = zip(*tunnel["corridor_polygon"])
        row = int(round(((min(ys) + max(ys)) / 2) / 0.05))
        for col in range(int(round(min(xs) / 0.05)), int(round(max(xs) / 0.05))):
            assert not occupied[row, col], (tunnel["name"], col * 0.05)
        assert max(ys) - min(ys) >= 0.75                               # 0.571 m robot plus margin
        # both walls beside the lintel are occupied
        col = int(round(((min(xs) + max(xs)) / 2) / 0.05))
        assert occupied[int(round(min(ys) / 0.05)) - 2, col] and occupied[int(round(max(ys) / 0.05)) + 1, col]


def test_navigation_override_rects_cover_the_tunnels(tunnels):
    rects = yaml.safe_load(NAV_YAML.read_text())["ats_rog_map_adapter"]["ros__parameters"]["terrain_free_override_rects"]
    boxes = [rects[i:i + 4] for i in range(0, len(rects), 4)]
    for tunnel in tunnels:
        xs, ys = zip(*tunnel["corridor_polygon"])
        assert any(b[0] <= min(xs) and b[2] >= max(xs) and b[1] <= min(ys) + 1e-6 and b[3] >= max(ys) - 1e-6
                   for b in boxes), tunnel["name"]


def test_prior_pcd_matches_the_field_frame():
    data = PCD.read_bytes()
    end = data.index(b"DATA binary\n") + len(b"DATA binary\n")
    header = dict(line.split(" ", 1) for line in data[:end].decode().splitlines() if " " in line)
    count = int(header["POINTS"])
    cloud = np.frombuffer(data[end:], dtype="<f4", count=count * 6).reshape(count, 6)
    assert cloud[:, 0].min() >= -0.01 and cloud[:, 0].max() <= WIDTH_M + 0.01
    assert cloud[:, 1].min() >= -0.01 and cloud[:, 1].max() <= HEIGHT_M + 0.01
    assert cloud[:, 2].min() >= -0.01 and cloud[:, 2].max() <= 0.43
    lintel = cloud[(np.abs(cloud[:, 0] - 3.8) < 0.15) & (np.abs(cloud[:, 1] - 0.84) < 0.3)]
    assert (lintel[:, 2] > 0.3).any() and not ((lintel[:, 2] > 0.1) & (lintel[:, 2] < 0.3)).any()
