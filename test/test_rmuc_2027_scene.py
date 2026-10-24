"""The RMUC 2027 MuJoCo scene must be built from the 2027 field, not a 2025 asset."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MODEL_XML = PACKAGE_ROOT / "models/rmuc_2027_swerve.xml"
TUNNELS_YAML = PACKAGE_ROOT / "config/tunnels_2027.yaml"
MAP_WIDTH_M, MAP_HEIGHT_M = 28.0, 16.0   # rmuc_2027.yaml: 560 x 320 cells at 0.05 m
FIELD_TOP_M = 3.7987                      # RMUC2027_V1.0.0.STL, highest point


@pytest.fixture(scope="module")
def scene():
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _hfield_heights(model):
    rows, cols = int(model.hfield_nrow[0]), int(model.hfield_ncol[0])
    start = int(model.hfield_adr[0])
    return model.hfield_data[start:start + rows * cols].reshape(rows, cols) * float(
        model.hfield_size[0][2]
    )


def _hfield_height(model, x, y):
    """Nearest-sample hfield height at map point (x, y); the geom is centred on the map."""
    size_x, size_y = (float(v) for v in model.hfield_size[0][:2])
    rows, cols = int(model.hfield_nrow[0]), int(model.hfield_ncol[0])
    column = round((x - MAP_WIDTH_M / 2 + size_x) / (2 * size_x) * (cols - 1))
    row = round((y - MAP_HEIGHT_M / 2 + size_y) / (2 * size_y) * (rows - 1))
    return _hfield_heights(model)[row, column]


def _field_top(model, data, x, y):
    """Height of the LiDAR-visible field mesh (group 2) under map point (x, y)."""
    groups = np.zeros(mujoco.mjNGROUP, dtype=np.uint8)
    groups[2] = 1
    geom_id = np.zeros(1, dtype=np.int32)
    distance = mujoco.mj_ray(
        model, data, np.array([x, y, 6.0]), np.array([0.0, 0.0, -1.0]), groups, 1, -1, geom_id,
    )
    return 6.0 - distance if distance >= 0.0 else float("nan")


def test_scene_does_not_reference_the_2025_field():
    text = MODEL_XML.read_text(encoding="utf-8")
    assert "rmuc_2025" not in text
    assert 'file="meshes/rmuc_2027.obj"' in text
    assert 'file="rmuc_2027_height.png"' in text


def test_field_mesh_fills_the_2027_map_frame(scene):
    model, data = scene
    geom = model.geom("rmuc_2027_field_visual")
    mesh_id = int(model.geom_dataid[geom.id])
    first = int(model.mesh_vertadr[mesh_id])
    vertices = model.mesh_vert[first:first + int(model.mesh_vertnum[mesh_id])]
    world = data.geom_xpos[geom.id] + vertices @ data.geom_xmat[geom.id].reshape(3, 3).T
    low, high = world.min(axis=0), world.max(axis=0)
    assert low[:2] == pytest.approx([0.0, 0.0], abs=0.01)
    assert high[0] == pytest.approx(MAP_WIDTH_M, abs=0.01)
    assert high[1] == pytest.approx(MAP_HEIGHT_M, abs=0.01)
    assert low[2] == pytest.approx(0.0, abs=0.01)
    assert high[2] == pytest.approx(FIELD_TOP_M, abs=0.01)


def test_hfield_spans_the_map_and_sits_on_its_centre(scene):
    model, data = scene
    assert (int(model.hfield_ncol[0]), int(model.hfield_nrow[0])) == (560, 320)
    assert list(model.hfield_size[0][:2]) == [MAP_WIDTH_M / 2, MAP_HEIGHT_M / 2]
    assert list(data.geom_xpos[model.geom("rmuc_2027_field").id][:2]) == [14.0, 8.0]
    assert float(model.hfield_data.max()) == pytest.approx(1.0)


def test_hfield_follows_the_field_mesh_where_the_robot_can_drive(scene):
    model, data = scene
    rng = np.random.default_rng(0)
    points = np.c_[rng.uniform(1.0, 27.0, 1500), rng.uniform(1.0, 15.0, 1500)]
    clamp = float(model.hfield_size[0][2])
    mesh = np.array([_field_top(model, data, x, y) for x, y in points])
    field = np.array([_hfield_height(model, x, y) for x, y in points])
    error = np.abs(field - np.minimum(mesh, clamp))
    assert np.isfinite(mesh).all()
    assert float(np.mean(error)) < 0.03
    assert float(np.mean(error < 0.05)) > 0.93
    # A transposed or mirrored image fails by a wide margin (mean error ~0.1 m).
    mirrored = np.array([_hfield_height(model, x, 16.0 - y) for x, y in points])
    assert float(np.mean(np.abs(mirrored - np.minimum(mesh, clamp)))) > 2 * float(np.mean(error))


def _hits_down(model, data, x, y):
    """Every surface of the LiDAR-visible mesh under (x, y), highest first."""
    groups = np.zeros(mujoco.mjNGROUP, dtype=np.uint8)
    groups[2] = 1
    geom_id = np.zeros(1, dtype=np.int32)
    heights, z = [], 1.5
    for _ in range(8):
        distance = mujoco.mj_ray(
            model, data, np.array([x, y, z]), np.array([0.0, 0.0, -1.0]), groups, 1, -1, geom_id,
        )
        if distance < 0.0:
            break
        z -= distance
        heights.append(z)
        z -= 1e-4
    return heights


def test_tunnel_yaml_matches_the_measured_mesh(scene):
    """tunnels_2027.yaml was measured on the official mesh; a drifted yaml fails here."""
    model, data = scene
    config = yaml.safe_load(TUNNELS_YAML.read_text(encoding="utf-8"))
    for tunnel in config["tunnels"]:
        name = tunnel["name"]
        xs = [p[0] for p in tunnel["polygon"]]
        ys = [p[1] for p in tunnel["polygon"]]
        centre = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
        top, bottom, floor = _hits_down(model, data, *centre)[:3]
        assert top == pytest.approx(tunnel["roof_top_z"], abs=0.02), name
        assert bottom == pytest.approx(tunnel["roof_bottom_z"], abs=0.02), name
        assert floor == pytest.approx(tunnel["floor_z"], abs=0.02), name
        # The lintel is as long as the yaml says: just outside it there is no roof.
        for outside_x in (min(xs) - 0.1, max(xs) + 0.1):
            assert len(_hits_down(model, data, outside_x, centre[1])) <= 2, name


def test_tunnel_corridor_is_as_wide_as_the_yaml_says(scene):
    model, data = scene
    config = yaml.safe_load(TUNNELS_YAML.read_text(encoding="utf-8"))
    for tunnel in config["tunnels"]:
        name = tunnel["name"]
        yaw = tunnel["axis_yaw"]
        axis = np.array([np.cos(yaw), np.sin(yaw)])
        lateral = np.array([-np.sin(yaw), np.cos(yaw)])
        roof_xs = [p[0] for p in tunnel["polygon"]]
        roof_centre = np.array([(min(roof_xs) + max(roof_xs)) / 2,
                                sum(p[1] for p in tunnel["polygon"]) / 4])
        axis_point = np.array(tunnel["entry"][:2])
        # Axis line of the corridor: the entry/exit points lie on it.
        along = float(np.dot(roof_centre - axis_point, axis))
        centre = axis_point + axis * along
        probe = centre + axis * (0.34 / 2 + 0.2)           # beyond the lintel, walls still on
        floor = tunnel["floor_z"]
        half = tunnel["width"] / 2
        for side in (-1.0, 1.0):
            inside = probe + lateral * side * (half - 0.05)
            wall = probe + lateral * side * (half + 0.05)
            assert _hits_down(model, data, *inside)[0] == pytest.approx(floor, abs=0.03), name
            assert _hits_down(model, data, *wall)[0] > floor + 0.15, name


def test_tunnel_interiors_are_open_at_the_mesh_floor(scene):
    model, data = scene
    config = yaml.safe_load(TUNNELS_YAML.read_text(encoding="utf-8"))
    for tunnel in config["tunnels"]:
        xs = [p[0] for p in tunnel["polygon"]]
        ys = [p[1] for p in tunnel["polygon"]]
        centre = (sum(xs) / len(xs), sum(ys) / len(ys))
        floor = _hfield_height(model, *centre)
        assert floor == pytest.approx(tunnel["floor_z"], abs=0.02), tunnel["name"]
        assert floor < tunnel["roof_bottom_z"] - 0.2, tunnel["name"]
