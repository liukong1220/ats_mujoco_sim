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


VEHICLE_HEIGHT_M = 0.242   # highest point incl. LiDAR, user 2026-10-30; = base_collision top above the floor


def _hits_up_from(model, data, x, y, z):
    """Distance from (x, y, z) up to the first collision geom (group 3), nan when nothing is above."""
    groups = np.zeros(mujoco.mjNGROUP, dtype=np.uint8)
    groups[3] = 1
    geom_id = np.zeros(1, dtype=np.int32)
    distance = mujoco.mj_ray(
        model, data, np.array([x, y, z]), np.array([0.0, 0.0, 1.0]), groups, 1, -1, geom_id,
    )
    return distance if distance >= 0.0 else float("nan")


def test_overhanging_plates_leave_the_floor_below_open(scene):
    """A single-valued hfield projects an overhang down into a solid wall (domain 115 stuck at ramp_bridge).

    Across each overhangs: plate (tilted ~1.5 deg about x) the yaml plane must match the mesh underside,
    the roof box underside must match the mesh, the mesh gap must exceed the vehicle height and the hfield
    must not rise into it, so the 0.242 m robot drives under the plate as on the real field.
    """
    model, data = scene
    config = yaml.safe_load(TUNNELS_YAML.read_text(encoding="utf-8"))
    assert {o["name"] for o in config["overhangs"]} == {
        "ramp_bridge", "ramp_bridge_mirror", "south_plate", "south_plate_mirror"}
    for overhang in config["overhangs"]:
        name = overhang["name"]
        xs = [p[0] for p in overhang["polygon"]]
        ys = [p[1] for p in overhang["polygon"]]
        centre_y = (min(ys) + max(ys)) / 2
        for x in np.linspace(min(xs) + 0.03, max(xs) - 0.03, 5):
            for y in np.linspace(min(ys) + 0.03, max(ys) - 0.03, 7):
                top, bottom, floor = _hits_down(model, data, x, y)[:3]
                plane = overhang["bottom_z"] + overhang["slope_y"] * (y - centre_y)
                assert bottom == pytest.approx(plane, abs=0.003), (name, x, y)
                assert top - bottom == pytest.approx(
                    overhang["top_z"] - overhang["bottom_z"], abs=0.003), (name, x, y)
                assert bottom - floor > VEHICLE_HEIGHT_M, (name, x, y)
                assert _hfield_height(model, x, y) < bottom - VEHICLE_HEIGHT_M, (name, x, y)
                box_gap = _hits_up_from(model, data, x, y, bottom - 0.03)
                assert box_gap == pytest.approx(0.03, abs=0.003), (name, x, y)


def test_robot_base_passes_under_the_ramp_bridge():
    """Push the free base south along x = 9.3 under ramp_bridge; it must get past the plate (y < 2.5).

    The wheels' drag holds the pushed base to about 0.14 m/s on the ramp, so it needs about 6 s for
    the 0.85 m: on this machine it is at y = 2.61 after 6 s and at 1.60 after 12 s. The old test
    budget of exactly 6 s sat on that margin. Blocked is a different outcome: with the plate in
    the hfield (07497c1) the base stops at y = 3.24 and stays there, so a 12 s budget still
    separates a solid wall from a slow pass.
    """
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)
    base = model.body("base_link").id
    qpos = int(model.jnt_qposadr[model.body_jntadr[base]])
    dof = int(model.jnt_dofadr[model.body_jntadr[base]])
    data.qpos[qpos:qpos + 3] = [9.3, 3.35, 0.37]
    data.qpos[qpos + 3:qpos + 7] = [np.cos(-np.pi / 4), 0.0, 0.0, np.sin(-np.pi / 4)]
    mujoco.mj_forward(model, data)
    for _ in range(int(12.0 / model.opt.timestep)):
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[dof + 1] = 400.0 * (-0.5 - data.qvel[dof + 1])
        data.qfrc_applied[dof] = 400.0 * -data.qvel[dof] + 2000.0 * (9.3 - data.qpos[qpos])
        data.qfrc_applied[dof + 5] = -50.0 * data.qvel[dof + 5]
        mujoco.mj_step(model, data)
        if data.qpos[qpos + 1] < 2.5:
            break
    assert data.qpos[qpos + 1] < 2.5
