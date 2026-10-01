"""RMUC 2025 map, Gazebo mesh and MuJoCo scene alignment contracts."""

from hashlib import sha256
import re
from pathlib import Path
from xml.etree import ElementTree

import mujoco
import numpy as np
from PIL import Image
import yaml

from ats_mujoco_sim.rmuc_wall_collisions import decompose_rectangles


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path(__file__).resolve().parents[4]
MAP_YAML = WORKSPACE / "src/ats_sentry_bringup/map/rmuc_2025.yaml"
GAZEBO_MESH = (
    WORKSPACE
    / "src/sim/gazebo_simulator/rmu_gazebo_simulator/resource/models/"
    "rmuc_2025/meshes/rmuc_2025.stl"
)
MUJOCO_MESH = PACKAGE_ROOT / "models/meshes/rmuc_2025.stl"
MODEL_XML = PACKAGE_ROOT / "models/rmuc_2025_swerve.xml"
HEIGHT_IMAGE = PACKAGE_ROOT / "models/rmuc_2025_height.png"
NAVIGATION_PROFILE = PACKAGE_ROOT / "config/rmuc_2025_navigation.yaml"
RMUC_LAUNCH = PACKAGE_ROOT / "launch/rmuc_2025_mujoco.launch.py"
MUJOCO_RVIZ = PACKAGE_ROOT / "rviz/mujoco_navigation.rviz"
MUJOCO_REGRESSION_SCRIPTS = (
    WORKSPACE / "scripts/test_mujoco_minco_mpc_chain.sh",
    WORKSPACE / "scripts/test_mujoco_unsafe_trajectory.sh",
    WORKSPACE / "scripts/test_mujoco_localization_fault.sh",
    WORKSPACE / "scripts/collect_realtime_profile.sh",
    WORKSPACE / "scripts/test_mujoco_goal_set.sh",
)
GAZEBO_WORLD_YAML = (
    WORKSPACE
    / "src/sim/gazebo_simulator/rmu_gazebo_simulator/config/gz_world.yaml"
)
GAZEBO_XMACRO = WORKSPACE / "src/ats_robot_description/resource/xmacro/ats_sentry_robot.sdf.xmacro"
GAZEBO_SPAWN_LAUNCH = (
    WORKSPACE
    / "src/sim/gazebo_simulator/rmu_gazebo_simulator/launch/spawn_robots.launch.py"
)
WALL_BOXES = PACKAGE_ROOT / "models/rmuc_2025_wall_boxes.xml"
HEIGHT_ELEVATION_M = 0.899085


def _map_data():
    metadata = yaml.safe_load(MAP_YAML.read_text(encoding="utf-8"))
    image_path = MAP_YAML.parent / metadata["image"]
    return metadata, np.asarray(Image.open(image_path).convert("L"))


def _pixel(metadata, image, x, y):
    column = int((x - metadata["origin"][0]) / metadata["resolution"])
    row = image.shape[0] - 1 - int(
        (y - metadata["origin"][1]) / metadata["resolution"]
    )
    return row, column


def test_mujoco_installs_the_exact_gazebo_field_mesh() -> None:
    assert sha256(MUJOCO_MESH.read_bytes()).digest() == sha256(
        GAZEBO_MESH.read_bytes()
    ).digest()


def test_start_and_central_highland_goal_have_full_free_footprint_patches() -> None:
    metadata, image = _map_data()
    assert image.shape == (324, 583)
    assert metadata["resolution"] == 0.05
    assert metadata["origin"] == [-3.58, -9.44, 0]

    for x, y in ((-0.18, 0.06), (10.45, 0.35)):
        row, column = _pixel(metadata, image, x, y)
        footprint_patch = image[row - 8 : row + 9, column - 8 : column + 9]
        assert footprint_patch.shape == (17, 17)
        assert bool((footprint_patch >= 200).all())


def test_heightfield_preserves_start_floor_and_central_highland() -> None:
    metadata, _ = _map_data()
    height_image = np.asarray(Image.open(HEIGHT_IMAGE).convert("L"))

    start_row, start_column = _pixel(metadata, height_image, -0.18, 0.06)
    goal_row, goal_column = _pixel(metadata, height_image, 10.45, 0.35)
    start_height = height_image[start_row, start_column] / 255.0 * HEIGHT_ELEVATION_M
    goal_height = height_image[goal_row, goal_column] / 255.0 * HEIGHT_ELEVATION_M

    assert np.isclose(start_height, 0.201, atol=0.01)
    assert np.isclose(goal_height, 0.504, atol=0.01)
    assert goal_height - start_height > 0.28


def test_rog_projection_excludes_rmuc_2025_traversable_ground_returns() -> None:
    metadata, _ = _map_data()
    height_image = np.asarray(Image.open(HEIGHT_IMAGE).convert("L"))
    profile = yaml.safe_load(NAVIGATION_PROFILE.read_text(encoding="utf-8"))
    adapter = profile["ats_rog_map_adapter"]["ros__parameters"]
    projection_min = float(adapter["projection_min_height"])
    projection_max = float(adapter["projection_max_height"])

    heights = []
    for x, y in ((-0.18, 0.06), (10.45, 0.35)):
        row, column = _pixel(metadata, height_image, x, y)
        heights.append(height_image[row, column] / 255.0 * HEIGHT_ELEVATION_M)

    assert projection_min > max(heights) + 0.05
    assert projection_min < projection_max <= 1.0


def test_navigation_components_share_the_physical_chassis_footprint() -> None:
    profile = yaml.safe_load(NAVIGATION_PROFILE.read_text(encoding="utf-8"))
    expected = {
        "footprint_length": 0.58,
        "footprint_width": 0.58,
        "footprint_safety_margin": 0.02,
    }

    for node_name in (
        "ats_rog_map_adapter",
        "minco_planner",
        "ats_goal_manager",
    ):
        parameters = profile[node_name]["ros__parameters"]
        assert {name: parameters[name] for name in expected} == expected


def test_jps_clearance_matches_the_rmuc_2025_all_yaw_footprint() -> None:
    profile = yaml.safe_load(NAVIGATION_PROFILE.read_text(encoding="utf-8"))
    planner = profile["minco_planner"]["ros__parameters"]
    all_yaw_radius = np.hypot(0.58 / 2.0 + 0.02, 0.58 / 2.0 + 0.02)

    assert planner["jps_safe_distance"] >= all_yaw_radius
    assert planner["jps_safe_distance"] < all_yaw_radius + 0.01


def test_navigation_preserves_continuous_terrain_risk_until_hard_obstacle() -> None:
    profile = yaml.safe_load(NAVIGATION_PROFILE.read_text(encoding="utf-8"))
    adapter = profile["ats_rog_map_adapter"]["ros__parameters"]
    planner = profile["minco_planner"]["ros__parameters"]

    assert adapter["terrain_obstacle_value_threshold"] == 100
    assert planner["obstacle_value_threshold"] == 100


def test_navigation_bounds_escape_bypasses() -> None:
    profile = yaml.safe_load(NAVIGATION_PROFILE.read_text(encoding="utf-8"))
    planner = profile["minco_planner"]["ros__parameters"]
    goal_manager = profile["ats_goal_manager"]["ros__parameters"]

    # Contact escape is allowed only with a contact-depth bound of at most half
    # a planning cell; an unbounded prefix cannot tell grazing from crossing.
    if planner["escape_from_contact_enabled"]:
        depth = planner["escape_from_contact_max_contact_depth"]
        assert 0.0 < depth <= 0.05
    assert goal_manager["ego_blocked_escape_enabled"] is False
    # The watchdog tolerates exactly the contact depth the planner escapes from.
    assert goal_manager["ego_contact_max_depth"] == planner[
        "escape_from_contact_max_contact_depth"
    ]


def _goal_nine_wall_apex_x() -> float:
    """Westernmost physical wall face in the goal 9 approach band."""
    root = ElementTree.fromstring(
        "<root>" + WALL_BOXES.read_text(encoding="utf-8") + "</root>"
    )
    apex = float("inf")
    for geom in root.iter("geom"):
        px, py, _ = (float(v) for v in geom.attrib["pos"].split())
        sx, sy, _ = (float(v) for v in geom.attrib["size"].split())
        if -3.2 <= py <= -1.3 and 8.9 <= px <= 10.9:
            apex = min(apex, px - sx)
    assert apex < float("inf")
    return apex


def test_planar_lattice_keeps_goal_nine_reachable_in_every_phase() -> None:
    """The traversability grid is published at planarVoxelSize and sampled
    nearest-cell into the 0.10 m planning grid, so one coarse obstacle verdict
    is repainted onto every planning cell whose centre falls inside it. The
    lattice is anchored to the vehicle, so the resulting wall face slides with
    the robot; goal 9 must stay feasible at the worst lattice phase."""
    profile = yaml.safe_load(NAVIGATION_PROFILE.read_text(encoding="utf-8"))
    terrain = profile["terrain_analysis_ext"]["ros__parameters"]
    adapter = profile["ats_rog_map_adapter"]["ros__parameters"]
    planning_resolution = 0.10

    voxel = terrain["planarVoxelSize"]
    apex = _goal_nine_wall_apex_x()
    # Worst-case westward over-report: the coarse voxel may start up to one
    # voxel west of the apex, and the first repainted planning cell centre may
    # sit up to half a planning cell west of that voxel edge.
    worst_case_face = apex - voxel - 0.5 * planning_resolution
    # Forward footprint extent at goal yaw 0, plus the goal position tolerance.
    forward_extent = adapter["footprint_length"] / 2.0 + adapter[
        "footprint_safety_margin"
    ]
    required_face = 9.25 - 0.08 + forward_extent

    assert worst_case_face >= required_face
    # The historical 0.4 m lattice does not satisfy the same endpoint bound.
    # This contract covers goal-pose feasibility only; it does not prove that
    # the optimized terminal approach remains inside the same clearance bound.
    assert apex - 0.4 - 0.5 * planning_resolution < required_face
    # The lattice is centred on the vehicle cell, so the width stays odd; the
    # published extent stays within one voxel of the historical 101 x 0.4 m grid.
    assert terrain["planarVoxelWidth"] % 2 == 1
    assert abs(terrain["planarVoxelWidth"] * voxel - 101 * 0.4) <= voxel + 1e-6


def test_rmuc_launch_applies_the_profile_to_terrain_analysis_ext() -> None:
    """planarVoxelSize only reaches the node if the RMUC profile is layered on
    top of the shared params file for terrain_analysis_ext as well."""
    launch_text = RMUC_LAUNCH.read_text(encoding="utf-8")
    start = launch_text.index("terrain_ext = Node(")
    end = launch_text.index("localization_fusion = Node(", start)
    terrain_ext_block = launch_text[start:end]

    assert "rmuc_2025_navigation_profile" in terrain_ext_block
    assert terrain_ext_block.index('LaunchConfiguration("params_file")') < (
        terrain_ext_block.index("rmuc_2025_navigation_profile")
    )


def test_rmuc_profile_hides_debug_height_band_and_tunes_mpc_corner_tracking() -> None:
    """The brown band is a debug ESDF overlay; MuJoCo uses the same profile for MPC tuning."""
    profile = yaml.safe_load(NAVIGATION_PROFILE.read_text(encoding="utf-8"))
    adapter = profile["ats_rog_map_adapter"]["ros__parameters"]
    mpc = profile["ats_swerve_mpc"]["ros__parameters"]
    assert adapter["esdf_cloud_topic"] == ""
    assert mpc["control_delta_weight"] == [0.70, 0.70, 0.60]
    assert np.isclose(mpc["max_awz"], 2.0)
    assert np.isclose(
        profile["minco_planner"]["ros__parameters"]["yaw_acceleration_limit"], 2.0
    )

    launch_text = RMUC_LAUNCH.read_text(encoding="utf-8")
    mpc_start = launch_text.index("    mpc = Node(")
    bridge_start = launch_text.index("    twist_bridge = Node(", mpc_start)
    mpc_block = launch_text[mpc_start:bridge_start]
    assert "rmuc_2025_navigation_profile" in mpc_block

    rviz = yaml.safe_load(MUJOCO_RVIZ.read_text(encoding="utf-8"))
    esdf_displays = [
        display for display in rviz["Visualization Manager"]["Displays"]
        if display.get("Topic", {}).get("Value") == "/rc_esdf/esdf_cloud"
    ]
    assert len(esdf_displays) == 1
    assert esdf_displays[0]["Enabled"] is False


def test_rmuc_2025_gazebo_spawn_z_is_the_field_root_pose() -> None:
    world = yaml.safe_load(GAZEBO_WORLD_YAML.read_text(encoding="utf-8"))
    rmuc_entry = world["robots"]["rmuc_2025"][0]
    spawn_z = float(rmuc_entry["z_pose"])
    assert np.isclose(spawn_z, 0.20)

    xmacro_root = ElementTree.parse(GAZEBO_XMACRO).getroot()
    model_pose = xmacro_root.find("./model/pose")
    assert model_pose is not None
    model_z = float(model_pose.text.split()[2])
    # The model origin is base_footprint on the ground, so the spawn z is the
    # field surface height and the xmacro root pose adds no lift.
    assert np.isclose(model_z, 0.0)

    spawn_source = GAZEBO_SPAWN_LAUNCH.read_text(encoding="utf-8")
    assert '-z",\n                robot["z_pose"]' in spawn_source


def test_rmuc_2025_mujoco_start_z_matches_wheel_clearance() -> None:
    model = ElementTree.parse(MODEL_XML).getroot()
    base_link = model.find(".//body[@name='base_link']")
    steer_link = model.find(".//body[@name='front_left_steer_link']")
    wheel_default = model.find("./default/default[@class='wheel_geom']/geom")
    assert base_link is not None
    assert steer_link is not None
    assert wheel_default is not None
    assert np.isclose(float(base_link.attrib["pos"].split()[2]), 0.18)
    wheel_offset_z = float(steer_link.attrib["pos"].split()[2])
    wheel_radius = float(wheel_default.attrib["size"].split()[0])

    metadata, _ = _map_data()
    height_image = np.asarray(Image.open(HEIGHT_IMAGE).convert("L"))
    row, column = _pixel(metadata, height_image, -0.18, 0.06)
    field_height = height_image[row, column] / 255.0 * HEIGHT_ELEVATION_M
    expected_start_z = field_height - wheel_offset_z + wheel_radius
    launch_source = RMUC_LAUNCH.read_text(encoding="utf-8")
    match = re.search(
        r'DeclareLaunchArgument\("start_z", default_value="([0-9.]+)"\)',
        launch_source,
    )
    assert match is not None
    assert np.isclose(float(match.group(1)), expected_start_z, atol=0.002)


def test_rmuc_launch_enables_minco_diagnostic_outputs() -> None:
    launch_source = RMUC_LAUNCH.read_text(encoding="utf-8")
    minco_start = launch_source.index("    minco = Node(")
    mpc_start = launch_source.index("    mpc = Node(", minco_start)
    minco_block = launch_source[minco_start:mpc_start]
    assert "rmuc_2025_navigation_profile" in minco_block

    profile = yaml.safe_load(NAVIGATION_PROFILE.read_text(encoding="utf-8"))
    parameters = profile["minco_planner"]["ros__parameters"]
    assert parameters["preprocessed_guide_topic"] == "/minco/preprocessed_guide"
    assert parameters["esdf_refined_guide_topic"] == "/minco/esdf_refined_guide"
    assert parameters["debug_marker_topic"] == "/minco/debug_markers"


def test_rmuc_regression_entries_keep_robot_grounded_and_show_minco_output() -> None:
    for script in MUJOCO_REGRESSION_SCRIPTS:
        source = script.read_text(encoding="utf-8")
        assert not re.search(r"(?:START_Z[^\n]*0\.42|start_z:=0\.42)", source)
        assert "0.381" in source

    rviz = yaml.safe_load(MUJOCO_RVIZ.read_text(encoding="utf-8"))
    displays = {
        display["Topic"]["Value"]: display
        for display in rviz["Visualization Manager"]["Displays"]
        if isinstance(display.get("Topic"), dict)
    }
    # JPS, MINCO and MPC are the whole trajectory story; the intermediate MINCO
    # guides and ROGMap voxel clouds are no longer shown.
    for topic in (
        "/minco/raw_path",
        "/minco/reference_path",
        "/ats_swerve_mpc/predicted_path",
    ):
        assert displays[topic]["Enabled"] is True
    for topic in ("/minco/preprocessed_guide", "/minco/esdf_refined_guide", "/rog_map/viz"):
        assert topic not in displays

def test_wall_decomposition_exactly_covers_map_occupied_cells() -> None:
    metadata, image = _map_data()
    occupied = image < int(round(metadata["occupied_thresh"] * 255.0))
    reconstructed = np.zeros_like(occupied)
    rectangles = decompose_rectangles(occupied)
    for rectangle in rectangles:
        reconstructed[
            rectangle.y : rectangle.y + rectangle.height,
            rectangle.x : rectangle.x + rectangle.width,
        ] = True

    assert rectangles
    assert np.array_equal(reconstructed, occupied)


def test_scene_uses_mesh_for_rays_and_map_aligned_geometries_for_contact() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    visual_id = model.geom("rmuc_2025_field_visual").id
    terrain_id = model.geom("rmuc_2025_field").id
    assert model.geom_group[visual_id] == 2
    assert model.geom_contype[visual_id] == 0
    assert model.geom_conaffinity[visual_id] == 0
    mesh_id = int(model.geom_dataid[visual_id])
    vertex_address = int(model.mesh_vertadr[mesh_id])
    vertex_count = int(model.mesh_vertnum[mesh_id])
    compiled_vertices = np.asarray(
        model.mesh_vert[vertex_address : vertex_address + vertex_count]
    )
    visual_rotation = np.asarray(data.geom_xmat[visual_id]).reshape(3, 3)
    world_vertices = (
        compiled_vertices @ visual_rotation.T + data.geom_xpos[visual_id]
    )
    assert np.allclose(
        world_vertices.min(axis=0), (-3.66185, -9.517887, 0.0), atol=1e-5
    )
    assert np.allclose(
        world_vertices.max(axis=0), (25.50185, 6.637887, 18.754619), atol=1e-5
    )
    assert np.allclose(model.geom_pos[terrain_id], (10.995, -1.34, 0.0))
    assert model.geom_group[terrain_id] == 3
    assert np.allclose(
        model.hfield_size[0], (14.575, 8.100, HEIGHT_ELEVATION_M, 0.030)
    )

    ray_groups = np.zeros(mujoco.mjNGROUP, dtype=np.uint8)
    ray_groups[2] = 1
    hit_id = np.array([-1], dtype=np.int32)
    distance = mujoco.mj_ray(
        model,
        data,
        np.array([-0.18, 0.06, 0.40]),
        np.array([1.0, 0.0, 0.0]),
        ray_groups,
        1,
        -1,
        hit_id,
    )
    assert distance > 0.0
    assert hit_id[0] == visual_id


def test_min_point_count_tracks_the_planar_cell_area() -> None:
    """细化体素不得顺带把密度门槛抬高,否则地形证据变少的方向朝不安全一侧。"""
    profile = yaml.safe_load(NAVIGATION_PROFILE.read_text(encoding="utf-8"))
    terrain = profile["terrain_analysis_ext"]["ros__parameters"]
    voxel = terrain["planarVoxelSize"]
    min_points = terrain["traversabilityMinPointCount"]

    baseline_voxel = 0.4
    baseline_min_points = 3
    baseline_density = baseline_min_points / baseline_voxel ** 2
    density = min_points / voxel ** 2

    # 不得比历史设置更宽松。
    assert density >= baseline_density
    # 也不得让单点噪声独自构成地形判定。
    assert min_points >= 2
