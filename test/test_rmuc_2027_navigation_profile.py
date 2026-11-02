"""The 2027-only navigation overlay: the ramp_bridge plate must not be a terrain obstacle.

Attribution (report 5.1.12 item 1): with the vehicle parked north of the plate the planning grid is
occupied under it, the ROG projection (absolute odom z band 0.60..1.00) and the slope grid
(4..13 deg) do not mark it, and `/traversability_grid` does (hard obstacle, 100).  The plate
underside is 0.26..0.35 m above the floor, so terrain_analysis_ext's voxel hard-obstacle rule
(max(height difference, max relative height) >= 0.18 m) fires.  The overlay lets terrain_analysis_ext
ignore points higher than vehicle height + a margin that is smaller than the plate's 1.8 cm clearance.
"""

from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path(__file__).resolve().parents[4]
OVERLAY = PACKAGE_ROOT / "config/rmuc_2027_navigation.yaml"
PROFILE_2025 = PACKAGE_ROOT / "config/rmuc_2025_navigation.yaml"
LAUNCH_2025 = PACKAGE_ROOT / "launch/rmuc_2025_mujoco.launch.py"
LAUNCH_2027 = PACKAGE_ROOT / "launch/rmuc_2027_mujoco.launch.py"
NODE_PARAMS = WORKSPACE / "src/ats_sentry_bringup/params/node_params.yaml"
MODEL_XML = PACKAGE_ROOT / "models/rmuc_2027_swerve.xml"

VEHICLE_HEIGHT_M = 0.242          # including the lidar (user 2026-10-17)
PLATE_MIN_CLEARANCE_M = 0.260     # measured at the plate's south edge (report 5.1.10 G)
KEY = "traversabilityOverheadHeight"


def _terrain_ext(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data.get("terrain_analysis_ext", {}).get("ros__parameters", {})


def test_overlay_margin_over_vehicle_height_is_below_the_plate_clearance_margin() -> None:
    overhead = _terrain_ext(OVERLAY)[KEY]
    margin = overhead - VEHICLE_HEIGHT_M
    assert margin > 0.0, "points at vehicle height must still be obstacles"
    assert margin < PLATE_MIN_CLEARANCE_M - VEHICLE_HEIGHT_M  # < 1.8 cm
    assert overhead < PLATE_MIN_CLEARANCE_M


def test_the_2025_profile_and_the_real_robot_do_not_enable_the_overhead_filter() -> None:
    assert KEY not in _terrain_ext(PROFILE_2025)
    assert KEY not in _terrain_ext(NODE_PARAMS)


def test_the_overlay_changes_nothing_else() -> None:
    data = yaml.safe_load(OVERLAY.read_text(encoding="utf-8"))
    assert list(data) == ["terrain_analysis_ext"]
    assert list(data["terrain_analysis_ext"]["ros__parameters"]) == [KEY]


def test_launch_layers_the_scene_overlay_after_the_profile_for_terrain_analysis_ext() -> None:
    text = LAUNCH_2025.read_text(encoding="utf-8")
    start = text.index("terrain_ext = Node(")
    block = text[start:text.index("localization_fusion = Node(", start)]
    assert block.index('LaunchConfiguration("params_file")') < block.index(
        "rmuc_2025_navigation_profile"
    ) < block.index('LaunchConfiguration("scene_navigation_profile")')
    # The default is the 2025 profile itself, so the 2025 scene reads the same file twice.
    arg_start = text.index('"scene_navigation_profile",')
    assert "default_value=rmuc_2025_navigation_profile" in text[arg_start:arg_start + 200]


def test_2027_launch_selects_the_2027_overlay() -> None:
    text = LAUNCH_2027.read_text(encoding="utf-8")
    assert '"scene_navigation_profile"' in text
    assert "rmuc_2027_navigation.yaml" in text


@pytest.fixture(scope="module")
def scene():
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_every_plate_cell_clears_the_overhead_height(scene) -> None:
    """terrain_analysis_ext measures relative height from a low-quantile floor, so the true clearance
    is a lower bound on what it sees; the filter threshold has to stay under it."""
    model, data = scene
    geom_id = np.array([-1], dtype=np.int32)
    gaps = []
    for x in np.arange(8.95, 9.70, 0.05):
        for y in np.arange(2.60, 2.93, 0.01):
            down = mujoco.mj_ray(
                model, data, np.array([x, y, 0.60]), np.array([0.0, 0.0, -1.0]),
                None, 1, -1, geom_id,
            )
            assert down >= 0.0
            floor = 0.60 - down
            up = mujoco.mj_ray(
                model, data, np.array([x, y, floor + 0.005]), np.array([0.0, 0.0, 1.0]),
                None, 1, -1, geom_id,
            )
            if up >= 0.0 and floor + 0.005 + up < 0.8:
                gaps.append(floor + 0.005 + up - floor)
    assert len(gaps) > 100
    overhead = _terrain_ext(OVERLAY)[KEY]
    assert min(gaps) > overhead, f"plate clearance {min(gaps):.4f} <= overhead height {overhead}"
