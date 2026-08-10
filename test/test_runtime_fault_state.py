"""LiDAR occlusion publication contract: empty returns keep a legal cadence.

Like ``test_freeze_motion_parameter``, this imports only the ROS-free
``runtime_fault_state`` module so it is collectible without a built workspace.
"""

import numpy as np

from ats_mujoco_sim.runtime_fault_state import empty_lidar_return
from ats_mujoco_sim.runtime_fault_state import normalize_pointcloud_points
from ats_mujoco_sim.runtime_fault_state import pointcloud_payload
from ats_mujoco_sim.runtime_fault_state import resolve_lidar_return

#: XYZI, four float32 fields.
POINT_STEP = 16


class RecordingPointCloud:
    """Minimal stand-in for the reused ``PointCloud2`` message object."""

    def __init__(self) -> None:
        self.point_step = POINT_STEP
        self.width = -1
        self.row_step = -1
        self.data = b""
        self.stamp = None
        self.frame_id = "front_mid360"


def publish(msg: RecordingPointCloud, stamp, points) -> None:
    """Mirror of the publication path in ``sim_node._publish_pointcloud``."""
    payload = pointcloud_payload(points, msg.point_step)
    msg.stamp = stamp
    msg.width = payload.width
    msg.row_step = payload.row_step
    msg.data = payload.data


def test_occluded_return_is_an_empty_but_legally_shaped_cloud() -> None:
    points = empty_lidar_return()

    assert points.shape == (0, 3)
    assert points.dtype == np.float32


def test_occlusion_skips_the_ray_sweep_entirely() -> None:
    sweeps = []

    def raycast():
        sweeps.append(1)
        return np.ones((5, 3), dtype=np.float32)

    occluded = resolve_lidar_return(True, raycast)
    assert occluded.shape == (0, 3)
    assert sweeps == []

    clear = resolve_lidar_return(False, raycast)
    assert clear.shape == (5, 3)
    assert sweeps == [1]


def test_empty_cloud_still_carries_a_legal_frame_stamp_and_zero_width() -> None:
    msg = RecordingPointCloud()

    publish(msg, stamp=(12, 340000000), points=empty_lidar_return())

    assert msg.frame_id == "front_mid360"
    assert msg.stamp == (12, 340000000)
    assert msg.width == 0
    assert msg.row_step == 0
    assert msg.data == b""


def test_publication_cadence_continues_across_an_occlusion_window() -> None:
    msg = RecordingPointCloud()
    # Clear, occluded, occluded, clear: every tick must publish.
    schedule = [False, True, True, False]
    stamps = []

    for index, occluded in enumerate(schedule):
        points = resolve_lidar_return(
            occluded, lambda: np.ones((3, 3), dtype=np.float32)
        )
        publish(msg, stamp=(index, 0), points=points)
        stamps.append((msg.stamp, msg.width))

    assert stamps == [((0, 0), 3), ((1, 0), 0), ((2, 0), 0), ((3, 0), 3)]


def test_three_column_points_gain_an_intensity_column() -> None:
    normalized = normalize_pointcloud_points(np.ones((2, 3), dtype=np.float32))

    assert normalized.shape == (2, 4)
    assert np.allclose(normalized[:, 3], 1.0)


def test_four_column_points_pass_through_unchanged() -> None:
    source = np.arange(8, dtype=np.float32).reshape(2, 4)

    normalized = normalize_pointcloud_points(source)

    assert normalized.shape == (2, 4)
    assert np.array_equal(normalized, source)


def test_row_step_tracks_the_published_width() -> None:
    payload = pointcloud_payload(np.ones((7, 3), dtype=np.float32), POINT_STEP)

    assert payload.width == 7
    assert payload.row_step == 7 * POINT_STEP
    assert len(payload.data) == 7 * POINT_STEP


def test_malformed_point_shape_is_rejected() -> None:
    try:
        normalize_pointcloud_points(np.ones((2, 5), dtype=np.float32))
    except ValueError as exception:
        assert "shape (N, 3) or (N, 4)" in str(exception)
    else:  # pragma: no cover - the call above must raise
        raise AssertionError("a (2, 5) cloud must be rejected")
