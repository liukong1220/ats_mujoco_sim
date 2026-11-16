"""Scan-order bookkeeping for the time-resolved Mid360 simulation.

No ROS imports. The lidar process splits each frame's rays into short segments,
traces every segment against the simulation state of its own time, and tags
each hit with the time the beam fired (``offset_time``) and its scan line.
The C++ ``ats_livox_adapter`` turns the resulting PointCloud2 into
``livox_ros_driver2/CustomMsg`` (building that message in rclpy costs about
121 ms per frame, see gz_livox_bridge.cpp).
"""

import math
from array import array

import numpy as np

# The library's 24k template is geometry, not the sensor's firing rate.
# Livox MID360 specifies 200k first-return points/s and a typical 10 Hz frame.
POINT_RATE_HZ = 200000
PATTERN_PERIOD_S = 0.1

SCAN_LINES = 4
DEFAULT_MAX_SEGMENT_S = 0.005

# PointCloud2 layout shared with the C++ adapter (point_step 24).
POINT_STEP = 24
LIVOX_POINT_DTYPE = np.dtype(
    {
        "names": ["x", "y", "z", "intensity", "offset_time", "line", "tag"],
        "formats": ["<f4", "<f4", "<f4", "<f4", "<u4", "u1", "u1"],
        "offsets": [0, 4, 8, 12, 16, 20, 21],
        "itemsize": POINT_STEP,
    }
)


def rays_per_frame(frame_period_s, downsample=1):
    """Rays fired in one frame so the point rate matches the real sensor."""
    native = POINT_RATE_HZ * float(frame_period_s)
    return max(1, int(round(native / max(1, int(downsample)))))


def segment_count(frame_period_s, max_segment_s=DEFAULT_MAX_SEGMENT_S):
    return max(1, int(math.ceil(float(frame_period_s) / float(max_segment_s) - 1e-9)))


def segment_bounds(ray_count, segments):
    """Cumulative ray indices ``[0 .. ray_count]`` of the segment edges."""
    return np.rint(np.linspace(0, int(ray_count), int(segments) + 1)).astype(np.int64)


def sensor_pattern(pattern):
    """Uniformly resample a full template period to the specified sensor rate."""
    samples = int(round(POINT_RATE_HZ * PATTERN_PERIOD_S))
    indices = np.arange(samples, dtype=np.int64) * len(pattern) // samples
    return np.ascontiguousarray(pattern[indices])


def take_rays(pattern, cursor, count, stride=1):
    """Thin across the whole exposure instead of slowing the scan pattern."""
    total = len(pattern)
    start = int(cursor) % total
    indices = (start + np.arange(count, dtype=np.int64) * int(stride)) % total
    return pattern[indices], (start + count * int(stride)) % total


def offset_time_ns(ray_indices, ray_count, frame_period_s):
    """Firing time of each ray inside its frame, in whole nanoseconds."""
    period_ns = int(round(float(frame_period_s) * 1e9))
    return (np.asarray(ray_indices, dtype=np.int64) * period_ns // int(ray_count)).astype(
        np.uint32
    )


def segment_midtime(frame_start_s, frame_period_s, bounds, segment):
    """Simulation time at the middle of segment ``segment``."""
    total = float(bounds[-1])
    first = float(bounds[segment])
    last = float(bounds[segment + 1])
    return float(frame_start_s) + float(frame_period_s) * 0.5 * (first + last) / total


def nearest_state_index(state_times, target):
    """Index of the buffered state closest in time to ``target``."""
    times = np.asarray(state_times, dtype=np.float64)
    pos = int(np.searchsorted(times, target))
    if pos <= 0:
        return 0
    if pos >= len(times):
        return len(times) - 1
    return pos if (times[pos] - target) < (target - times[pos - 1]) else pos - 1


def pack_points(points_xyz, offset_ns, line, intensity=100.0):
    """Pack a typed byte buffer accepted directly by PointCloud2.data."""
    count = len(points_xyz)
    cloud = np.zeros(count, dtype=LIVOX_POINT_DTYPE)
    if count:
        cloud["x"] = points_xyz[:, 0]
        cloud["y"] = points_xyz[:, 1]
        cloud["z"] = points_xyz[:, 2]
        cloud["intensity"] = intensity
        cloud["offset_time"] = offset_ns
        cloud["line"] = line
    return array("B", cloud.tobytes())
