"""Launch arguments shared by the MuJoCo launch files, loaded from YAML config."""

from pathlib import Path
from typing import Dict, Optional

import yaml

# Keys that belong to the shared LIO/sensor launch-argument dictionary.
_LIO_ARGUMENT_KEYS = (
    "start_roll",
    "start_pitch",
    "enable_test_obstacle_fixture",
    "use_identified_odometry",
    "localization_source",
    "imu_topic",
    "imu_rate_hz",
    "ground_truth_odom_topic",
    "ground_truth_rate_hz",
    "livox_points_topic",
    "lidar_segment_ms",
    "lidar_state_queue_depth",
    "lidar_threads",
    "lidar_max_range",
    "lidar_self_exclusion",
    "noise_seed",
    "enable_range_noise",
    "range_noise_sigma_m",
    "enable_imu_noise",
    "enable_imu_saturation",
    "lidar_mount_preset",
    "lidar_mount_x",
    "lidar_mount_y",
    "lidar_mount_height",
    "lidar_mount_tilt_direction",
    "lidar_mount_tilt_deg",
    "base_footprint_height",
)


def _as_launch_str(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def resolve_mujoco_launch_defaults_path(config_path: Optional[str] = None) -> Path:
    if config_path:
        return Path(config_path)
    # 1) Installed ROS 2 share directory when available.
    try:
        from ament_index_python.packages import get_package_share_directory
        installed = Path(get_package_share_directory("ats_mujoco_sim")) / "config" / "mujoco_launch_defaults.yaml"
        if installed.is_file():
            return installed
    except Exception:
        pass
    # 2) Source-tree fallback (works without sourcing ROS or during static inspection).
    return Path(__file__).resolve().parents[1] / "config" / "mujoco_launch_defaults.yaml"


def load_mujoco_launch_defaults(
    profile: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Dict[str, str]:
    """Load merged launch defaults (localization + auto_mapping + simulation + profile)."""
    path = resolve_mujoco_launch_defaults_path(config_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    merged: Dict[str, str] = {}
    for section in ("localization", "auto_mapping", "simulation"):
        for key, value in (raw.get(section) or {}).items():
            merged[str(key)] = _as_launch_str(value)
    if profile:
        profile_cfg = (raw.get("profiles") or {}).get(profile) or {}
        for key, value in profile_cfg.items():
            merged[str(key)] = _as_launch_str(value)
    return merged


def load_lio_arguments(
    profile: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Dict[str, str]:
    """Return the shared LIO/sensor argument defaults for a given MuJoCo profile."""
    merged = load_mujoco_launch_defaults(profile=profile, config_path=config_path)
    return {key: merged[key] for key in _LIO_ARGUMENT_KEYS if key in merged}


# Dynamically loaded from config/mujoco_launch_defaults.yaml (no inline hardcoding).
SIM_LIO_ARGUMENTS: Dict[str, str] = load_lio_arguments()
