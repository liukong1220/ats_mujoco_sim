"""Launch arguments shared by the MuJoCo launch files (no ROS import)."""

# LIO-mode and lidar-mount parameters (report appendix A.21, 2.1.8).  The
# defaults keep the pre-existing truth behaviour; rmuc_2027_mujoco.launch.py
# switches the mount preset to the chassis mount.
SIM_LIO_ARGUMENTS = {
    # Optional parking attitude for slope sampling; existing spawns stay level.
    "start_roll": "0.0",
    "start_pitch": "0.0",
    "enable_test_obstacle_fixture": "false",
    "use_identified_odometry": "false",
    "localization_source": "truth",
    "imu_topic": "/livox/imu",
    "imu_rate_hz": "200.0",
    "ground_truth_odom_topic": "/simulation/ground_truth/odometry",
    "ground_truth_rate_hz": "50.0",
    "livox_points_topic": "/mujoco/livox_points",
    "lidar_segment_ms": "5.0",
    "lidar_state_queue_depth": "2",
    "lidar_threads": "0",  # LIO ray-tracing threads; 0 = auto (cpu_count // 4, at most 8)
    "lidar_max_range": "40.0",
    "lidar_self_exclusion": "true",
    "noise_seed": "20260723",
    "enable_range_noise": "false",
    "range_noise_sigma_m": "0.02",
    "enable_imu_noise": "false",
    "enable_imu_saturation": "false",
    "lidar_mount_preset": "legacy_gimbal",
    "lidar_mount_x": "0.2405",
    "lidar_mount_y": "0.0",
    "lidar_mount_height": "0.20",
    "lidar_mount_tilt_direction": "front",
    "lidar_mount_tilt_deg": "30.0",     # real vehicle (user 2026-10-25), same as bringup's 0.5236 rad
    "base_footprint_height": "0.18",
}
