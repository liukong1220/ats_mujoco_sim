"""Launch the MuJoCo swerve chassis simulation node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Create the launch description."""
    model_path = LaunchConfiguration("model_path")
    scene_file = LaunchConfiguration("scene_file")
    map_dir = LaunchConfiguration("map_dir")
    map_manifest_path = LaunchConfiguration("map_manifest_path")
    map_ready_file = LaunchConfiguration("map_ready_file")
    map_ready_token = LaunchConfiguration("map_ready_token")
    map_wait_timeout_sec = LaunchConfiguration("map_wait_timeout_sec")
    sim_rate_hz = LaunchConfiguration("sim_rate_hz")
    feedback_rate_hz = LaunchConfiguration("feedback_rate_hz")
    truth_rate_hz = LaunchConfiguration("truth_rate_hz")
    command_timeout = LaunchConfiguration("command_timeout")
    freeze_motion = LaunchConfiguration("freeze_motion")
    wheel_radius = LaunchConfiguration("wheel_radius")
    max_wheel_speed = LaunchConfiguration("max_wheel_speed")
    max_wheel_acceleration = LaunchConfiguration("max_wheel_acceleration")
    max_steer_rate = LaunchConfiguration("max_steer_rate")
    emergency_stop_topic = LaunchConfiguration("emergency_stop_topic")
    swerve_telemetry_topic = LaunchConfiguration("swerve_telemetry_topic")
    show_viewer = LaunchConfiguration("show_viewer")
    use_viewer = LaunchConfiguration("use_viewer")
    viewer_rate_hz = LaunchConfiguration("viewer_rate_hz")
    enable_lidar = LaunchConfiguration("enable_lidar")
    lidar_backend = LaunchConfiguration("lidar_backend")
    lidar_model = LaunchConfiguration("lidar_model")
    lidar_downsample = LaunchConfiguration("lidar_downsample")
    lidar_rate_hz = LaunchConfiguration("lidar_rate_hz")
    lidar_rate_clock = LaunchConfiguration("lidar_rate_clock")
    lidar_state_rate_hz = LaunchConfiguration("lidar_state_rate_hz")
    lidar_topic = LaunchConfiguration("lidar_topic")
    lidar_frame_id = LaunchConfiguration("lidar_frame_id")
    registered_scan_topic = LaunchConfiguration("registered_scan_topic")
    registered_scan_frame_id = LaunchConfiguration("registered_scan_frame_id")
    enable_tof = LaunchConfiguration("enable_tof")
    tof_backend = LaunchConfiguration("tof_backend")
    tof_range = LaunchConfiguration("tof_range")
    tof_min_range = LaunchConfiguration("tof_min_range")
    tof_rate_hz = LaunchConfiguration("tof_rate_hz")
    tof_width = LaunchConfiguration("tof_width")
    tof_height = LaunchConfiguration("tof_height")
    tof_horizontal_fov_deg = LaunchConfiguration("tof_horizontal_fov_deg")
    tof_vertical_fov_deg = LaunchConfiguration("tof_vertical_fov_deg")
    tof_footprint_length = LaunchConfiguration("tof_footprint_length")
    tof_footprint_width = LaunchConfiguration("tof_footprint_width")
    tof_footprint_expand = LaunchConfiguration("tof_footprint_expand")
    tof_footprint_resolution = LaunchConfiguration("tof_footprint_resolution")
    tof_footprint_z_min = LaunchConfiguration("tof_footprint_z_min")
    tof_footprint_z_max = LaunchConfiguration("tof_footprint_z_max")
    merged_tof_topic = LaunchConfiguration("merged_tof_topic")
    left_tof_topic = LaunchConfiguration("left_tof_topic")
    right_tof_topic = LaunchConfiguration("right_tof_topic")
    odom_topic = LaunchConfiguration("odom_topic")
    publish_map_to_odom_tf = LaunchConfiguration("publish_map_to_odom_tf")
    lidar_odometry_topic = LaunchConfiguration("lidar_odometry_topic")
    robot_base_frame_id = LaunchConfiguration("robot_base_frame_id")
    pose_cmd_topic = LaunchConfiguration("pose_cmd_topic")
    reset_pose_service_topic = LaunchConfiguration("reset_pose_service_topic")
    start_x = LaunchConfiguration("start_x")
    start_y = LaunchConfiguration("start_y")
    start_z = LaunchConfiguration("start_z")
    start_yaw = LaunchConfiguration("start_yaw")
    launch_mujoco_rviz = LaunchConfiguration("launch_mujoco_rviz")
    rviz_config_file = LaunchConfiguration("rviz_config_file")
    rviz_delay_sec = LaunchConfiguration("rviz_delay_sec")

    return LaunchDescription([
        DeclareLaunchArgument("model_path", default_value=""),
        DeclareLaunchArgument("scene_file", default_value=""),
        DeclareLaunchArgument("map_dir", default_value=""),
        DeclareLaunchArgument("map_manifest_path", default_value=""),
        DeclareLaunchArgument("map_ready_file", default_value=""),
        DeclareLaunchArgument("map_ready_token", default_value=""),
        DeclareLaunchArgument("map_wait_timeout_sec", default_value="0.0"),
        DeclareLaunchArgument("sim_rate_hz", default_value="300.0"),
        DeclareLaunchArgument("feedback_rate_hz", default_value="10.0"),
        DeclareLaunchArgument("truth_rate_hz", default_value="10.0"),
        DeclareLaunchArgument("command_timeout", default_value="0.5"),
        DeclareLaunchArgument(
            "freeze_motion",
            default_value="false",
            description="Hold the simulated chassis while keeping sensors and localization active.",
        ),
        DeclareLaunchArgument("wheel_radius", default_value="0.0425"),
        DeclareLaunchArgument("max_wheel_speed", default_value="1.6689711"),
        DeclareLaunchArgument("max_wheel_acceleration", default_value="2.0"),
        DeclareLaunchArgument("max_steer_rate", default_value="10.4719755"),
        DeclareLaunchArgument(
            "emergency_stop_topic", default_value="/planner/emergency_stop"
        ),
        DeclareLaunchArgument(
            "swerve_telemetry_topic", default_value="/swerve/telemetry"
        ),
        DeclareLaunchArgument("show_viewer", default_value="true"),
        DeclareLaunchArgument("use_viewer", default_value="true"),
        DeclareLaunchArgument("viewer_rate_hz", default_value="30.0"),
        DeclareLaunchArgument("enable_lidar", default_value="false"),
        DeclareLaunchArgument("lidar_backend", default_value="cpu"),
        DeclareLaunchArgument("lidar_model", default_value="mid360"),
        DeclareLaunchArgument("lidar_downsample", default_value="1"),
        DeclareLaunchArgument("lidar_rate_hz", default_value="10.0"),
        DeclareLaunchArgument("lidar_rate_clock", default_value="wall"),
        DeclareLaunchArgument("lidar_state_rate_hz", default_value="30.0"),
        DeclareLaunchArgument("lidar_topic", default_value="/local_pointcloud"),
        DeclareLaunchArgument("lidar_frame_id", default_value="front_mid360"),
        DeclareLaunchArgument("registered_scan_topic", default_value="/registered_scan"),
        DeclareLaunchArgument("registered_scan_frame_id", default_value=""),
        DeclareLaunchArgument("enable_tof", default_value="true"),
        DeclareLaunchArgument("tof_backend", default_value="cpu"),
        DeclareLaunchArgument("tof_range", default_value="1.0"),
        DeclareLaunchArgument("tof_min_range", default_value="0.03"),
        DeclareLaunchArgument("tof_rate_hz", default_value="10.0"),
        DeclareLaunchArgument("tof_width", default_value="248"),
        DeclareLaunchArgument("tof_height", default_value="180"),
        DeclareLaunchArgument("tof_horizontal_fov_deg", default_value="98.0"),
        DeclareLaunchArgument("tof_vertical_fov_deg", default_value="72.0"),
        DeclareLaunchArgument("tof_footprint_length", default_value="0.60"),
        DeclareLaunchArgument("tof_footprint_width", default_value="0.50"),
        DeclareLaunchArgument("tof_footprint_expand", default_value="1.0"),
        DeclareLaunchArgument("tof_footprint_resolution", default_value="0.05"),
        DeclareLaunchArgument("tof_footprint_z_min", default_value="-0.10"),
        DeclareLaunchArgument("tof_footprint_z_max", default_value="0.13"),
        DeclareLaunchArgument(
            "merged_tof_topic",
            default_value="/perception/tof/points_merged",
        ),
        DeclareLaunchArgument("left_tof_topic", default_value="/left_tof/points"),
        DeclareLaunchArgument("right_tof_topic", default_value="/right_tof/points"),
        DeclareLaunchArgument("odom_topic", default_value="/localization"),
        DeclareLaunchArgument("publish_map_to_odom_tf", default_value="true"),
        DeclareLaunchArgument("lidar_odometry_topic", default_value="/lidar_odometry"),
        DeclareLaunchArgument("robot_base_frame_id", default_value="gimbal_yaw_odom"),
        DeclareLaunchArgument("pose_cmd_topic", default_value="/simulation/PoseSub"),
        DeclareLaunchArgument(
            "reset_pose_service_topic",
            default_value="/simulation/reset_pose",
            description=(
                "std_srvs/Trigger endpoint that restores start_x/y/z/yaw; "
                "set empty to disable."
            ),
        ),
        DeclareLaunchArgument("start_x", default_value="0.0"),
        DeclareLaunchArgument("start_y", default_value="0.0"),
        DeclareLaunchArgument("start_z", default_value="0.18"),
        DeclareLaunchArgument("start_yaw", default_value="0.0"),
        DeclareLaunchArgument(
            "launch_mujoco_rviz",
            default_value="false",
            description="Whether to start the lightweight MuJoCo-only RViz view.",
        ),
        DeclareLaunchArgument("rviz_delay_sec", default_value="4.0"),
        DeclareLaunchArgument(
            "rviz_config_file",
            default_value=PathJoinSubstitution([
                FindPackageShare("ats_mujoco_sim"),
                "rviz",
                "mujoco_sim_observe.rviz",
            ]),
        ),
        Node(
            package="ats_mujoco_sim",
            executable="ats_mujoco_sim",
            name="ats_mujoco_sim",
            output="screen",
            parameters=[{
                "model_path": model_path,
                "scene_file": scene_file,
                "map_dir": map_dir,
                "map_manifest_path": map_manifest_path,
                "map_ready_file": map_ready_file,
                "map_ready_token": map_ready_token,
                "map_wait_timeout_sec": map_wait_timeout_sec,
                "sim_rate_hz": sim_rate_hz,
                "feedback_rate_hz": feedback_rate_hz,
                "truth_rate_hz": truth_rate_hz,
                "command_timeout": command_timeout,
                "freeze_motion": freeze_motion,
                "wheel_radius": wheel_radius,
                "max_wheel_speed": max_wheel_speed,
                "max_wheel_acceleration": max_wheel_acceleration,
                "max_steer_rate": max_steer_rate,
                "emergency_stop_topic": emergency_stop_topic,
                "swerve_telemetry_topic": swerve_telemetry_topic,
                "show_viewer": show_viewer,
                "use_viewer": use_viewer,
                "viewer_rate_hz": viewer_rate_hz,
                "enable_lidar": enable_lidar,
                "lidar_backend": lidar_backend,
                "lidar_model": lidar_model,
                "lidar_downsample": lidar_downsample,
                "lidar_rate_hz": lidar_rate_hz,
                "lidar_rate_clock": lidar_rate_clock,
                "lidar_state_rate_hz": lidar_state_rate_hz,
                "lidar_topic": lidar_topic,
                "lidar_frame_id": lidar_frame_id,
                "registered_scan_topic": registered_scan_topic,
                "registered_scan_frame_id": registered_scan_frame_id,
                "enable_tof": enable_tof,
                "tof_backend": tof_backend,
                "tof_range": tof_range,
                "tof_min_range": tof_min_range,
                "tof_rate_hz": tof_rate_hz,
                "tof_width": tof_width,
                "tof_height": tof_height,
                "tof_horizontal_fov_deg": tof_horizontal_fov_deg,
                "tof_vertical_fov_deg": tof_vertical_fov_deg,
                "tof_footprint_length": tof_footprint_length,
                "tof_footprint_width": tof_footprint_width,
                "tof_footprint_expand": tof_footprint_expand,
                "tof_footprint_resolution": tof_footprint_resolution,
                "tof_footprint_z_min": tof_footprint_z_min,
                "tof_footprint_z_max": tof_footprint_z_max,
                "merged_tof_topic": merged_tof_topic,
                "left_tof_topic": left_tof_topic,
                "right_tof_topic": right_tof_topic,
                "odom_topic": odom_topic,
                "publish_map_to_odom_tf": publish_map_to_odom_tf,
                "lidar_odometry_topic": lidar_odometry_topic,
                "robot_base_frame_id": robot_base_frame_id,
                "pose_cmd_topic": pose_cmd_topic,
                "reset_pose_service_topic": reset_pose_service_topic,
                "start_x": start_x,
                "start_y": start_y,
                "start_z": start_z,
                "start_yaw": start_yaw,
            }],
        ),
        TimerAction(
            period=rviz_delay_sec,
            actions=[
                Node(
                    condition=IfCondition(launch_mujoco_rviz),
                    package="rviz2",
                    executable="rviz2",
                    name="ats_mujoco_sim_rviz2",
                    output="screen",
                    arguments=["-d", rviz_config_file],
                ),
            ],
        ),
    ])
