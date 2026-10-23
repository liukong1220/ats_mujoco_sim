#!/usr/bin/env python3
"""MuJoCo physics + the LIO localization chain only (no planning stack).

For S0 localization drift/ATE runs: the scenario driver publishes
/motion_control itself, so no planner or MPC may command the chassis.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import OpaqueFunction
from launch.actions import TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

from ats_mujoco_sim.launch_args import SIM_LIO_ARGUMENTS
from ats_mujoco_sim.lio_launch import lio_chain_nodes

ARGUMENTS = {
    **SIM_LIO_ARGUMENTS,
    "localization_source": "lio",
    "lidar_mount_preset": "chassis",
}


def _chain(context, *args, **kwargs):
    return lio_chain_nodes(context.launch_configurations)


def generate_launch_description() -> LaunchDescription:
    share = FindPackageShare("ats_mujoco_sim")
    model = PathJoinSubstitution([share, "models", LaunchConfiguration("model_file")])
    return LaunchDescription([
        *[DeclareLaunchArgument(n, default_value=v) for n, v in ARGUMENTS.items()],
        DeclareLaunchArgument("model_file", default_value="rmuc_2027_swerve.xml"),
        DeclareLaunchArgument("start_x", default_value="4.0"),
        DeclareLaunchArgument("start_y", default_value="5.5"),
        DeclareLaunchArgument("start_z", default_value="0.381"),
        DeclareLaunchArgument("start_yaw", default_value="0.0"),
        DeclareLaunchArgument("params_file", default_value=PathJoinSubstitution([
            FindPackageShare("ats_sentry_bringup"), "params", "node_params.yaml"])),
        DeclareLaunchArgument("sim_rate_hz", default_value="400.0"),
        DeclareLaunchArgument("lidar_rate_hz", default_value="20.0"),
        DeclareLaunchArgument("lidar_downsample", default_value="1"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([share, "launch", "ats_mujoco_sim.launch.py"])),
            launch_arguments={
                "model_path": model,
                "scene_file": model,
                "enable_static_walls": "false",
                "use_viewer": "false",
                "show_viewer": "false",
                "enable_lidar": "true",
                "enable_tof": "false",
                "lidar_backend": "cpu",
                "lidar_rate_hz": LaunchConfiguration("lidar_rate_hz"),
                "lidar_downsample": LaunchConfiguration("lidar_downsample"),
                "sim_rate_hz": LaunchConfiguration("sim_rate_hz"),
                "start_x": LaunchConfiguration("start_x"),
                "start_y": LaunchConfiguration("start_y"),
                "start_z": LaunchConfiguration("start_z"),
                "start_yaw": LaunchConfiguration("start_yaw"),
                "publish_map_to_odom_tf": "false",
                **{name: LaunchConfiguration(name) for name in ARGUMENTS},
            }.items(),
        ),
        TimerAction(period=1.0, actions=[OpaqueFunction(function=_chain)]),
    ])
