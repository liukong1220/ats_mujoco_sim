"""Builds the point_lio -> loam_interface -> sensor_scan_generation chain for LIO mode.

Imported by the MuJoCo launch files so the chain is defined once.  Point-LIO
takes every parameter from the real-vehicle node_params.yaml except
gravity/gravity_init, which depend on how the simulated sensor is mounted and
are derived here from the same mount that patches the MuJoCo model.
"""

from launch_ros.actions import Node

from ats_mujoco_sim.lidar_mount import chassis_mount
from ats_mujoco_sim.lidar_mount import legacy_gimbal_mount


def mount_pose_from_configurations(lc):
    """MountPose described by resolved launch configurations."""
    if lc["lidar_mount_preset"] == "legacy_gimbal":
        return legacy_gimbal_mount()
    return chassis_mount(
        x=float(lc["lidar_mount_x"]),
        y=float(lc["lidar_mount_y"]),
        height_above_ground=float(lc["lidar_mount_height"]),
        tilt_direction=lc["lidar_mount_tilt_direction"],
        tilt_deg=float(lc["lidar_mount_tilt_deg"]),
        body_height=float(lc["base_footprint_height"]),
    )


def lio_chain_nodes(lc):
    """Nodes of the LIO localization chain, or [] outside localization_source:=lio."""
    if lc["localization_source"] != "lio":
        return []
    gravity = [round(v, 6) for v in mount_pose_from_configurations(lc).gravity_in_sensor()]
    params_file = lc["params_file"]
    return [
        Node(
            package="ats_livox_adapter",
            executable="pointcloud_to_livox_node",
            name="pointcloud_to_livox",
            output="screen",
            parameters=[{
                "use_sim_time": True,
                "input_topic": lc["livox_points_topic"],
                "output_topic": "/livox/lidar",
                "frame_id": "front_mid360",
            }],
        ),
        Node(
            package="point_lio",
            executable="pointlio_mapping",
            name="point_lio",
            output="screen",
            parameters=[params_file, {
                "use_sim_time": True,
                "mapping.gravity": gravity,
                "mapping.gravity_init": gravity,
            }],
        ),
        Node(
            package="loam_interface",
            executable="loam_interface_node",
            name="loam_interface",
            output="screen",
            parameters=[params_file, {
                "use_sim_time": True,
                # The simulated chassis frame sits base_footprint_height above ground.
                "base_frame_height": float(lc["base_footprint_height"]),
            }],
        ),
        Node(
            package="sensor_scan_generation",
            executable="sensor_scan_generation_node",
            name="sensor_scan_generation",
            output="screen",
            parameters=[params_file, {
                "use_sim_time": True,
                "lidar_frame": "front_mid360",
                "robot_base_frame": "gimbal_yaw_odom",
                # The simulator already publishes gimbal_yaw_odom->base_footprint
                # statically; an empty base_frame keeps this node the sole
                # odom->gimbal_yaw_odom publisher (no second parent).
                "base_frame": "",
            }],
        ),
    ]
