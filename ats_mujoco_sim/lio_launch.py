"""Builds the point_lio -> loam_interface -> sensor_scan_generation chain for LIO mode.

Imported by the MuJoCo launch files so the chain is defined once.  Point-LIO
takes every parameter from the real-vehicle node_params.yaml except
mapping.gravity, which depends on how the simulated sensor is mounted and is
derived here from the same mount that patches the MuJoCo model.
"""

from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

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
            }],
        ),
        # loam_interface 与 sensor_scan_generation 放进同一个容器并开 intra-process，
        # registered_scan / lidar_odometry 不再经过 DDS 序列化。
        # 仿真里 base_frame 为空，sensor_scan_generation 缓存外参后不再阻塞，单线程 component_container 够用。
        # 不用 component_container_mt：开 intra-process 会丢订阅消息；不用 isolated：收不到 SIGINT 不退出。
        # LoadComposableNodes 只在容器启动时执行一次，容器崩溃后不会自动重新加载这两个组件；
        # 实车因此仍是两个独立进程（见 ats_nav_bringup navigation_launch.py 的注释）。
        ComposableNodeContainer(
            package="rclcpp_components",
            executable="component_container",
            name="lidar_odom_container",
            namespace="",
            output="screen",
            composable_node_descriptions=[
                ComposableNode(
                    package="loam_interface",
                    plugin="loam_interface::LoamInterfaceNode",
                    name="loam_interface",
                    parameters=[params_file, {
                        "use_sim_time": True,
                        # The simulated chassis frame sits base_footprint_height above ground.
                        "base_frame_height": float(lc["base_footprint_height"]),
                    }],
                    extra_arguments=[{"use_intra_process_comms": True}],
                ),
                ComposableNode(
                    package="sensor_scan_generation",
                    plugin="sensor_scan_generation::SensorScanGenerationNode",
                    name="sensor_scan_generation",
                    parameters=[params_file, {
                        "use_sim_time": True,
                        "lidar_frame": "front_mid360",
                        "robot_base_frame": "gimbal_yaw_odom",
                        # The simulator already publishes gimbal_yaw_odom->base_footprint
                        # statically; an empty base_frame keeps this node the sole
                        # odom->gimbal_yaw_odom publisher (no second parent).
                        "base_frame": "",
                    }],
                    extra_arguments=[{"use_intra_process_comms": True}],
                ),
            ],
        ),
    ]
