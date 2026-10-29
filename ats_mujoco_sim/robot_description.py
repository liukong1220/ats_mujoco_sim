"""Vehicle model for the MuJoCo RViz: robot_state_publisher on the shared sentry description.

The description (ats_robot_description/urdf/ats_sentry.urdf.xacro) has the navigation base frame
`gimbal_yaw_odom` as its root link and every other part as a `part_*` child (RViz colours one
material per link).  robot_state_publisher therefore publishes only identity /tf_static edges
gimbal_yaw_odom -> part_*, and cannot become a second owner of the edges the simulator already
publishes: odom -> gimbal_yaw_odom, gimbal_yaw_odom -> base_footprint and
gimbal_yaw_odom -> front_mid360 (AGENTS.md 2.3).

The Mid360 box is placed with the MountPose that also feeds the simulated static transform and
Point-LIO's gravity vector, and the frame height is the simulator's `base_footprint_height`, so the
drawn vehicle follows the sim parameters instead of repeating their numbers.
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from ats_mujoco_sim.lio_launch import mount_pose_from_configurations

XACRO_RELATIVE_PATH = Path("urdf") / "ats_sentry.urdf.xacro"


def description_mappings(lc):
    """xacro arguments for the resolved launch configurations ``lc``."""
    mount = mount_pose_from_configurations(lc)
    return {
        "frame_height": f"{float(lc['base_footprint_height']):.4f}",
        "lidar_xyz": " ".join(f"{v:.6f}" for v in mount.position),
        "lidar_rpy": " ".join(f"{v:.6f}" for v in mount.rpy()),
    }


def description_xml(lc, xacro_path=None):
    import xacro  # imported late: only the launch process needs it

    path = Path(xacro_path) if xacro_path else (
        Path(get_package_share_directory("ats_robot_description")) / XACRO_RELATIVE_PATH)
    return xacro.process_file(str(path), mappings=description_mappings(lc)).toxml()


def robot_state_publisher_nodes(lc):
    """The publisher for the RViz RobotModel; nothing unless RViz is requested."""
    if str(lc.get("use_rviz", "false")).lower() not in ("true", "1"):
        return []
    try:
        description = description_xml(lc)
    except Exception as error:      # stale install without ats_robot_description, xacro missing, ...
        # The vehicle shape is a convenience for the person watching RViz; it must not abort the stack.
        print(f"[ats_mujoco_sim] WARNING: no vehicle model in RViz, robot_description was not built: {error}")
        return []
    return [Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{
            "robot_description": description,
            "use_sim_time": lc.get("localization_source") == "lio",
        }],
    )]
