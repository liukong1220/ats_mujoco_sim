"""Check launch defaults and forwarding without starting ROS nodes or a viewer."""
from pathlib import Path
import runpy

from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, TimerAction
from launch.utilities import perform_substitutions
from launch_ros.actions import Node
from launch_ros.utilities import evaluate_parameters


LAUNCH = Path(__file__).resolve().parents[1] / "launch"


def description(name):
    return runpy.run_path(str(LAUNCH / name))["generate_launch_description"]()


def defaults(name, overrides=None):
    context = LaunchContext()
    context.launch_configurations.update(overrides or {})
    for entity in description(name).entities:
        if isinstance(entity, DeclareLaunchArgument):
            entity.execute(context)
    return context


def test_2026_alias_selects_2027_and_base_start():
    name = "rmuc_2026_mujoco.launch.py"
    context = defaults(name)
    include = next(e for e in description(name).entities if isinstance(e, IncludeLaunchDescription))
    include.launch_description_source.get_launch_description(context)
    assert include.launch_description_source.location.endswith(
        "/launch/rmuc_2027_mujoco.launch.py")
    assert context.launch_configurations["start_x"] == "3.37"
    assert context.launch_configurations["start_y"] == "9.03"
    assert context.launch_configurations["start_z"] == "0.465"
    assert context.launch_configurations["start_yaw"] == "0.0"
    context = defaults(name, {"start_yaw": "0.7", "start_x": "4.5"})
    assert context.launch_configurations["start_yaw"] == "0.7"
    assert context.launch_configurations["start_x"] == "4.5"


def test_2027_selects_consistent_assets_and_spin_cap():
    name = "rmuc_2027_mujoco.launch.py"
    context = defaults(name)
    values = context.launch_configurations
    assert values["model_file"] == "rmuc_2027_swerve.xml"
    assert values["map_yaml_file"].endswith("/map/rmuc_2027.yaml")
    assert values["prior_pcd_file"].endswith("/pcd/rmuc_2027.pcd")
    assert values["scene_navigation_profile"].endswith("/config/rmuc_2027_navigation.yaml")
    assert values["lidar_mount_preset"] == "chassis"
    assert values["max_angular_z"] == "3.0"
    include = next(e for e in description(name).entities if isinstance(e, IncludeLaunchDescription))
    forwarded = dict(include.launch_arguments)
    assert perform_substitutions(context, [forwarded["max_angular_z"]]) == "3.0"
    override = defaults(name, {"max_angular_z": "1.0"})
    assert perform_substitutions(override, [forwarded["max_angular_z"]]) == "1.0"


def test_common_launch_exposes_spin_switch_without_changing_2025_cap():
    name = "rmuc_2025_mujoco.launch.py"
    context = defaults(name)
    values = context.launch_configurations
    assert values["spin_enabled"] == "true"
    assert values["spin_rate"] == "3.0"
    assert values["max_angular_z"] == "2.0"
    override = defaults(name, {"spin_enabled": "false", "spin_rate": "1.5"})
    assert override.launch_configurations["spin_enabled"] == "false"
    assert override.launch_configurations["spin_rate"] == "1.5"
    # ParameterValue overrides must reach MINCO, not only be advertised as arguments.
    def descendants(entity):
        yield entity
        children = entity.entities if isinstance(entity, LaunchDescription) else (
            entity.actions if isinstance(entity, TimerAction) else (
                entity.get_sub_entities() if isinstance(entity, GroupAction) else []))
        for child in children:
            yield from descendants(child)

    minco = next(e for e in descendants(description(name)) if isinstance(e, Node)
                 and e.node_package == "minco_planner")
    # Humble exposes normalized Node parameters only through this internal member.
    parameters = evaluate_parameters(override, minco._Node__parameters)
    effective = {}
    for value in parameters:
        if isinstance(value, dict):
            effective.update(value)
    assert effective["spin_enabled"] is False
    assert effective["spin_rate"] == 1.5
