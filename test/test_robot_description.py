"""The RViz vehicle model must agree with the real vehicle, the MuJoCo model and the TF contract."""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

from ats_mujoco_sim.launch_args import SIM_LIO_ARGUMENTS
from ats_mujoco_sim.lidar_mount import chassis_mount
from ats_mujoco_sim.robot_description import description_mappings
from ats_mujoco_sim.robot_description import description_xml

WORKSPACE = Path(__file__).resolve().parents[4]
XACRO = WORKSPACE / "src/ats_robot_description/urdf/ats_sentry.urdf.xacro"
MODEL_2027 = WORKSPACE / "src/sim/ats_mujoco_sim/models/rmuc_2027_swerve.xml"
BRINGUP_LAUNCH = WORKSPACE / "src/ats_sentry_bringup/launch/bringup.launch.py"
MPC_REALITY = WORKSPACE / "src/ats_sentry_nav/ats_swerve_mpc/config/ats_swerve_mpc_reality.yaml"


def sim_configurations(**changes):
    lc = {**SIM_LIO_ARGUMENTS, "lidar_mount_preset": "chassis", "use_rviz": "true",
          "localization_source": "lio"}
    lc.update(changes)
    return lc


def parse(lc=None):
    xml = description_xml(lc, XACRO) if lc is not None else None
    if xml is None:
        import xacro
        xml = xacro.process_file(str(XACRO)).toxml()
    return ET.fromstring(xml)


def visuals(robot):
    """name -> (xyz, rpy, shape, attributes) in the root frame (every part frame is an identity child)."""
    result = {}
    for visual in robot.iter("visual"):
        origin = visual.find("origin")
        xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]
        rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
        geometry = visual.find("geometry")[0]
        result[visual.get("name")] = (xyz, rpy, geometry.tag, geometry.attrib)
    return result


def launch_default(name):
    text = BRINGUP_LAUNCH.read_text(encoding="utf-8")
    return float(re.search(rf'DeclareLaunchArgument\("{name}",\s*default_value="([^"]+)"', text).group(1))


def test_every_part_is_its_own_link_so_rviz_colours_it():
    robot = parse(sim_configurations())
    links = robot.findall("link")
    assert all(len(link.findall("visual")) == 1 for link in links), "RViz uses one material per link"
    colours = [link.find("visual/material").get("name") for link in links]
    assert colours.count("wheel_color") == 4
    assert len(set(colours)) == 5, "chassis, upper block, front marker, wheels and Mid360 each have a colour"
    assert {m.get("name") for m in robot.findall("material")} == set(colours)


def test_description_only_adds_identity_edges_below_gimbal_yaw_odom():
    """robot_state_publisher must not become a second owner of any edge the simulator publishes."""
    robot = parse(sim_configurations())
    links = [link.get("name") for link in robot.findall("link")]
    joints = robot.findall("joint")
    assert links[0] == "gimbal_yaw_odom"
    assert len(joints) == len(links) - 1
    for joint in joints:
        assert joint.get("type") == "fixed"
        assert joint.find("parent").get("link") == "gimbal_yaw_odom"
        assert joint.find("origin") is None, "part frames coincide with the chassis frame"
    children = [joint.find("child").get("link") for joint in joints]
    assert all(name.startswith("part_") for name in children)
    assert not set(links) & {"odom", "map", "base_footprint", "base_link", "front_mid360"}


def test_body_is_581_by_571_and_242_high_standing_on_the_ground():
    lc = sim_configurations()
    parts = visuals(parse(lc))
    frame_height = float(lc["base_footprint_height"])
    chassis_xyz, _, _, chassis = parts["chassis"]
    assert (float(chassis["size"].split()[0]), float(chassis["size"].split()[1])) == (0.581, 0.571)
    top = max(xyz[2] + float(attrib["size"].split()[2]) / 2
              for xyz, _, tag, attrib in parts.values() if tag == "box" and "size" in attrib
              and not attrib["size"].startswith("0.065"))
    bottom = min(xyz[2] - (float(attrib["radius"]) if tag == "cylinder" else 0.0)
                 for xyz, _, tag, attrib in parts.values() if tag == "cylinder")
    assert bottom == pytest.approx(-frame_height, abs=1e-4), "the wheels must touch the ground"
    assert top - bottom == pytest.approx(0.242, abs=1e-4)


def test_wheels_sit_where_the_mujoco_2027_model_has_its_steer_modules():
    # The MJCF has comments a strict XML parser rejects (MuJoCo's own parser is lenient).
    text = MODEL_2027.read_text(encoding="utf-8")
    steer = {
        match.group(1): [float(v) for v in match.group(2).split()]
        for match in re.finditer(r'<body name="(\w+)_steer_link" pos="([^"]+)"', text)
    }
    assert len(steer) == 4
    parts = visuals(parse(sim_configurations()))
    wheels = {name.replace("wheel_", ""): xyz for name, (xyz, _, _, _) in parts.items() if name.startswith("wheel_")}
    for module, (x, y, _) in steer.items():
        assert wheels[module][:2] == pytest.approx([x, y], abs=1e-6), module
    # Real vehicle: the MPC parameter file of the real chassis.
    mpc = yaml.safe_load(MPC_REALITY.read_text(encoding="utf-8"))
    parameters = next(iter(mpc.values()))["ros__parameters"]
    for module, (x, y, _) in wheels.items():
        assert abs(x) == pytest.approx(parameters["wheel_base_x"], abs=1e-9), module
        assert abs(y) == pytest.approx(parameters["wheel_base_y"], abs=1e-9), module


def test_mid360_box_is_drawn_at_the_pose_of_the_simulated_static_transform():
    lc = sim_configurations()
    mount = chassis_mount(
        x=float(lc["lidar_mount_x"]), y=float(lc["lidar_mount_y"]),
        height_above_ground=float(lc["lidar_mount_height"]),
        tilt_direction=lc["lidar_mount_tilt_direction"], tilt_deg=float(lc["lidar_mount_tilt_deg"]),
        body_height=float(lc["base_footprint_height"]))
    xyz, rpy, _, _ = visuals(parse(lc))["mid360"]
    assert xyz == pytest.approx(list(mount.position), abs=1e-6)
    assert rpy == pytest.approx(list(mount.rpy()), abs=1e-6)
    assert description_mappings(lc)["frame_height"] == "0.1800"


def test_simulated_mount_matches_the_real_vehicle_static_transform():
    """Real: gimbal_yaw_odom -> front_mid360 = (0.2405, 0, 0.13), pitch 0.5236 rad (30 deg)."""
    real = chassis_mount(body_height=0.07)    # defaults: x 0.2405, 0.20 m optical-centre height, 30 deg
    assert real.position == pytest.approx(
        [launch_default("lidar_static_tf_x"), launch_default("lidar_static_tf_y"),
         launch_default("lidar_static_tf_z")], abs=1e-6)
    assert real.rpy() == pytest.approx(
        [launch_default("lidar_static_tf_roll"), launch_default("lidar_static_tf_pitch"),
         launch_default("lidar_static_tf_yaw")], abs=1e-6)
    assert float(SIM_LIO_ARGUMENTS["lidar_mount_tilt_deg"]) == 30.0


def test_xacro_defaults_are_the_real_vehicle():
    xyz, rpy, _, _ = visuals(parse())["mid360"]
    assert xyz == pytest.approx([launch_default("lidar_static_tf_x"), launch_default("lidar_static_tf_y"),
                                 launch_default("lidar_static_tf_z")], abs=1e-6)
    assert rpy[1] == pytest.approx(launch_default("lidar_static_tf_pitch"), abs=1e-6)


def test_missing_description_package_does_not_abort_the_launch(monkeypatch, capsys):
    import ats_mujoco_sim.robot_description as module

    def missing(lc, xacro_path=None):
        raise RuntimeError("package 'ats_robot_description' not found")

    monkeypatch.setattr(module, "description_xml", missing)
    assert module.robot_state_publisher_nodes(sim_configurations()) == []
    assert "no vehicle model in RViz" in capsys.readouterr().out
    # without RViz the publisher is not even attempted
    assert module.robot_state_publisher_nodes(sim_configurations(use_rviz="false")) == []
