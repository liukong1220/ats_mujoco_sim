from builtin_interfaces.msg import Time

from ats_mujoco_sim.robot_footprint_marker import build_markers, rectangle_points


def test_rectangle_is_closed_and_heading_is_plus_x():
    points = rectangle_points(0.581, 0.571)
    assert len(points) == 5
    assert (points[0].x, points[0].y) == (points[-1].x, points[-1].y)
    assert max(p.x for p in points) == 0.5 * 0.581
    assert max(p.y for p in points) == 0.5 * 0.571


def test_margin_grows_every_side_and_markers_use_chassis_frame():
    markers = build_markers("gimbal_yaw_odom", 0.581, 0.571, 0.02, Time()).markers
    by_ns = {m.ns: m for m in markers}
    assert set(by_ns) == {"footprint", "footprint_margin", "front_face", "heading"}
    assert all(m.header.frame_id == "gimbal_yaw_odom" for m in markers)
    inner = max(p.x for p in by_ns["footprint"].points)
    outer = max(p.x for p in by_ns["footprint_margin"].points)
    assert abs(outer - inner - 0.02) < 1e-9
    assert by_ns["heading"].points[1].x > by_ns["front_face"].points[0].x
