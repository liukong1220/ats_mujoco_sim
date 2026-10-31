"""Twist bridge planar limit (report 5.1.10 item 4): vector scaling like the real exit limit.

Imports only ``ats_mujoco_sim.kinematics`` (no ROS messages), like test_freeze_motion_parameter.
"""

from math import atan2, hypot, isclose

import pytest

from ats_mujoco_sim.kinematics import scale_planar_speed


def test_command_inside_the_limit_passes_unchanged() -> None:
    assert scale_planar_speed(0.9, -0.6, 1.5) == (0.9, -0.6)


@pytest.mark.parametrize("vx, vy", [(1.4, 0.9), (-1.2, 1.3), (0.3, -1.9), (2.0, 0.0)])
def test_command_above_the_limit_keeps_direction_and_hits_the_limit(vx, vy) -> None:
    sx, sy = scale_planar_speed(vx, vy, 1.5)
    assert isclose(hypot(sx, sy), 1.5, rel_tol=1e-12)
    assert isclose(atan2(sy, sx), atan2(vy, vx), abs_tol=1e-12)


def test_per_axis_clamp_would_bend_the_same_command() -> None:
    # The old bridge clamped x and y separately at 1.0: (1.4, 0.9) became (1.0, 0.9), 9.3 deg off.
    vx, vy = 1.4, 0.9
    old = (min(vx, 1.0), min(vy, 1.0))
    bend_deg = abs(atan2(old[1], old[0]) - atan2(vy, vx)) * 180.0 / 3.141592653589793
    assert bend_deg > 9.0
    sx, sy = scale_planar_speed(vx, vy, 1.0)
    assert isclose(atan2(sy, sx), atan2(vy, vx), abs_tol=1e-12)


def test_non_positive_limit_disables_the_cap() -> None:
    assert scale_planar_speed(3.0, 4.0, 0.0) == (3.0, 4.0)
