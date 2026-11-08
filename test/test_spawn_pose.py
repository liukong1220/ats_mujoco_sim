import math

import numpy as np
import pytest

from ats_mujoco_sim.spawn_pose import quaternion_from_rpy


@pytest.mark.parametrize("yaw", [0.0, 0.7, -math.pi / 2, math.pi])
def test_zero_tilt_preserves_existing_yaw(yaw):
    assert quaternion_from_rpy(0.0, 0.0, yaw) == pytest.approx(
        [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def test_slope_attitude_is_normalized_and_uses_body_axes():
    q = quaternion_from_rpy(math.atan(0.026625), 0.0, 0.0)
    assert np.linalg.norm(q) == pytest.approx(1.0)
    assert q == pytest.approx([0.999911431856, 0.013308961010, 0.0, 0.0])


@pytest.mark.parametrize("angle", [math.nan, math.inf, -math.inf])
def test_invalid_attitude_rejected(angle):
    with pytest.raises(ValueError):
        quaternion_from_rpy(angle, 0.0, 0.0)
