"""The latency fixture changes physical ray tracing only after explicit opt-in."""
import mujoco
import numpy as np
import pytest

from ats_mujoco_sim.obstacle_fixture import ObstacleFixture


def scene():
    model = mujoco.MjModel.from_xml_string("""
      <mujoco><worldbody>
        <geom type="box" pos="4 0 0.5" size="0.1 1 0.5"/>
        <body name="s3_test_obstacle" pos="2 0 -10">
          <freejoint/><geom type="box" size="0.3 0.3 0.3" mass="1"/>
        </body>
      </worldbody></mujoco>""")
    return model, mujoco.MjData(model)


def distance(model, data):
    hit = np.array([-1], dtype=np.int32)
    return mujoco.mj_ray(model, data, np.array([0., 0., 0.2]), np.array([1., 0., 0.]),
                         None, 1, -1, hit)


def test_fixture_is_disabled_by_default_and_cannot_target_the_robot():
    model, data = scene()
    fixture = ObstacleFixture(model, data)
    before = data.qpos.copy()
    with pytest.raises(ValueError, match="disabled"):
        fixture.set_position([2., 0., 0.3])
    np.testing.assert_array_equal(data.qpos, before)
    with pytest.raises(ValueError, match="s3_test_obstacle"):
        ObstacleFixture(model, data, enabled=True, body_name="base_link")


def test_physical_obstacle_appears_and_disappears_in_the_actual_ray_caster():
    model, data = scene()
    fixture = ObstacleFixture(model, data, enabled=True)
    mujoco.mj_forward(model, data)
    assert distance(model, data) == pytest.approx(3.9)
    marker = fixture.set_position([2., 0., 0.3])
    assert marker["steady_ns"] > 0 and marker["sim_s"] == data.time
    assert distance(model, data) == pytest.approx(1.7)
    fixture.set_position([2., 0., -10.])
    assert distance(model, data) == pytest.approx(3.9)


def test_invalid_position_never_partially_moves_a_body():
    model, data = scene()
    fixture = ObstacleFixture(model, data, enabled=True)
    before = data.qpos.copy()
    for position in ([1., 2.], [float("nan"), 0., 0.], [101., 0., 0.]):
        with pytest.raises(ValueError):
            fixture.set_position(position)
        np.testing.assert_array_equal(data.qpos, before)


def test_ros_double_array_parameter_is_accepted_without_changing_its_values():
    from array import array
    model, data = scene()
    fixture = ObstacleFixture(model, data, enabled=True)
    marker = fixture.set_position(array("d", [2., 0., 0.3]))
    assert marker["position"] == [2., 0., 0.3]
    assert distance(model, data) == pytest.approx(1.7)
