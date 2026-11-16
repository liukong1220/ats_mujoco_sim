"""Regression: changing the wheel limit must also update MuJoCo ctrlrange."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from ats_mujoco_sim.model_patch import configure_wheel_speed_limit


def test_five_mps_command_survives_the_old_xml_motor_range():
    path = Path(__file__).resolve().parents[1] / "models/rmuc_2025_swerve.xml"
    model = mujoco.MjModel.from_xml_path(str(path))
    names = [model.actuator(i).name for i in range(model.nu) if "wheel" in model.actuator(i).name]
    assert len(names) == 4
    forces = np.array(model.actuator_forcerange, copy=True)
    rate = configure_wheel_speed_limit(model, names, 5.0, 0.0425)
    assert rate == pytest.approx(117.6470588235)
    for name in names:
        index = model.actuator(name).id
        np.testing.assert_allclose(model.actuator_ctrlrange[index], [-rate, rate])
        assert model.actuator_ctrllimited[index]
    # Raising the permitted speed does not fabricate stronger drive torque.
    np.testing.assert_array_equal(model.actuator_forcerange, forces)


@pytest.mark.parametrize("speed,radius", [(float('nan'), .0425), (5, 0), (-1, .0425)])
def test_invalid_speed_or_radius_is_rejected(speed, radius):
    with pytest.raises(ValueError):
        configure_wheel_speed_limit(None, [], speed, radius)
