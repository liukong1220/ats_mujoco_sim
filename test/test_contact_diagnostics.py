"""Real MuJoCo contacts retain geom identity/full attitude without unbounded logs."""
import json
from types import SimpleNamespace

import mujoco

from ats_mujoco_sim.sim_node import SwerveMujocoSim


def test_contact_trace_is_bounded_and_keeps_simulation_contact_counts():
    model = mujoco.MjModel.from_xml_string("""
      <mujoco><worldbody>
        <body name="robot" pos="0 0 0.3"><freejoint/>
          <geom name="base" type="box" size=".3 .3 .1"/>
        </body>
        <geom name="wall" type="box" pos=".3 0 .3" size=".1 1 1"/>
      </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.ncon > 0
    lines = []
    sim = SimpleNamespace(
        model=model, data=data, robot_geom_ids={model.geom("base").id},
        wheel_geom_ids=set(), ground_geom_ids=set(), base_body_id=model.body("robot").id,
        contact_violation_count=0, contact_diagnostic_count=0, max_contact_force=0.0,
        get_logger=lambda: SimpleNamespace(warning=lines.append))
    for _ in range(50):
        SwerveMujocoSim._evaluate_contacts(sim)
    assert sim.contact_violation_count == 50 * data.ncon
    assert len(lines) == sim.contact_diagnostic_count == 32
    record = json.loads(lines[0].split("】", 1)[1])
    assert set(record["geom_names"]) == {"base", "wall"}
    assert record["distance_m"] < 0.0
    assert len(record["position"]) == len(record["base_position"]) == 3
    assert record["base_quaternion_wxyz"] == [1.0, 0.0, 0.0, 0.0]
