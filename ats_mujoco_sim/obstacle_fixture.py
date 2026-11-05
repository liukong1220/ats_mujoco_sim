"""Opt-in physical obstacle fixture, isolated from normal scenario bodies."""
from collections.abc import Sequence
import math
import time

import mujoco
import numpy as np


class ObstacleFixture:
    def __init__(self, model, data, enabled=False, body_name="s3_test_obstacle"):
        self.model, self.data, self.enabled = model, data, bool(enabled)
        self.sequence = 0
        if body_name != "s3_test_obstacle":
            raise ValueError("fixture may only move s3_test_obstacle")
        if not self.enabled:
            return
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body < 0 or model.body_jntnum[body] != 1:
            raise ValueError("fixture needs the explicit s3_test_obstacle body")
        joint = model.body_jntadr[body]
        if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("s3_test_obstacle requires a free joint")
        self.qpos = int(model.jnt_qposadr[joint])
        self.dof = int(model.jnt_dofadr[joint])

    def validate_position(self, position):
        if not self.enabled:
            raise ValueError("test obstacle fixture is disabled at startup")
        if (not isinstance(position, (Sequence, np.ndarray)) or len(position) != 3 or
                not all(isinstance(x, (float, int)) and math.isfinite(x) and abs(x) <= 100
                        for x in position)):
            raise ValueError("test_obstacle_position needs three finite coordinates within 100 m")

    def set_position(self, position):
        self.validate_position(position)
        applied = time.monotonic_ns()
        self.data.qpos[self.qpos:self.qpos + 3] = position
        self.data.qpos[self.qpos + 3:self.qpos + 7] = [1., 0., 0., 0.]
        self.data.qvel[self.dof:self.dof + 6] = 0.
        mujoco.mj_forward(self.model, self.data)
        self.sequence += 1
        return {"fixture_sequence": self.sequence, "steady_ns": applied,
                "sim_s": float(self.data.time), "position": list(position)}
