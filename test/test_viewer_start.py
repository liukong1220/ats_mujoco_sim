"""The MuJoCo window must never block the simulation from starting (locked Wayland desktop)
and must never touch the simulation's own MjData (launch_passive runs mj_forward on its caller's
thread, which would race mj_step)."""

import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import mujoco
import mujoco.viewer

from ats_mujoco_sim import sim_node
from ats_mujoco_sim.sim_node import SwerveMujocoSim, _refresh_viewer_data

BOX = ("<mujoco model='probe'><worldbody><geom type='plane' size='2 2 .1'/>"
       "<body pos='0 0 .5'><freejoint/><geom type='box' size='.1 .1 .1'/></body></worldbody></mujoco>")


class _Logger:
    def __init__(self):
        self.lines = []

    def info(self, text):
        self.lines.append(("info", text))

    def error(self, text):
        self.lines.append(("error", text))


class _Viewer:
    def __init__(self):
        self.opt = SimpleNamespace(geomgroup=[0, 0, 0, 0, 0, 0])


def _simulator():
    logger = _Logger()
    model = mujoco.MjModel.from_xml_string(BOX)
    data = mujoco.MjData(model)
    simulator = SimpleNamespace(
        model=model, data=data, viewer="unset", viewer_data="unset", get_logger=lambda: logger)
    return simulator, logger


def test_start_viewer_returns_while_the_compositor_has_not_mapped_the_window(monkeypatch):
    release = threading.Event()
    handle = _Viewer()
    received = {}

    def blocking_launch(model, data, **kwargs):
        received["data"] = data
        release.wait(10.0)
        return handle

    monkeypatch.setattr(mujoco.viewer, "launch_passive", blocking_launch)
    simulator, logger = _simulator()

    started = time.monotonic()
    SwerveMujocoSim._start_viewer(simulator)
    assert time.monotonic() - started < 1.0, "start-up waited for the window"
    assert simulator.viewer is None            # the simulation loop runs headless meanwhile

    release.set()                              # desktop unlocked: the window exists now
    deadline = time.monotonic() + 5.0
    while simulator.viewer is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert simulator.viewer is handle
    assert handle.opt.geomgroup[3] == 1        # debug geom group shown, as before
    assert ("info", "【MuJoCo查看器】已启动") in logger.lines
    # The window gets a private copy of the state, never the data the simulation steps.
    assert received["data"] is not simulator.data
    assert simulator.viewer_data is received["data"]


def test_refresh_gives_the_window_a_consistent_private_copy_of_the_state():
    """The call the physics loop makes every viewer frame, against the real MuJoCo bindings."""
    model = mujoco.MjModel.from_xml_string(BOX)
    data = mujoco.MjData(model)
    shadow = mujoco.MjData(model)
    mujoco.mj_resetData(model, shadow)
    for _ in range(120):                       # let the box fall and touch the plane
        mujoco.mj_step(model, data)

    _refresh_viewer_data(model, data, shadow)

    assert shadow.time == data.time
    assert (shadow.qpos == data.qpos).all() and (shadow.qvel == data.qvel).all()
    body = 1                                    # the falling box (body 0 is the world)
    # derived poses are recomputed on the copy: the renderer reads them, and a reset copy has zeros
    assert abs(shadow.xpos[body][2] - data.qpos[2]) < 1e-9
    held = shadow.qpos.copy()
    for _ in range(50):                         # the simulation keeps stepping its own data
        mujoco.mj_step(model, data)
    assert (shadow.qpos == held).all()          # the window's copy is not touched by mj_step


def test_every_mujoco_function_the_simulator_calls_exists_in_the_installed_bindings():
    """Guards the failure the first viewer fix had: a function that this MuJoCo version lacks was only
    reached once a window existed, so no unit test saw it and every viewer run lost its /clock."""
    source = Path(sim_node.__file__).read_text(encoding="utf-8")
    names = sorted(set(re.findall(r"\bmujoco\.(mj[A-Za-z0-9_]*|mju_[A-Za-z0-9_]+)", source)))
    assert names, "pattern found no mujoco calls"
    missing = [name for name in names if not hasattr(mujoco, name)]
    assert not missing, f"mujoco {mujoco.__version__} has no {missing}"


def test_failed_viewer_start_is_logged_and_leaves_the_simulation_headless(monkeypatch):
    def broken(model, data, **kwargs):
        raise RuntimeError("no display")

    monkeypatch.setattr(mujoco.viewer, "launch_passive", broken)
    simulator, logger = _simulator()
    SwerveMujocoSim._start_viewer(simulator)
    simulator._viewer_open_thread.join(5.0)
    assert simulator.viewer is None
    assert any(level == "error" and "no display" in text for level, text in logger.lines)
