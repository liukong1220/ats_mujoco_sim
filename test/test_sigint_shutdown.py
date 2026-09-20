"""Exercise installed ROS executables, not a launch wrapper's exit status.

Run after building and sourcing the workspace:
  python3 -m pytest src/sim/ats_mujoco_sim/test/test_sigint_shutdown.py -v
"""

import os
from pathlib import Path
import signal
import subprocess
import time

from ament_index_python.packages import get_package_prefix
import pytest


PACKAGE_DIR = Path(__file__).resolve().parents[1]
CASES = (
    (
        "static_map_publisher",
        ["-p", f"map_yaml_file:={PACKAGE_DIR / 'maps/rmuc_2026.yaml'}"],
        ("Published durable P3 static map",),
    ),
    ("twist_to_motion_ctrl", [], ("Bridging Twist",)),
    (
        "ats_mujoco_sim",
        [
            "-p", f"model_path:={PACKAGE_DIR / 'models/swerve_chassis.xml'}",
            "-p", "use_viewer:=false",
            "-p", "lidar_backend:=cpu",
            "-p", "lidar_downsample:=100",
            "-p", "lidar_rate_hz:=2.0",
            "-p", "tof_rate_hz:=2.0",
        ],
        (
            "Loaded MuJoCo swerve model",
            "LiDAR process started:",
            "Side ToF process started:",
        ),
    ),
)


@pytest.mark.parametrize("executable,parameters,ready_markers", CASES)
@pytest.mark.parametrize("signal_group", [False, True], ids=["owner", "group"])
def test_sigint_exits_cleanly(executable, parameters, ready_markers, signal_group, tmp_path):
    """Normal interrupt must reap sensor workers and return zero without traceback."""
    binary = Path(get_package_prefix("ats_mujoco_sim")) / "lib/ats_mujoco_sim" / executable
    assert binary.is_file(), f"Build ats_mujoco_sim first: {binary}"
    env = os.environ.copy()
    env["ROS_LOCALHOST_ONLY"] = "1"
    env["ROS_DOMAIN_ID"] = os.environ.get("ATS_SHUTDOWN_TEST_DOMAIN_ID", "197")
    env["PYTHONUNBUFFERED"] = "1"
    env["ROS_LOG_DIR"] = str(tmp_path / "ros_logs")
    log_path = tmp_path / "node.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [str(binary), "--ros-args", *parameters],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 30.0
            while True:
                output = log_path.read_text()
                assert process.poll() is None, f"Exited before SIGINT: {process.returncode}\n{output}"
                if all(marker in output for marker in ready_markers):
                    break
                assert time.monotonic() < deadline, f"Readiness timeout\n{output}"
                time.sleep(0.05)
            if signal_group:
                os.killpg(process.pid, signal.SIGINT)
            else:
                process.send_signal(signal.SIGINT)
            try:
                returncode = process.wait(timeout=12.0)
            except subprocess.TimeoutExpired:
                pytest.fail(f"SIGINT teardown exceeded 12 seconds\n{log_path.read_text()}")
            output = log_path.read_text()
            assert returncode == 0, f"SIGINT returned {returncode}\n{output}"
            assert "Traceback (most recent call last)" not in output, output
        finally:
            # Clean up the entire test-owned group, including any orphan workers.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5.0)
