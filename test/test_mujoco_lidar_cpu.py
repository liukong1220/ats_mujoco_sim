import numpy as np

import mujoco

from mujoco_lidar.core_cpu.mjlidar_cpu import MjLidarCPU


def test_cpu_lidar_uses_python_multiray_signature_without_normal_buffer(
    monkeypatch,
) -> None:
    captured = {}

    def record_multiray(*args) -> None:
        captured["args"] = args
        args[7][...] = -1

    monkeypatch.setattr(mujoco, "mj_multiRay", record_multiray)
    lidar = MjLidarCPU(
        object(),
        cutoff_dist=12.0,
        geomgroup=np.array([1, 1, 1, 0, 1, 1], dtype=np.uint8),
        bodyexclude=7,
    )
    lidar.update(object())
    lidar.trace_rays(
        np.eye(4, dtype=np.float32),
        np.array([0.0, np.pi / 2.0], dtype=np.float32),
        np.zeros(2, dtype=np.float32),
    )

    args = captured["args"]
    assert len(args) == 11
    assert args[3].dtype == np.float64
    assert args[3].shape == (6,)
    assert args[8].dtype == np.float64
    assert args[8].shape == (2,)
    assert args[9] == 2
    assert args[10] == 12.0
    assert np.array_equal(lidar.get_distances(), np.zeros(2))
