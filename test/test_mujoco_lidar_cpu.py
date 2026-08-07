import numpy as np

import mujoco

from mujoco_lidar.core_cpu.mjlidar_cpu import MjLidarCPU


def test_cpu_lidar_uses_python_multiray_signature_with_optional_normal_argument(
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
    assert len(args) == 12
    assert args[3].dtype == np.float64
    assert args[3].shape == (6,)
    assert args[8].dtype == np.float64
    assert args[8].shape == (2,)
    assert args[9] is None
    assert args[10] == 2
    assert args[11] == 12.0
    assert np.array_equal(lidar.get_distances(), np.zeros(2))


def test_cpu_lidar_calls_current_mujoco_binding_without_child_process_crash() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <geom type=\"plane\" size=\"5 5 0.1\"/>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    lidar = MjLidarCPU(model, cutoff_dist=12.0)
    lidar.update(data)
    pose = np.eye(4, dtype=np.float64)
    pose[2, 3] = 1.0

    lidar.trace_rays(
        pose,
        np.array([0.0], dtype=np.float64),
        np.array([-np.pi / 2.0], dtype=np.float64),
    )

    distances = lidar.get_distances()
    hit_points = lidar.get_hit_points()
    assert distances is not None
    assert hit_points is not None
    assert distances.shape == (1,)
    assert hit_points.shape == (1, 3)
    assert np.isfinite(distances[0])
    assert np.isclose(distances[0], 1.0, atol=1e-6)
