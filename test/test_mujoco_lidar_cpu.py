import numpy as np
import pytest

import mujoco

from mujoco_lidar.core_cpu.mjlidar_cpu import MjLidarCPU, StaticFieldMesh


def test_field_index_matches_native_ranges_with_moving_fixture():
    pytest.importorskip("embreex.rtcore_scene")
    model = mujoco.MjModel.from_xml_string("""
        <mujoco>
          <asset><mesh name="field" vertex="0 -2 -2  0 2 -2  0 0 2  1 0 0"
                       face="0 2 1  0 1 3  1 2 3  2 0 3"/></asset>
          <worldbody>
            <geom name="rmuc_2027_field_visual" type="mesh" mesh="field" pos="5 0 0" group="2"/>
            <body mocap="true"><geom type="sphere" size="0.5"/></body>
          </worldbody>
        </mujoco>""")
    data = mujoco.MjData(model)
    groups = np.ones(6, dtype=np.uint8)
    field = StaticFieldMesh.from_model(model, groups, -1)
    assert field is not None
    native = MjLidarCPU(model, geomgroup=groups)
    indexed = MjLidarCPU(model, geomgroup=groups, field_mesh=field)
    for obstacle, expected in [([2, 0, 0], 1.5), ([2, 4, 0], 5.0)]:
        data.mocap_pos[0] = obstacle
        mujoco.mj_forward(model, data)
        for lidar in (native, indexed):
            lidar.update(data)
            lidar.trace_rays(np.eye(4), np.array([0.0, np.pi]), np.zeros(2))
        np.testing.assert_allclose(indexed.get_distances(), native.get_distances(), atol=1e-7)
        np.testing.assert_allclose(indexed.get_distances(), [expected, 0.0], atol=1e-7)


def _trace_two_rays(lidar: MjLidarCPU) -> None:
    lidar.update(object())
    lidar.trace_rays(
        np.eye(4, dtype=np.float32),
        np.array([0.0, np.pi / 2.0], dtype=np.float32),
        np.zeros(2, dtype=np.float32),
    )


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
    _trace_two_rays(lidar)

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


def test_cpu_lidar_falls_back_when_binding_has_no_normal_slot(monkeypatch) -> None:
    """绑定不接受 `normal` 槽位时必须退回 11 参数形式，而不是让子进程崩掉。

    MuJoCo 3.4 就是这一侧；3.10 是带槽位的那一侧。任何一侧被写死都会让第一次
    raycast 抛 TypeError，表现为 `/registered_scan` 永不发布。
    """
    captured = {}

    def eleven_arg_multiray(
        m, d, pnt, vec, geomgroup, flg_static, bodyexclude, geomid, dist, nray, cutoff
    ) -> None:
        captured["args"] = (
            m, d, pnt, vec, geomgroup, flg_static, bodyexclude, geomid, dist, nray, cutoff
        )
        geomid[...] = -1

    monkeypatch.setattr(mujoco, "mj_multiRay", eleven_arg_multiray)
    lidar = MjLidarCPU(
        object(),
        cutoff_dist=12.0,
        geomgroup=np.array([1, 1, 1, 0, 1, 1], dtype=np.uint8),
        bodyexclude=7,
    )
    _trace_two_rays(lidar)

    args = captured["args"]
    assert len(args) == 11
    assert args[3].dtype == np.float64
    assert args[3].shape == (6,)
    assert args[8].dtype == np.float64
    assert args[8].shape == (2,)
    assert args[9] == 2
    assert args[10] == 12.0
    assert np.array_equal(lidar.get_distances(), np.zeros(2))


def test_cpu_lidar_does_not_swallow_genuine_argument_errors(monkeypatch) -> None:
    """两种槽位都拒绝时必须把 TypeError 抛出去，不能被兼容探测吞掉。"""

    def always_reject(*args) -> None:
        raise TypeError("incompatible function arguments")

    monkeypatch.setattr(mujoco, "mj_multiRay", always_reject)
    lidar = MjLidarCPU(object(), cutoff_dist=12.0)
    with pytest.raises(TypeError):
        _trace_two_rays(lidar)
