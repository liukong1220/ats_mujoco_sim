"""`mj_multiRay` Python 绑定的 ABI 兼容层。

MuJoCo 3.4 的绑定签名为::

    mj_multiRay(m, d, pnt, vec, geomgroup, flg_static, bodyexclude,
                geomid, dist, nray, cutoff)

3.10 在 ``dist`` 与 ``nray`` 之间插入了本桥不消费的 ``normal`` 槽位。写死任一侧
都会让第一次 raycast 抛 ``TypeError``：LiDAR 子进程随即退出，
``/local_pointcloud`` 与 ``/registered_scan`` 永不发布，闭环外观表现为
ROGMap ``cloud_age=inf`` 与定位缺失，而不是一次 API 失配。

因此这里不假设版本，只在首次调用时探测实际绑定并按被调用对象缓存结论。
pybind11 在参数转换阶段就抛 ``TypeError``，早于任何 C 调用，所以探测不产生副作用。
"""

from __future__ import annotations

from typing import Any

import mujoco

# 以被调用对象为键，避免同一进程内（例如单测 monkeypatch 与真实绑定混用）
# 把一个绑定的结论错用到另一个绑定上。
_normal_slot_cache: dict[Any, bool] = {}


def multi_ray(
    model: Any,
    data: Any,
    pnt: Any,
    vec: Any,
    geomgroup: Any,
    flg_static: int,
    bodyexclude: int,
    geomid: Any,
    dist: Any,
    nray: int,
    cutoff: float,
) -> None:
    """按实际绑定的槽位数调用 ``mujoco.mj_multiRay``。

    参数顺序固定为 3.4 的语义；``normal`` 槽位存在时由本函数补 ``None``。
    """
    func = mujoco.mj_multiRay
    head = (model, data, pnt, vec, geomgroup, flg_static, bodyexclude, geomid, dist)
    with_normal = (*head, None, nray, cutoff)
    without_normal = (*head, nray, cutoff)

    cached = _normal_slot_cache.get(func)
    if cached is True:
        func(*with_normal)
        return
    if cached is False:
        func(*without_normal)
        return

    try:
        func(*with_normal)
    except TypeError as with_normal_error:
        try:
            func(*without_normal)
        except TypeError as without_normal_error:
            # 两种槽位都不接受说明是真正的参数错误（dtype/shape），
            # 不能被兼容探测吞掉。
            raise without_normal_error from with_normal_error
        _normal_slot_cache[func] = False
        return
    _normal_slot_cache[func] = True
