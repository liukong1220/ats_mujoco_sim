"""Thread-parallel ray tracing for the simulated Mid360 (CPU backend).

``mj_multiRay`` releases the GIL, so the rays of one segment can be shared by
several threads, each with its own ``MjData`` (the model is read-only).  A real
Mid360 fires its 200 k points per second continuously; one thread traces about
12000 rays in 0.4 s on the RMUC 2027 field, which is why the simulation ran at a
tenth of real time.  Splitting every segment across the cores brings that down
roughly with the thread count (measured 425 / 221 / 125 / 87 ms for 1 / 2 / 4 / 8
threads).  The split changes nothing about the result: every ray is still traced
against the same state with the same function, so distances and points are bit
identical to the serial run.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from typing import Callable

import mujoco
import numpy as np

MAX_AUTO_THREADS = 8


def resolve_thread_count(requested: int, cpu_count: int | None = None) -> int:
    """0 means auto: a quarter of the logical CPUs, 1..MAX_AUTO_THREADS."""
    requested = int(requested)
    if requested < 0:
        raise ValueError("lidar_threads must be >= 0 (0 = auto)")
    if requested > 0:
        return requested
    cpus = int(cpu_count if cpu_count is not None else (os.cpu_count() or 1))
    return max(1, min(MAX_AUTO_THREADS, cpus // 4))


def split_bounds(count: int, parts: int) -> list[tuple[int, int]]:
    """Contiguous, non-empty ``(first, last)`` index ranges covering ``count`` items."""
    if count <= 0:
        return []
    edges = np.rint(np.linspace(0, count, min(max(1, parts), count) + 1)).astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


class ParallelRayTracer:
    """One ``MjData`` and one LiDAR wrapper per worker thread."""

    def __init__(self, model: mujoco.MjModel, make_lidar: Callable[[], object], threads: int) -> None:
        if threads < 1:
            raise ValueError("threads must be >= 1")
        self.model = model
        self.threads = int(threads)
        self._datas = [mujoco.MjData(model) for _ in range(self.threads)]
        self._lidars = [make_lidar() for _ in range(self.threads)]
        self._pool = (
            ThreadPoolExecutor(max_workers=self.threads, thread_name_prefix="lidar-trace")
            if self.threads > 1 else None
        )

    def set_state(self, time: float, qpos: np.ndarray, qvel: np.ndarray) -> None:
        """Put every worker's ``MjData`` at the same simulation state."""
        for data in self._datas:
            data.time = float(time)
            data.qpos[:] = qpos
            data.qvel[:] = qvel
            mujoco.mj_kinematics(self.model, data)

    def _trace_part(self, worker: int, theta: np.ndarray, phi: np.ndarray):
        lidar = self._lidars[worker]
        distances = np.array(lidar.trace_rays(self._datas[worker], theta, phi), copy=True)
        points = np.array(lidar.get_hit_points(), copy=True)
        return distances, points

    def trace(self, theta: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Distances and sensor-frame hit points of every ray, in input order."""
        parts = split_bounds(theta.shape[0], self.threads)
        if not parts:
            return np.zeros(0), np.zeros((0, 3))
        if self._pool is None or len(parts) == 1:
            first, last = parts[0][0], parts[-1][1]
            return self._trace_part(0, theta[first:last], phi[first:last])
        futures = [
            self._pool.submit(self._trace_part, worker, theta[first:last], phi[first:last])
            for worker, (first, last) in enumerate(parts)
        ]
        results = [future.result() for future in futures]
        return (
            np.concatenate([result[0] for result in results]),
            np.concatenate([result[1] for result in results]),
        )

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
