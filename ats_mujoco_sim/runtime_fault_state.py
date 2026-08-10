"""Runtime fault-state transitions for the MuJoCo simulator.

This module is deliberately free of ROS message imports (``carstatemsgs``,
``manda_can_control``), ``rclpy`` and ``mujoco`` so the fault-state contract
stays collectible and executable from a plain ``pytest`` run against a source
checkout.  ``sim_node`` delegates to these helpers and keeps its own ROS-facing
plumbing; the behaviour and the operator-visible log wording live here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np


#: Runtime parameters this module owns.  Anything else is left to the caller.
RUNTIME_FAULT_PARAMETERS = ("freeze_motion", "lidar_occlusion_enabled")


@dataclass
class RuntimeFaultParameterOutcome:
    """Result of screening and applying one ``SetParameters`` request."""

    successful: bool = True
    reason: str = ""
    #: Warn-level messages, in emission order, one per applied transition.
    messages: list[str] = field(default_factory=list)


def freeze_motion_message(value: bool) -> str:
    """Operator-visible wording for a chassis freeze transition."""
    execution_state = "held" if value else "enabled"
    return (
        f"freeze_motion={value}: chassis execution {execution_state} "
        "while sensors remain active"
    )


def lidar_occlusion_message(value: bool) -> str:
    """Operator-visible wording for a LiDAR occlusion transition."""
    sensor_state = "empty returns enabled" if value else "raycast enabled"
    return (
        f"lidar_occlusion_enabled={value}: {sensor_state}; "
        "LiDAR headers and publication cadence remain active"
    )


def apply_runtime_fault_parameters(
    state: Any, requests: Iterable[Any]
) -> RuntimeFaultParameterOutcome:
    """Apply ``freeze_motion`` / ``lidar_occlusion_enabled`` to ``state``.

    ``state`` must expose ``sim_lock`` plus the attribute each request targets;
    ``lidar_occlusion_enabled`` additionally uses ``lidar_occlusion_event`` to
    keep the spawned LiDAR worker in sync.  A non-boolean value is rejected
    before anything is applied, so a malformed request never leaves the
    simulator half-transitioned.  Parameters outside
    :data:`RUNTIME_FAULT_PARAMETERS` are ignored here.
    """
    requested = list(requests)
    # ROS rejects a SetParameters request as one transaction.  Validate the
    # whole owned subset before touching simulator state so a valid first item
    # followed by an invalid item cannot leave a partially applied fault.
    for request in requested:
        name = request.name
        if name not in RUNTIME_FAULT_PARAMETERS:
            continue
        value = request.value
        if not isinstance(value, bool):
            return RuntimeFaultParameterOutcome(
                successful=False,
                reason=f"{name} must be a boolean",
            )
    outcome = RuntimeFaultParameterOutcome()
    for request in requested:
        name = request.name
        if name not in RUNTIME_FAULT_PARAMETERS:
            continue
        value = request.value
        if name == "freeze_motion":
            with state.sim_lock:
                state.freeze_motion = value
            outcome.messages.append(freeze_motion_message(value))
            continue
        with state.sim_lock:
            state.lidar_occlusion_enabled = value
            event = getattr(state, "lidar_occlusion_event", None)
            if event is not None:
                if value:
                    event.set()
                else:
                    event.clear()
        outcome.messages.append(lidar_occlusion_message(value))
    return outcome


def empty_lidar_return() -> np.ndarray:
    """The occluded LiDAR return: zero points, still a legal ``(0, 3)`` array."""
    return np.zeros((0, 3), dtype=np.float32)


def normalize_pointcloud_points(points: Any, intensity: float = 1.0) -> np.ndarray:
    """Coerce raycast output into the published ``(N, 4)`` XYZI layout.

    An empty return stays empty rather than raising, so an occluded sensor keeps
    publishing on cadence instead of tearing down the worker.
    """
    points = np.ascontiguousarray(points, dtype=np.float32)
    if points.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] not in (3, 4):
        raise ValueError("Point cloud must have shape (N, 3) or (N, 4)")
    if points.shape[1] == 4:
        return points
    intensities = np.full((points.shape[0], 1), float(intensity), dtype=np.float32)
    return np.ascontiguousarray(np.hstack((points, intensities)), dtype=np.float32)


@dataclass(frozen=True)
class PointcloudPayload:
    """Wire fields for one ``PointCloud2`` publication."""

    width: int
    row_step: int
    data: bytes


def pointcloud_payload(
    points: Any, point_step: int, intensity: float = 1.0
) -> PointcloudPayload:
    """Build the width/row_step/data triple for a cloud of ``points``."""
    normalized = normalize_pointcloud_points(points, intensity)
    width = int(normalized.shape[0])
    return PointcloudPayload(
        width=width,
        row_step=int(point_step) * width,
        data=normalized.tobytes(),
    )


def resolve_lidar_return(occluded: bool, raycast: Any) -> np.ndarray:
    """Return the occluded empty cloud, or the result of ``raycast()``.

    ``raycast`` is only invoked when the sensor is not occluded, which is what
    keeps the occlusion fixture from paying for a full ray sweep.
    """
    if occluded:
        return empty_lidar_return()
    return raycast()
