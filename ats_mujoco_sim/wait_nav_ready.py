#!/usr/bin/env python3
"""Event-driven full-stack readiness barrier for launching behavior-tree cruise.

Blocks until simulation, localization (Point-LIO + fusion TRACKING + TF),
planning map (ROGMap adapter ready + valid snapshot matching localization epoch),
navigation action server, and core planning/control nodes are all ready and hold
continuously for `stable_sec`. Exits 0 on readiness, non-zero on signal abort.
"""

import argparse
import signal
import sys
import time

import rclpy
from ats_navigation_interfaces.action import NavigateToPose
from ats_navigation_interfaces.msg import LocalizationStatus, PlanningMapSnapshot
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformException, TransformListener


REQUIRED_NODES = (
    "localization_fusion",
    "ats_goal_manager",
    "minco_planner",
    "ats_swerve_mpc",
    "cmd_vel_arbiter",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait for full sim + localization + nav stack readiness.")
    parser.add_argument("--stable-sec", type=float, default=1.5, help="Continuous ready hold duration (wall s).")
    parser.add_argument("--log-period-sec", type=float, default=5.0, help="Status log interval (wall s).")
    parser.add_argument(
        "--require-sim-truth",
        action="store_true",
        help="Require /clock and /simulation/ground_truth/odometry (simulation mode).",
    )
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--base-frame", default="gimbal_yaw_odom")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    stable_sec = max(0.0, float(args.stable_sec))
    log_period = max(1.0, float(args.log_period_sec))

    stop = {"flag": False}

    def _on_signal(_signum, _frame):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    rclpy.init()
    node = Node("wait_mapping_cruise_ready")
    tf_buffer = Buffer()
    _tf_listener = TransformListener(tf_buffer, node, spin_thread=False)

    latched_qos = QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )

    state = {
        "clock_ok": not args.require_sim_truth,
        "truth_ok": not args.require_sim_truth,
        "loc_odom_ok": False,
        "loc_state": -1,
        "loc_epoch": 0,
        "map_to_odom_valid": False,
        "map_ready": False,
        "snap_ready": False,
        "snap_gen": 0,
        "snap_epoch": 0,
    }

    if args.require_sim_truth:
        def _on_clock(msg: Clock):
            if msg.clock.sec > 0 or msg.clock.nanosec > 0:
                state["clock_ok"] = True

        node.create_subscription(Clock, "/clock", _on_clock, qos_profile_sensor_data)
        node.create_subscription(
            Odometry,
            "/simulation/ground_truth/odometry",
            lambda _m: state.__setitem__("truth_ok", True),
            10,
        )

    def _on_loc_status(msg: LocalizationStatus):
        state["loc_state"] = int(msg.state)
        state["loc_epoch"] = int(msg.epoch)
        state["map_to_odom_valid"] = bool(msg.map_to_odom_valid)

    def _on_snap(msg: PlanningMapSnapshot):
        state["snap_ready"] = bool(msg.ready)
        state["snap_gen"] = int(msg.source_generation)
        state["snap_epoch"] = int(msg.localization_epoch)

    node.create_subscription(LocalizationStatus, "/localization/status", _on_loc_status, latched_qos)
    node.create_subscription(
        Odometry,
        "/localization",
        lambda _m: state.__setitem__("loc_odom_ok", True),
        qos_profile_sensor_data,
    )
    node.create_subscription(
        Bool,
        "/rog_map_adapter/ready",
        lambda m: state.__setitem__("map_ready", bool(m.data)),
        latched_qos,
    )
    node.create_subscription(PlanningMapSnapshot, "/rog_map_adapter/planning_snapshot", _on_snap, latched_qos)

    action_client = ActionClient(node, NavigateToPose, "/ats_navigate_to_pose")

    node.get_logger().info(
        f"【等待巡航就绪】等待仿真、定位(TRACKING+TF)、规划地图快照与控制链全部稳定 {stable_sec:.1f} s 后加载行为树..."
    )

    ready_since = None
    last_log = time.monotonic()

    try:
        while rclpy.ok() and not stop["flag"]:
            rclpy.spin_once(node, timeout_sec=0.1)
            now = time.monotonic()

            sim_ok = state["clock_ok"] and state["truth_ok"]
            loc_ok = (
                state["loc_odom_ok"]
                and state["loc_state"] == int(LocalizationStatus.STATE_TRACKING)
                and state["loc_epoch"] > 0
                and state["map_to_odom_valid"]
            )
            tf_ok = False
            if loc_ok:
                try:
                    tf_buffer.lookup_transform(
                        args.map_frame, args.odom_frame, Time(), timeout=Duration(seconds=0.0)
                    )
                    tf_buffer.lookup_transform(
                        args.odom_frame, args.base_frame, Time(), timeout=Duration(seconds=0.0)
                    )
                    tf_ok = True
                except TransformException:
                    tf_ok = False

            map_ok = (
                state["map_ready"]
                and state["snap_ready"]
                and state["snap_gen"] > 0
                and state["snap_epoch"] == state["loc_epoch"]
            )
            action_ok = action_client.server_is_ready()

            live_nodes = set(node.get_node_names())
            present_count = sum(1 for name in REQUIRED_NODES if name in live_nodes)
            nodes_ok = present_count == len(REQUIRED_NODES)

            all_ok = sim_ok and loc_ok and tf_ok and map_ok and action_ok and nodes_ok
            if all_ok:
                if ready_since is None:
                    ready_since = now
                    node.get_logger().info(
                        f"【巡航条件满足】仿真/定位(TRACKING epoch={state['loc_epoch']})/TF/地图(gen={state['snap_gen']})"
                        f"/Action/核心节点({present_count}/{len(REQUIRED_NODES)}) 均已就绪，保持稳定 {stable_sec:.1f} s..."
                    )
                if now - ready_since >= stable_sec:
                    node.get_logger().info(
                        f"【巡航就绪】全链已稳定维持 {now - ready_since:.2f} s，开始加载行为树巡航。"
                    )
                    return 0
            else:
                ready_since = None
                if now - last_log >= log_period:
                    last_log = now
                    missing = [n for n in REQUIRED_NODES if n not in live_nodes]
                    node.get_logger().info(
                        f"【等待巡航就绪】sim={int(sim_ok)} loc={int(loc_ok)}"
                        f"(state={state['loc_state']},epoch={state['loc_epoch']},odom={int(state['loc_odom_ok'])}) "
                        f"tf={int(tf_ok)} map={int(map_ok)}(ready={int(state['map_ready'])},gen={state['snap_gen']}) "
                        f"action={int(action_ok)} nodes={present_count}/{len(REQUIRED_NODES)}"
                        + (f" missing={missing}" if missing else "")
                    )
        return 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
