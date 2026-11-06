"""Publish the chassis footprint and heading as RViz markers (visualisation only).

The markers are attached to the chassis frame (`gimbal_yaw_odom`, origin at the
chassis floor centre), so RViz draws them through TF and this node needs no
pose subscription.
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import Point
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray


def rectangle_points(length, width, margin=0.0, z=0.02):
    """Closed rectangle outline; +x is the vehicle heading."""
    hx = 0.5 * length + margin
    hy = 0.5 * width + margin
    corners = [(hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy), (hx, hy)]
    return [Point(x=x, y=y, z=z) for x, y in corners]


def build_markers(frame_id, length, width, margin, stamp):
    def base(ns, marker_id, kind):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = kind
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    def colour(marker, r, g, b, a=1.0):
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = r, g, b, a

    footprint = base("footprint", 0, Marker.LINE_STRIP)
    footprint.scale.x = 0.025
    colour(footprint, 0.0, 0.8, 0.0)
    footprint.points = rectangle_points(length, width)

    inflated = base("footprint_margin", 0, Marker.LINE_STRIP)
    inflated.scale.x = 0.015
    colour(inflated, 1.0, 0.8, 0.0)
    inflated.points = rectangle_points(length, width, margin)

    front = base("front_face", 0, Marker.LINE_STRIP)
    front.scale.x = 0.06
    colour(front, 1.0, 0.1, 0.0)
    hx, hy = 0.5 * length, 0.5 * width
    front.points = [Point(x=hx, y=hy, z=0.02), Point(x=hx, y=-hy, z=0.02)]

    heading = base("heading", 0, Marker.ARROW)
    heading.scale.x, heading.scale.y, heading.scale.z = 0.05, 0.10, 0.12
    colour(heading, 1.0, 0.1, 0.0)
    heading.points = [Point(x=0.0, y=0.0, z=0.05), Point(x=hx + 0.35, y=0.0, z=0.05)]

    return MarkerArray(markers=[footprint, inflated, front, heading])


class RobotFootprintMarker(Node):
    def __init__(self) -> None:
        super().__init__("robot_footprint_marker")
        self.declare_parameter("frame_id", "gimbal_yaw_odom")
        self.declare_parameter("footprint_length", 0.581)
        self.declare_parameter("footprint_width", 0.571)
        self.declare_parameter("footprint_safety_margin", 0.02)
        self.declare_parameter("topic", "/viz/robot_footprint")
        self.declare_parameter("rate_hz", 5.0)
        self._publisher = self.create_publisher(
            MarkerArray, self.get_parameter("topic").value, 1)
        self.create_timer(1.0 / float(self.get_parameter("rate_hz").value), self._tick)

    def _tick(self) -> None:
        self._publisher.publish(build_markers(
            self.get_parameter("frame_id").value,
            float(self.get_parameter("footprint_length").value),
            float(self.get_parameter("footprint_width").value),
            float(self.get_parameter("footprint_safety_margin").value),
            self.get_clock().now().to_msg(),
        ))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RobotFootprintMarker()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
