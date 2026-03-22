import rclpy
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import Pose2D, PoseStamped, PointStamped
from robot_interfaces.msg import ObstacleArray, Force2D

from tf2_ros import (
    Buffer,
    TransformListener,
    LookupException,
    ConnectivityException,
    ExtrapolationException,
)
import tf2_geometry_msgs

import numpy as np


class GradientDescentPlanner(Node):

    def __init__(self) -> None:
        super().__init__("gradient_descent_planner")

        # Parameters
        self.declare_parameter("repulsion_radius", 5.0)
        self.declare_parameter("goal_gain_near", 5.0)
        self.declare_parameter("goal_gain_far", 3.0)
        self.declare_parameter("repulsion_gain", 7.0)
        self.declare_parameter("step_size", 0.35)
        self.declare_parameter("goal_threshold", 3.0)
        self.declare_parameter("goal_tolerance", 0.5)
        self.declare_parameter("robot_frame", "base_footprint")

        # Internal state
        self._current_pose: Pose2D | None = None
        self._goal_stamped: PoseStamped | None = None
        self._obstacles = []

        # tf2
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # Publishers
        self._descent_pub = self.create_publisher(
            Force2D, "/gradient_descent_vector", 10
        )

        # Subscriptions
        self.create_subscription(Pose2D, "/robot_pose", self._on_pose, 10)
        self.create_subscription(PoseStamped, "/goal_pose", self._on_goal, 10)
        self.create_subscription(ObstacleArray, "/obstacles", self._on_obstacles, 10)

        # Timer
        self.create_timer(0.05, self._update)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_pose(self, msg: Pose2D) -> None:
        self._current_pose = msg

    def _on_goal(self, msg: PoseStamped) -> None:
        self._goal_stamped = msg

    def _on_obstacles(self, msg: ObstacleArray) -> None:
        self._obstacles = msg.obstacles

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _update(self) -> None:
        if self._current_pose is None or self._goal_stamped is None:
            return

        global_position = self._position_vector(self._current_pose)
        goal_global = np.array(
            [
                self._goal_stamped.pose.position.x,
                self._goal_stamped.pose.position.y,
            ]
        )

        dist_to_goal = np.linalg.norm(global_position - goal_global)
        self.get_logger().info(
            f"[POSE] x={global_position[0]:.3f}  y={global_position[1]:.3f}  θ={self._current_pose.theta:.3f} rad  |  "
            f"[GOAL global] x={goal_global[0]:.3f}  y={goal_global[1]:.3f}  |  "
            f"dist={dist_to_goal:.3f} m",
            throttle_duration_sec=0.5,
        )

        if self._is_within_goal_tolerance(global_position, goal_global):
            self.get_logger().info("✓ Goal alcanzado — publicando vector cero.")
            self._publish_zero_vector()
            return

        goal_local = self._project_goal_to_robot_frame()
        if goal_local is None:
            return

        self.get_logger().info(
            f"[GOAL local] x={goal_local[0]:.3f}  y={goal_local[1]:.3f}  "
            f"dist={np.linalg.norm(goal_local):.3f} m",
            throttle_duration_sec=0.5,
        )

        grad_goal = self._goal_potential_gradient(goal_local)
        grad_obs = self._obstacle_potential_gradient()
        gradient = grad_goal + grad_obs

        self.get_logger().info(
            f"[GRAD] goal=({grad_goal[0]:.3f}, {grad_goal[1]:.3f})  "
            f"obs=({grad_obs[0]:.3f}, {grad_obs[1]:.3f})  "
            f"total=({gradient[0]:.3f}, {gradient[1]:.3f})  "
            f"|total|={np.linalg.norm(gradient):.3f}",
            throttle_duration_sec=0.5,
        )

        next_position = self._gradient_descent_step(np.zeros(2), gradient)

        self.get_logger().info(
            f"[CMD] fx={next_position[0]:.3f}  fy={next_position[1]:.3f}",
            throttle_duration_sec=0.5,
        )

        self._publish_vector(next_position)

    # ------------------------------------------------------------------
    # Gradient computation
    # ------------------------------------------------------------------

    def _goal_potential_gradient(self, goal_local: np.ndarray) -> np.ndarray:
        goal_gain_near = self.get_parameter("goal_gain_near").value
        goal_gain_far = self.get_parameter("goal_gain_far").value
        goal_threshold = self.get_parameter("goal_threshold").value

        delta = -goal_local
        distance = np.linalg.norm(delta)

        if distance < 1e-6:
            return np.zeros(2)

        if distance < goal_threshold:
            return goal_gain_near * delta

        return goal_gain_far * delta / distance

    def _obstacle_potential_gradient(self) -> np.ndarray:
        repulsion_radius = self.get_parameter("repulsion_radius").value
        repulsion_gain = self.get_parameter("repulsion_gain").value

        total_gradient = np.zeros(2)
        robot_position = np.zeros(2)

        self.get_logger().info(
            f"[OBS] total={len(self._obstacles)}  dentro de radio({repulsion_radius}m):",
            throttle_duration_sec=0.5,
        )

        for i, obstacle in enumerate(self._obstacles):
            obstacle_position = np.array([obstacle.x, obstacle.y])
            delta = robot_position - obstacle_position
            distance = np.linalg.norm(delta)

            if distance < 1e-6 or distance > repulsion_radius:
                continue

            scaling = (
                -repulsion_gain
                * (1 / distance - 1 / repulsion_radius)
                * (1 / distance**3)
            )

            contrib = scaling * delta
            total_gradient += contrib

            self.get_logger().info(
                f"  obs[{i}] x={obstacle.x:.3f}  y={obstacle.y:.3f}  "
                f"dist={distance:.3f}m  "
                f"grad=({contrib[0]:.3f}, {contrib[1]:.3f})",
                throttle_duration_sec=0.5,
            )

        return total_gradient

    # ------------------------------------------------------------------
    # Gradient descent step
    # ------------------------------------------------------------------

    def _gradient_descent_step(
        self, position: np.ndarray, gradient: np.ndarray
    ) -> np.ndarray:
        step_size = self.get_parameter("step_size").value
        magnitude = np.linalg.norm(gradient)

        if magnitude < 1e-6:
            return position

        return position - step_size * gradient

    # ------------------------------------------------------------------
    # Proyección del goal con tf2
    # ------------------------------------------------------------------

    def _project_goal_to_robot_frame(self) -> np.ndarray | None:
        robot_frame = self.get_parameter("robot_frame").value
        goal_stamped = self._goal_stamped

        point_in = PointStamped()
        point_in.header = goal_stamped.header
        point_in.point.x = goal_stamped.pose.position.x
        point_in.point.y = goal_stamped.pose.position.y
        point_in.point.z = 0.0

        try:
            tf = self._tf_buffer.lookup_transform(
                robot_frame,
                goal_stamped.header.frame_id,
                Time(),
                timeout=rclpy.duration.Duration(seconds=0.05),
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(
                f"TF no disponible: {goal_stamped.header.frame_id} → "
                f"{robot_frame}: {e}",
                throttle_duration_sec=2.0,
            )
            return None

        point_out = tf2_geometry_msgs.do_transform_point(point_in, tf)
        return np.array([point_out.point.x, point_out.point.y])

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _is_within_goal_tolerance(self, position: np.ndarray, goal: np.ndarray) -> bool:
        tolerance = self.get_parameter("goal_tolerance").value
        return np.linalg.norm(position - goal) < tolerance

    def _position_vector(self, pose: Pose2D) -> np.ndarray:
        return np.array([pose.x, pose.y])

    def _publish_vector(self, vector: np.ndarray) -> None:
        msg = Force2D()
        msg.fx = float(vector[0])
        msg.fy = float(vector[1])
        self._descent_pub.publish(msg)

    def _publish_zero_vector(self) -> None:
        msg = Force2D()
        msg.fx = 0.0
        msg.fy = 0.0
        self._descent_pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = GradientDescentPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
