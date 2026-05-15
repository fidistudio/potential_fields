import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose2D, PoseStamped
from robot_interfaces.msg import ObstacleArray, Force2D

import numpy as np


class GradientDescentPlanner(Node):

    def __init__(self) -> None:
        super().__init__("gradient_descent_planner")

        # Parameters
        self.declare_parameter("repulsion_radius", 3.0)
        self.declare_parameter("goal_gain_near", 1.0)
        self.declare_parameter("goal_gain_far", 1.0)
        self.declare_parameter("repulsion_gain", 1.0)
        self.declare_parameter("tangential_gain", 2.0)
        self.declare_parameter("step_size", 0.35)
        self.declare_parameter("goal_threshold", 3.0)
        self.declare_parameter("goal_tolerance", 0.1)

        # Internal state
        self._current_pose: Pose2D | None = None
        self._goal_position: np.ndarray | None = None
        self._obstacles = []

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
        self._goal_position = np.array([msg.pose.position.x, msg.pose.position.y])

    def _on_obstacles(self, msg: ObstacleArray) -> None:
        self._obstacles = msg.obstacles

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _update(self) -> None:
        logger = self.get_logger()
        if self._current_pose is None or self._goal_position is None:
            return

        global_position = self._position_vector(self._current_pose)
        logger.info(f"Global position: {global_position}")

        # Condición de llegada en coordenadas globales (mapa)
        if self._is_within_goal_tolerance(global_position):
            logger.info("Objective is within the tolerance zone")
            self._publish_zero_vector()
            return

        # Proyección del objetivo al sistema del robot
        goal_local = self._project_goal_to_robot_frame()
        logger.info(f"Local frame goal: {goal_local}")

        goal_grad = self._goal_potential_gradient(goal_local)
        obstacle_grad = self._obstacle_potential_gradient()

        logger.info(f"Gradient goal: {goal_grad}")
        logger.info(f"Obstacle gradient: {obstacle_grad}")

        gradient = goal_grad + obstacle_grad
        logger.info(f"Total gradient: {gradient}")

        next_position = -1 * gradient

        logger.info(f"Next vector (cmd): {next_position}")

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
        tangential_gain = self.get_parameter("tangential_gain").value

        total_gradient = np.zeros(2)

        robot_position = np.zeros(2)
        goal_local = self._project_goal_to_robot_frame()

        for obstacle in self._obstacles:
            obstacle_position = np.array([obstacle.x, obstacle.y])

            delta = robot_position - obstacle_position
            d = np.linalg.norm(delta)

            if d < 1e-6 or d > repulsion_radius:
                continue

            # -----------------------------
            # Dirección hacia el objetivo
            # -----------------------------
            goal_dir = goal_local - robot_position
            goal_norm = np.linalg.norm(goal_dir)

            if goal_norm < 1e-6:
                continue

            g_hat = goal_dir / goal_norm
            t_hat = np.array([-g_hat[1], g_hat[0]])

            # Elegir lado consistente
            d_hat = delta / d
            if np.dot(t_hat, d_hat) < 0:
                t_hat = -t_hat

            # -----------------------------
            # Repulsión clásica
            # -----------------------------
            grad_rep = (
                -repulsion_gain * (1 / d - 1 / repulsion_radius) * (1 / d**3) * delta
            )

            # -----------------------------
            # Término tangencial
            # -----------------------------
            td = np.dot(t_hat, delta)

            grad_tang = -tangential_gain * (
                (1 / d**3) * td * delta + (1 / d - 1 / repulsion_radius) * t_hat
            )

            total_gradient += grad_rep + grad_tang

            self.get_logger().info(
                f"Obstacle ({obstacle.x:.2f}, {obstacle.y:.2f}) " f"d={d:.2f}"
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

        return position - step_size * (gradient / magnitude)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _project_goal_to_robot_frame(self) -> np.ndarray:
        pose = self._current_pose
        goal = self._goal_position

        theta = pose.theta
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        dx = goal[0] - pose.x
        dy = goal[1] - pose.y

        x_local = cos_theta * dx + sin_theta * dy
        y_local = -sin_theta * dx + cos_theta * dy

        return np.array([x_local, y_local])

    def _is_within_goal_tolerance(self, position: np.ndarray) -> bool:
        tolerance = self.get_parameter("goal_tolerance").value
        distance = np.linalg.norm(position - self._goal_position)
        return distance < tolerance

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
