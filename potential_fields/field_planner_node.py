import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose2D, PoseStamped
from robot_interfaces.msg import ObstacleArray, Force2D

import numpy as np
import math


class FieldPlanner(Node):

    def __init__(self):
        super().__init__("field_planner")

        # Potential field parameters
        self.declare_parameter("d0", 5.0)
        self.declare_parameter("epsilon1", 1.0)
        self.declare_parameter("epsilon2", 1.0)
        self.declare_parameter("eta", 2.0)

        # Internal state
        self.pose: Pose2D | None = None
        self.goal: np.ndarray | None = None
        self.obstacles = []

        # ROS interfaces
        self.force_pub = self.create_publisher(Force2D, "/force_vector", 10)

        self.create_subscription(Pose2D, "/robot_pose", self.pose_callback, 10)
        self.create_subscription(PoseStamped, "/goal_pose", self.goal_callback, 10)
        self.create_subscription(
            ObstacleArray, "/obstacles", self.obstacles_callback, 10
        )

        self.timer = self.create_timer(0.05, self.compute_force)

    # -------------------- Callbacks --------------------

    def pose_callback(self, msg: Pose2D) -> None:
        self.pose = msg

    def goal_callback(self, msg: PoseStamped) -> None:
        self.goal = np.array([msg.pose.position.x, msg.pose.position.y])

    def obstacles_callback(self, msg: ObstacleArray) -> None:
        self.obstacles = msg.obstacles

    # -------------------- Core logic --------------------

    def compute_force(self) -> None:
        if self.pose is None or self.goal is None:
            return

        q = np.array([self.pose.x, self.pose.y])

        f_attr = self.attractive_force(q, self.goal)
        f_rep = self.repulsive_force(q, self.obstacles)

        f_total = f_attr + f_rep

        force_msg = Force2D()
        force_msg.fx = float(f_total[0])
        force_msg.fy = float(f_total[1])

        self.force_pub.publish(force_msg)

    # -------------------- Potential fields --------------------

    def attractive_force(self, q: np.ndarray, q_goal: np.ndarray) -> np.ndarray:
        epsilon1 = self.get_parameter("epsilon1").value
        epsilon2 = self.get_parameter("epsilon2").value

        vector = q_goal - q
        distance = np.linalg.norm(vector)

        if distance < 1e-6:
            return np.zeros(2)

        if distance < 5.0:
            return epsilon1 * vector

        return epsilon2 * vector / distance

    def repulsive_force(self, q: np.ndarray, obstacles) -> np.ndarray:
        d0 = self.get_parameter("d0").value
        eta = self.get_parameter("eta").value

        f_rep = np.zeros(2)

        for obs in obstacles:
            q_obs = np.array([obs.x, obs.y])
            vector = q - q_obs
            distance = np.linalg.norm(vector)

            if distance < 1e-6 or distance > d0:
                continue

            gain = eta * (1 / distance - 1 / d0) * (1 / distance**3)
            f_rep += gain * vector

        return f_rep


def main():
    rclpy.init()
    node = FieldPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
