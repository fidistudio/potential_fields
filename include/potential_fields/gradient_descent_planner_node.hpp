#pragma once

#include "potential_fields/goal_projection_strategy.hpp"
#include "potential_fields/potential_field_computer.hpp"

#include <memory>
#include <optional>
#include <vector>

#include "geometry_msgs/msg/pose2_d.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "robot_interfaces/msg/force2_d.hpp"
#include "robot_interfaces/msg/obstacle.hpp"
#include "robot_interfaces/msg/obstacle_array.hpp"

namespace potential_fields {

/**
 * @brief ROS 2 node that computes a potential field velocity command.
 *
 * Pipeline per timer tick:
 *   1. Guard: pose and goal must be available.
 *   2. Check goal tolerance in the world frame → publish zero if reached.
 *   3. Delegate goal projection to IGoalProjection.
 *   4. Compute gradients via PotentialFieldComputer.
 *   5. Publish cmd = -(grad_goal + grad_obstacle).
 *
 * The node is decoupled from the projection method:
 * inject ManualProjection for simulation or Tf2Projection for the real robot.
 */
class GradientDescentPlannerNode : public rclcpp::Node {
public:
  /**
   * @param projection  Concrete goal projection strategy (injected).
   * @param options     Standard NodeOptions forwarded to rclcpp::Node.
   */
  explicit GradientDescentPlannerNode(
      std::shared_ptr<IGoalProjection> projection,
      const rclcpp::NodeOptions &options = rclcpp::NodeOptions{});

  /**
   * @brief Replace the active projection strategy at runtime.
   *
   * Allows real_node_main to inject Tf2Projection after construction,
   * before the first timer tick fires.
   */
  void setProjection(std::shared_ptr<IGoalProjection> projection);

private:
  // ── Timer callback ─────────────────────────────────────────────────────────
  void update();

  // ── Topic callbacks ────────────────────────────────────────────────────────
  void onPose(const geometry_msgs::msg::Pose2D &msg);
  void onGoal(const geometry_msgs::msg::PoseStamped &msg);
  void onObstacles(const robot_interfaces::msg::ObstacleArray &msg);

  // ── Helpers ────────────────────────────────────────────────────────────────
  [[nodiscard]] PotentialFieldParams readParams() const;
  [[nodiscard]] bool isWithinGoalTolerance(const Point2D &goal_local) const;
  void publishVector(const Point2D &vec);
  void publishZeroVector();

  // ── Strategy ───────────────────────────────────────────────────────────────
  std::shared_ptr<IGoalProjection> projection_;

  // ── State ──────────────────────────────────────────────────────────────────
  std::optional<geometry_msgs::msg::Pose2D> current_pose_;
  std::optional<geometry_msgs::msg::PoseStamped> goal_stamped_;
  std::vector<robot_interfaces::msg::Obstacle> obstacles_;

  // ── ROS interfaces ─────────────────────────────────────────────────────────
  rclcpp::Subscription<geometry_msgs::msg::Pose2D>::SharedPtr pose_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
  rclcpp::Subscription<robot_interfaces::msg::ObstacleArray>::SharedPtr
      obstacles_sub_;
  rclcpp::Publisher<robot_interfaces::msg::Force2D>::SharedPtr cmd_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

} // namespace potential_fields
