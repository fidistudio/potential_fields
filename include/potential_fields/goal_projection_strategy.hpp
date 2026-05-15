#pragma once

#include <optional>
#include <string>

#include "geometry_msgs/msg/pose2_d.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "rclcpp/rclcpp.hpp"

namespace potential_fields {

/**
 * @brief 2D point in the robot frame.
 */
struct Point2D {
  double x{0.0};
  double y{0.0};
};

/**
 * @brief Pure interface for goal projection strategies.
 *
 * Implementations decide how to express the goal position
 * in the robot's local frame — either via manual trigonometry
 * (simulation) or a tf2 lookup (real robot).
 */
class IGoalProjection {
public:
  virtual ~IGoalProjection() = default;

  /**
   * @brief Project the goal into the robot's local frame.
   *
   * @param pose         Current robot pose in the world frame.
   * @param goal_stamped Goal pose with its source frame_id and timestamp.
   * @return             Goal expressed in the robot frame, or
   *                     std::nullopt when the projection is unavailable.
   */
  [[nodiscard]] virtual std::optional<Point2D>
  project(const geometry_msgs::msg::Pose2D &pose,
          const geometry_msgs::msg::PoseStamped &goal_stamped) const = 0;
};

} // namespace potential_fields
