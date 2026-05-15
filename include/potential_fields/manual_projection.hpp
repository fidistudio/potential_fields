#pragma once

#include "potential_fields/goal_projection_strategy.hpp"

namespace potential_fields {

/**
 * @brief Simulation strategy: projects the goal using manual trigonometry.
 *
 * Applies a 2D rotation matrix derived from the robot's heading (theta)
 * to express the world-frame goal vector in the robot's local frame.
 * No tf2 required — suitable for Gazebo / mock environments.
 */
class ManualProjection final : public IGoalProjection {
public:
  [[nodiscard]] std::optional<Point2D>
  project(const geometry_msgs::msg::Pose2D &pose,
          const geometry_msgs::msg::PoseStamped &goal_stamped) const override;
};

} // namespace potential_fields
