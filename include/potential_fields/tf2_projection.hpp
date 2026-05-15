#pragma once

#include "potential_fields/goal_projection_strategy.hpp"

#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

namespace potential_fields {

/**
 * @brief Real-robot strategy: projects the goal via a tf2 lookup.
 *
 * Queries the tf tree at the latest available time to transform the
 * goal from its source frame into the robot frame.
 * Returns std::nullopt (with a throttled warning) when tf2 is
 * temporarily unavailable.
 */
class Tf2Projection final : public IGoalProjection {
public:
  /**
   * @param node         ROS node providing the clock and logger.
   * @param robot_frame  Target frame for the projection.
   */
  explicit Tf2Projection(rclcpp::Node &node,
                         std::string robot_frame = "base_footprint");

  [[nodiscard]] std::optional<Point2D>
  project(const geometry_msgs::msg::Pose2D &pose,
          const geometry_msgs::msg::PoseStamped &goal_stamped) const override;

private:
  rclcpp::Logger logger_;
  rclcpp::Clock::SharedPtr clock_;

  std::string robot_frame_;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
};

} // namespace potential_fields
