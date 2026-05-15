#include "potential_fields/manual_projection.hpp"

#include <cmath>

namespace potential_fields {

std::optional<Point2D> ManualProjection::project(
    const geometry_msgs::msg::Pose2D &pose,
    const geometry_msgs::msg::PoseStamped &goal_stamped) const {
  const double dx = goal_stamped.pose.position.x - pose.x;
  const double dy = goal_stamped.pose.position.y - pose.y;

  const double cos_theta = std::cos(pose.theta);
  const double sin_theta = std::sin(pose.theta);

  return Point2D{cos_theta * dx + sin_theta * dy,
                 -sin_theta * dx + cos_theta * dy};
}

} // namespace potential_fields
