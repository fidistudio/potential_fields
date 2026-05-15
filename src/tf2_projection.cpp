#include "potential_fields/tf2_projection.hpp"

#include <utility>

#include "geometry_msgs/msg/point_stamped.hpp"
#include "tf2/exceptions.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

namespace potential_fields {

Tf2Projection::Tf2Projection(rclcpp::Node &node, std::string robot_frame)
    : logger_{node.get_logger()}, clock_{node.get_clock()},
      robot_frame_{std::move(robot_frame)},
      tf_buffer_{std::make_shared<tf2_ros::Buffer>(node.get_clock())},
      tf_listener_{std::make_shared<tf2_ros::TransformListener>(*tf_buffer_,
                                                                &node, false)} {
}

std::optional<Point2D> Tf2Projection::project(
    const geometry_msgs::msg::Pose2D & /*pose*/,
    const geometry_msgs::msg::PoseStamped &goal_stamped) const {

  geometry_msgs::msg::TransformStamped tf_stamped;

  try {
    tf_stamped = tf_buffer_->lookupTransform(
        robot_frame_, goal_stamped.header.frame_id, rclcpp::Time(0),
        rclcpp::Duration::from_seconds(0.05));

  } catch (const tf2::TransformException &ex) {

    RCLCPP_WARN_THROTTLE(logger_, *clock_, 2000, "TF unavailable: %s -> %s: %s",
                         goal_stamped.header.frame_id.c_str(),
                         robot_frame_.c_str(), ex.what());

    return std::nullopt;
  }

  geometry_msgs::msg::PointStamped point_in;
  point_in.header = goal_stamped.header;

  point_in.point.x = goal_stamped.pose.position.x;
  point_in.point.y = goal_stamped.pose.position.y;
  point_in.point.z = 0.0;

  geometry_msgs::msg::PointStamped point_out;

  tf2::doTransform(point_in, point_out, tf_stamped);

  return Point2D{point_out.point.x, point_out.point.y};
}

} // namespace potential_fields
