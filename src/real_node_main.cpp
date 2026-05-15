#include "potential_fields/gradient_descent_planner_node.hpp"
#include "potential_fields/tf2_projection.hpp"

#include "rclcpp/rclcpp.hpp"

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);

  // Phase 1 — create the planner node without a projection strategy yet.
  // Tf2Projection needs a live node reference for its clock and tf listener.
  auto node =
      std::make_shared<potential_fields::GradientDescentPlannerNode>(nullptr);

  // Phase 2 — build Tf2Projection referencing the live node.
  // robot_frame can be overridden via the "robot_frame" ROS param if needed.
  auto projection = std::make_shared<potential_fields::Tf2Projection>(*node);

  // Phase 3 — inject before the first timer tick fires.
  node->setProjection(projection);

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
