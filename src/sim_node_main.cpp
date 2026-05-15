#include "potential_fields/gradient_descent_planner_node.hpp"
#include "potential_fields/manual_projection.hpp"

#include "rclcpp/rclcpp.hpp"

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);

  // ManualProjection has no runtime dependencies — inject directly.
  auto projection = std::make_shared<potential_fields::ManualProjection>();
  auto node = std::make_shared<potential_fields::GradientDescentPlannerNode>(
      projection);

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
