#include "potential_fields/gradient_descent_planner_node.hpp"
#include "potential_fields/goal_projection_strategy.hpp"
#include "potential_fields/potential_field_computer.hpp"

#include <cmath>
#include <string>

namespace potential_fields {

// ── Constructor
// ───────────────────────────────────────────────────────────────

GradientDescentPlannerNode::GradientDescentPlannerNode(
    std::shared_ptr<IGoalProjection> projection,
    const rclcpp::NodeOptions &options)
    : Node("gradient_descent_planner", options),
      projection_{std::move(projection)} {
  declare_parameter("pose_topic", "/robot_pose");
  declare_parameter("goal_topic", "/goal_pose");
  declare_parameter("obstacles_topic", "/obstacles");
  declare_parameter("cmd_topic", "/gradient_descent_vector");
  declare_parameter("goal_tolerance", 0.1);
  declare_parameter("repulsion_radius", 3.0);
  declare_parameter("goal_gain_near", 1.0);
  declare_parameter("goal_gain_far", 1.0);
  declare_parameter("goal_threshold", 3.0);
  declare_parameter("repulsion_gain", 1.0);
  declare_parameter("tangential_gain", 3.0);
  declare_parameter("step_size", 0.1);

  const auto pose_topic = get_parameter("pose_topic").as_string();
  const auto goal_topic = get_parameter("goal_topic").as_string();
  const auto obstacles_topic = get_parameter("obstacles_topic").as_string();
  const auto cmd_topic = get_parameter("cmd_topic").as_string();

  pose_sub_ = create_subscription<geometry_msgs::msg::Pose2D>(
      pose_topic, 10,
      [this](const geometry_msgs::msg::Pose2D &msg) { onPose(msg); });

  goal_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      goal_topic, 10,
      [this](const geometry_msgs::msg::PoseStamped &msg) { onGoal(msg); });

  obstacles_sub_ = create_subscription<robot_interfaces::msg::ObstacleArray>(
      obstacles_topic, 10,
      [this](const robot_interfaces::msg::ObstacleArray &msg) {
        onObstacles(msg);
      });

  cmd_pub_ = create_publisher<robot_interfaces::msg::Force2D>(cmd_topic, 10);

  timer_ =
      create_wall_timer(std::chrono::milliseconds(50), [this]() { update(); });

  RCLCPP_INFO(get_logger(), "GradientDescentPlannerNode ready");
}

// ── Strategy setter
// ───────────────────────────────────────────────────────────

void GradientDescentPlannerNode::setProjection(
    std::shared_ptr<IGoalProjection> projection) {
  projection_ = std::move(projection);
}

// ── Topic callbacks
// ───────────────────────────────────────────────────────────

void GradientDescentPlannerNode::onPose(const geometry_msgs::msg::Pose2D &msg) {
  current_pose_ = msg;
}

void GradientDescentPlannerNode::onGoal(
    const geometry_msgs::msg::PoseStamped &msg) {
  goal_stamped_ = msg;
}

void GradientDescentPlannerNode::onObstacles(
    const robot_interfaces::msg::ObstacleArray &msg) {
  obstacles_ = msg.obstacles;
}

// ── Timer callback
// ────────────────────────────────────────────────────────────

void GradientDescentPlannerNode::update() {
  if (!projection_) {
    RCLCPP_WARN_ONCE(get_logger(),
                     "No projection strategy set — tick ignored.");
    return;
  }

  if (!current_pose_.has_value() || !goal_stamped_.has_value()) {
    return;
  }

  // 1. Project goal into the robot frame
  const auto goal_local_opt =
      projection_->project(*current_pose_, *goal_stamped_);

  if (!goal_local_opt.has_value()) {
    return; // Projection unavailable this cycle (e.g. tf2 not ready)
  }

  const Point2D &goal_local = *goal_local_opt;

  // 2. Check goal tolerance
  if (isWithinGoalTolerance(goal_local)) {
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000 /*ms*/,
                         "Goal reached — publishing zero vector.");
    publishZeroVector();
    return;
  }

  // 3. Compute gradients
  const PotentialFieldParams params = readParams();

  const Point2D grad_goal =
      PotentialFieldComputer::goalGradient(goal_local, params);
  const Point2D grad_obs =
      PotentialFieldComputer::obstacleGradient(obstacles_, goal_local, params);

  const Point2D total_grad{(grad_goal.x + grad_obs.x),
                           (grad_goal.y + grad_goal.y)};

  RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 500 /*ms*/,
      "[GOAL local] x=%.3f y=%.3f | "
      "[GRAD] goal=(%.3f, %.3f) obs=(%.3f, %.3f) total=(%.3f, %.3f)",
      goal_local.x, goal_local.y, grad_goal.x, grad_goal.y, grad_obs.x,
      grad_obs.y, total_grad.x, total_grad.y);

  // const Point2D cmd = PotentialFieldComputer::normalizedGradientDescent(
  //     *current_pose_, total_grad, params.step_size);

  const Point2D cmd{-(grad_goal.x + grad_obs.x), -(grad_goal.y + grad_obs.y)};

  RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 500 /*ms*/,
                       "[CMD] fx=%.3f fy=%.3f", cmd.x, cmd.y);

  publishVector(cmd);
}

// ── Helpers
// ───────────────────────────────────────────────────────────────────

PotentialFieldParams GradientDescentPlannerNode::readParams() const {
  return PotentialFieldParams{get_parameter("repulsion_radius").as_double(),
                              get_parameter("goal_gain_near").as_double(),
                              get_parameter("goal_gain_far").as_double(),
                              get_parameter("goal_threshold").as_double(),
                              get_parameter("repulsion_gain").as_double(),
                              get_parameter("tangential_gain").as_double(),
                              get_parameter("step_size").as_double()};
}

bool GradientDescentPlannerNode::isWithinGoalTolerance(
    const Point2D &goal_local) const {
  const double tolerance = get_parameter("goal_tolerance").as_double();
  return std::hypot(goal_local.x, goal_local.y) < tolerance;
}

void GradientDescentPlannerNode::publishVector(const Point2D &vec) {
  robot_interfaces::msg::Force2D msg;
  msg.fx = static_cast<float>(vec.x);
  msg.fy = static_cast<float>(vec.y);
  cmd_pub_->publish(msg);
}

void GradientDescentPlannerNode::publishZeroVector() {
  publishVector({0.0, 0.0});
}

} // namespace potential_fields
