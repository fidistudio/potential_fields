#pragma once

#include "potential_fields/goal_projection_strategy.hpp"

#include <vector>

#include "robot_interfaces/msg/obstacle.hpp"

namespace potential_fields {

/**
 * @brief Parameters governing the potential field computation.
 *
 * Kept as a plain struct so they can be read from ROS params once
 * per timer tick and passed in — no ROS dependency inside the math.
 */
struct PotentialFieldParams {
  double repulsion_radius{3.0};
  double goal_gain_near{1.0};
  double goal_gain_far{1.0};
  double goal_threshold{3.0};
  double repulsion_gain{1.0};
  double tangential_gain{2.0};
};

/**
 * @brief Pure math class: computes potential field gradients.
 *
 * No ROS types, no node references — only geometry and physics.
 * This makes the computation unit-testable without spinning a ROS context.
 *
 * Single Responsibility: knows only how to compute gradients.
 */
class PotentialFieldComputer {
public:
  /**
   * @brief Gradient of the attractive goal potential.
   *
   * Uses a conic well far from the goal and a parabolic well close to it,
   * switching at goal_threshold to avoid overshooting.
   *
   * @param goal_local  Goal position in the robot frame [m].
   * @param params      Field parameters.
   * @return            Attractive gradient vector.
   */
  [[nodiscard]] static Point2D goalGradient(const Point2D &goal_local,
                                            const PotentialFieldParams &params);

  /**
   * @brief Gradient of the combined repulsive + tangential obstacle potential.
   *
   * For each obstacle within repulsion_radius:
   *   - Classical repulsion pushes the robot away.
   *   - Tangential term steers the robot around the obstacle toward the goal,
   *     breaking symmetry so the robot doesn't get stuck head-on.
   *
   * @param obstacles   Obstacles expressed in the robot frame.
   * @param goal_local  Goal position in the robot frame (used for tangential
   * direction).
   * @param params      Field parameters.
   * @return            Total repulsive + tangential gradient vector.
   */
  [[nodiscard]] static Point2D obstacleGradient(
      const std::vector<robot_interfaces::msg::Obstacle> &obstacles,
      const Point2D &goal_local, const PotentialFieldParams &params);
};

} // namespace potential_fields
