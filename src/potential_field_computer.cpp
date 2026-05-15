#include "potential_fields/potential_field_computer.hpp"

#include <cmath>

namespace potential_fields {

// ── Goal gradient
// ─────────────────────────────────────────────────────────────

Point2D
PotentialFieldComputer::goalGradient(const Point2D &goal_local,
                                     const PotentialFieldParams &params) {
  // delta points from goal toward robot origin (gradient ascent direction)
  const double dx = -goal_local.x;
  const double dy = -goal_local.y;
  const double distance = std::hypot(dx, dy);

  if (distance < 1e-6) {
    return {0.0, 0.0};
  }

  // Parabolic well (near): gradient magnitude grows with distance → strong pull
  if (distance < params.goal_threshold) {
    return {params.goal_gain_near * dx, params.goal_gain_near * dy};
  }

  // Conic well (far): constant magnitude → avoids runaway velocities
  return {params.goal_gain_far * dx / distance,
          params.goal_gain_far * dy / distance};
}

// ── Obstacle gradient
// ─────────────────────────────────────────────────────────

Point2D PotentialFieldComputer::obstacleGradient(
    const std::vector<robot_interfaces::msg::Obstacle> &obstacles,
    const Point2D &goal_local, const PotentialFieldParams &params) {
  double total_x = 0.0;
  double total_y = 0.0;

  // Goal direction in robot frame (robot is at origin)
  const double goal_norm = std::hypot(goal_local.x, goal_local.y);
  if (goal_norm < 1e-6) {
    return {0.0, 0.0}; // No goal direction → cannot compute tangential term
  }

  // Unit vector pointing toward the goal
  const double g_hat_x = goal_local.x / goal_norm;
  const double g_hat_y = goal_local.y / goal_norm;

  // Tangential unit vector (perpendicular to goal direction, CCW)
  const double t_hat_base_x = -g_hat_y;
  const double t_hat_base_y = g_hat_x;

  for (const auto &obstacle : obstacles) {
    // delta = robot_origin - obstacle  (points away from obstacle)
    const double delta_x = -obstacle.x;
    const double delta_y = -obstacle.y;
    const double d = std::hypot(delta_x, delta_y);

    if (d < 1e-6 || d > params.repulsion_radius) {
      continue;
    }

    // ── Choose consistent tangential side ────────────────────────────────────
    // d_hat points from obstacle toward robot
    const double d_hat_x = delta_x / d;
    const double d_hat_y = delta_y / d;

    // Flip t_hat so it always points to the same side as d_hat
    const double dot = t_hat_base_x * d_hat_x + t_hat_base_y * d_hat_y;
    const double sign = (dot >= 0.0) ? 1.0 : -1.0;
    const double t_hat_x = sign * t_hat_base_x;
    const double t_hat_y = sign * t_hat_base_y;

    // ── Classical repulsion
    // ─────────────────────────────────────────────────── grad_rep = -k_rep *
    // (1/d - 1/d0) * (1/d³) * delta
    const double rep_scalar = -params.repulsion_gain *
                              (1.0 / d - 1.0 / params.repulsion_radius) /
                              (d * d * d);

    const double grad_rep_x = rep_scalar * delta_x;
    const double grad_rep_y = rep_scalar * delta_y;

    // ── Tangential term
    // ─────────────────────────────────────────────────────── td = projection
    // of delta onto t_hat
    const double td = t_hat_x * delta_x + t_hat_y * delta_y;

    // grad_tang = -k_tang * [(1/d³)*td*delta + (1/d - 1/d0)*t_hat]
    const double tang_scalar = -params.tangential_gain / (d * d * d);
    const double edge_factor = (1.0 / d - 1.0 / params.repulsion_radius);

    const double grad_tang_x =
        tang_scalar * td * delta_x +
        (-params.tangential_gain) * edge_factor * t_hat_x;
    const double grad_tang_y =
        tang_scalar * td * delta_y +
        (-params.tangential_gain) * edge_factor * t_hat_y;

    total_x += grad_rep_x + grad_tang_x;
    total_y += grad_rep_y + grad_tang_y;
  }

  return {total_x, total_y};
}

} // namespace potential_fields
