cmake_minimum_required(VERSION 3.8)
project(potential_fields)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ── Dependencias ──────────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(robot_interfaces REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# ── Biblioteca interna: strategies + matemática pura ─────────────────────────
# Agrupa todo lo que no es main() en una STATIC lib reutilizada por
# ambos executables sin duplicar compilación.
add_library(potential_fields_core STATIC
  src/manual_projection.cpp
  src/tf2_projection.cpp
  src/potential_field_computer.cpp
  src/gradient_descent_planner_node.cpp
)

target_include_directories(potential_fields_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)

target_compile_features(potential_fields_core PUBLIC cxx_std_17)

ament_target_dependencies(potential_fields_core
  rclcpp
  geometry_msgs
  robot_interfaces
  tf2
  tf2_ros
  tf2_geometry_msgs
)

# ── Ejecutable: simulación (ManualProjection) ─────────────────────────────────
add_executable(field_planner_sim src/sim_node_main.cpp)
target_link_libraries(field_planner_sim potential_fields_core)
target_compile_features(field_planner_sim PUBLIC cxx_std_17)
install(TARGETS field_planner_sim DESTINATION lib/${PROJECT_NAME})

# ── Ejecutable: robot real (Tf2Projection) ────────────────────────────────────
add_executable(field_planner_real src/real_node_main.cpp)
target_link_libraries(field_planner_real potential_fields_core)
target_compile_features(field_planner_real PUBLIC cxx_std_17)
install(TARGETS field_planner_real DESTINATION lib/${PROJECT_NAME})

# ── Tests ─────────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
