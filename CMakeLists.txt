cmake_minimum_required(VERSION 3.5)
list(APPEND CMAKE_MODULE_PATH  "${CMAKE_SOURCE_DIR}/cmake/Modules/")
project(drl_grasping)

# Default to C11
if(NOT CMAKE_C_STANDARD)
    set(CMAKE_C_STANDARD 11)
endif()
# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif()

# Compiler options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Testing and linting
if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    ament_lint_auto_find_test_dependencies()
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install python scripts and examples
install(PROGRAMS
    src/panda_push_her/train_ddpg.py
    src/panda_push_her/train_sac.py
    DESTINATION lib/${PROJECT_NAME}
)

# Install directories
install(DIRECTORY launch worlds DESTINATION share/${PROJECT_NAME})

# Setup the project
ament_package()