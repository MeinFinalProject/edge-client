cmake_minimum_required(VERSION 3.21)

project(vision_native LANGUAGES CXX)

include(CTest)

option(VISION_RUNTIME_BUILD_TOOLS "Build native research/diagnostic tools" ON)

add_subdirectory(vision_runtime)
add_subdirectory(edge_app)

if(VISION_RUNTIME_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
