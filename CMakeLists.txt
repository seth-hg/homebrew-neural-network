cmake_minimum_required(VERSION 3.5.0)
project(homebrew-neural-network VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP REQUIRED)
add_compile_options(${OpenMP_CXX_FLAGS})

add_executable(homebrew-neural-network src/main.cc src/mlp.cc)
target_link_libraries(homebrew-neural-network ${OpenMP_CXX_LIBRARIES})

if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
target_compile_options(homebrew-neural-network PRIVATE "-mavx2")
endif ()