cmake_minimum_required(VERSION 3.5.0)
project(homebrew-neural-network VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_executable(homebrew-neural-network src/main.cc src/mlp.cc)
