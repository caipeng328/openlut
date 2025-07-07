cmake_minimum_required(VERSION 3.16.3)
project(CPPLUT LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置 Eigen 路径
set(EIGEN_DIR "/usr/include/eigen3")
include_directories(${EIGEN_DIR})

# 项目源码目录
include_directories(${CMAKE_SOURCE_DIR}/cpp)

# 查找 pybind11（推荐方式）
find_package(pybind11 REQUIRED)

# 添加 pybind11 模块（先创建目标）
pybind11_add_module(ApplyLUT cpp/ApplyLUT.cpp)

# 查找 OpenMP
find_package(OpenMP REQUIRED)

# 如果找到 OpenMP，链接到目标
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
    target_link_libraries(ApplyLUT PRIVATE OpenMP::OpenMP_CXX)
endif()
