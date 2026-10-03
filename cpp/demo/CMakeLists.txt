# C++ demos of DOLFINx-MPC, built against an installed DOLFINx and DOLFINx-MPC:
#
#   cmake -G Ninja -B build-demo -S cpp/demo
#   cmake --build build-demo
#   mpirun -n 2 build-demo/cross_block/demo_cross_block
cmake_minimum_required(VERSION 3.21)

project(dolfinx_mpc_demos LANGUAGES C CXX)

find_package(DOLFINX REQUIRED)
find_package(DOLFINX_MPC REQUIRED)
include(CTest)

add_subdirectory(cross_block)
