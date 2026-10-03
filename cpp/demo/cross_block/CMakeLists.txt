cmake_minimum_required(VERSION 3.21)

project(demo_cross_block LANGUAGES C CXX)

if(NOT TARGET dolfinx::dolfinx)
  find_package(DOLFINX REQUIRED)
endif()
if(NOT TARGET dolfinx_mpc)
  find_package(DOLFINX_MPC REQUIRED)
endif()

# The DOLFINx demo helper compiles the forms for the PETSc scalar type
if(DOLFINX_DIR)
  list(APPEND CMAKE_MODULE_PATH ${DOLFINX_DIR})
endif()
include(DolfinxDemo)
if(DOLFINX_DIR)
  list(REMOVE_ITEM CMAKE_MODULE_PATH ${DOLFINX_DIR})
endif()

dolfinx_add_demo(cross_block UFL cross_block.py)
target_link_libraries(demo_cross_block PRIVATE dolfinx_mpc)
