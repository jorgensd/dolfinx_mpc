#------------------------------------------------------------------------------
# Top level CMakeLists.txt file for DOLFIN
cmake_minimum_required(VERSION 3.10)

#------------------------------------------------------------------------------
# Set project name and version number

project(DOLFINX_MPC VERSION 2019.2.9.99)

#------------------------------------------------------------------------------
# Set CMake options, see `cmake --help-policy CMP000x`

cmake_policy(VERSION 3.10)
if (POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()
if (POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

#------------------------------------------------------------------------------
# Use C++17
set(CMAKE_CXX_STANDARD 17)

# Require C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Do not enable compler-specific extensions
set(CMAKE_CXX_EXTENSIONS OFF)

#------------------------------------------------------------------------------
# General configuration

# Set location of our FindFoo.cmake modules
set(CMAKE_MODULE_PATH "${DOLFINX_MPC_SOURCE_DIR}/cmake/modules")

# Make sure CMake uses the correct DOLFINConfig.cmake for tests and demos
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${CMAKE_CURRENT_BINARY_DIR}/dolfinx_mpc)

#------------------------------------------------------------------------------
# Configurable options for how we want to build

include(FeatureSummary)

option(BUILD_SHARED_LIBS "Build DOLFIN with shared libraries." ON)
add_feature_info(BUILD_SHARED_LIBS BUILD_SHARED_LIBS "Build DOLFIN with shared libraries.")

# Add shared library paths so shared libs in non-system paths are found
option(CMAKE_INSTALL_RPATH_USE_LINK_PATH "Add paths to linker search and installed rpath." ON)
add_feature_info(CMAKE_INSTALL_RPATH_USE_LINK_PATH CMAKE_INSTALL_RPATH_USE_LINK_PATH "Add paths to linker search and installed rpath.")

# Check for required package Eigen3
find_package(Eigen3 3.2.90 REQUIRED)
set_package_properties(Eigen3 PROPERTIES TYPE REQUIRED
  DESCRIPTION "Lightweight C++ template library for linear algebra"
  URL "http://eigen.tuxfamily.org")

# Check for required package DOLFINX
find_package(DOLFINX REQUIRED)
set_package_properties(DOLFINX PROPERTIES TYPE REQUIRED
    DESCRIPTION "New generation Dynamic Object-oriented Library for - FINite element computation"
    URL "https://github.com/FEniCS/dolfinx"
    PURPOSE "Main dependency of library")

feature_summary(WHAT ALL)
#------------------------------------------------------------------------------
# Installation of DOLFIN_MPC library

add_subdirectory(dolfinx_mpc)

# Add executable
add_executable(Test test/main.cpp)
target_link_libraries(Test PUBLIC dolfinx dolfinx_mpc)

#------------------------------------------------------------------------------
# Generate and install helper file dolfinx_mpc.conf

# FIXME: Can CMake provide the library path name variable?
if (APPLE)
  set(OS_LIBRARY_PATH_NAME "DYLD_LIBRARY_PATH")
else()
  set(OS_LIBRARY_PATH_NAME "LD_LIBRARY_PATH")
endif()

# FIXME: not cross-platform compatible
# Create and install dolfinx.conf file
configure_file(${DOLFINX_MPC_SOURCE_DIR}/cmake/templates/dolfinx_mpc.conf.in
               ${CMAKE_BINARY_DIR}/dolfinx_mpc.conf @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/dolfinx_mpc.conf
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/dolfinx_mpc
        COMPONENT Development)
