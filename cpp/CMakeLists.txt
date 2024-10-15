#------------------------------------------------------------------------------
# Top level CMakeLists.txt file for DOLFINX_MPC
cmake_minimum_required(VERSION 3.21)

#------------------------------------------------------------------------------
# Set project name and version number
project(DOLFINX_MPC VERSION "0.9.0.0")

#------------------------------------------------------------------------------
# General configuration
# Make sure CMake uses the correct DOLFINConfig.cmake for tests and demos
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${CMAKE_CURRENT_BINARY_DIR}/dolfinx_mpc)

#------------------------------------------------------------------------------
# configurable options for how we want to build

include(FeatureSummary)

option(BUILD_SHARED_LIBS "Build DOLFINX_MPC with shared libraries." ON)
add_feature_info(BUILD_SHARED_LIBS BUILD_SHARED_LIBS "Build DOLFINX_MPC with shared libraries.")

option(CMAKE_INSTALL_RPATH_USE_LINK_PATH "Add paths to linker search and installed rpath." ON)
add_feature_info(CMAKE_INSTALL_RPATH_USE_LINK_PATH CMAKE_INSTALL_RPATH_USE_LINK_PATH "Add paths to linker search and installed rpath.")

# Check for required package DOLFINX
find_package(DOLFINX 0.9.0.0 REQUIRED)
set_package_properties(DOLFINX PROPERTIES TYPE REQUIRED
    DESCRIPTION "New generation Dynamic Object-oriented Library for - FINite element computation"
    URL "https://github.com/FEniCS/dolfinx"
    PURPOSE "Main dependency of library")
feature_summary(WHAT ALL)


# Installation of DOLFIN_MPC library

# Declare the library (target)

add_library(dolfinx_mpc)
target_compile_features(dolfinx_mpc PUBLIC cxx_std_20)

# dolfinx gives us transitive dependency on mpi, petsc, basix, ufcx
# without us needing to reimplement detection/dependency
target_link_libraries(dolfinx_mpc PUBLIC dolfinx)


#------------------------------------------------------------------------------
include(GNUInstallDirs)

#------------------------------------------------------------------------------

install(FILES dolfinx_mpc.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dolfinx_mpc COMPONENT Development)

install(FILES assemble_utils.h mpi_utils.h ContactConstraint.h utils.h MultiPointConstraint.h SlipConstraint.h PeriodicConstraint.h assemble_matrix.h assemble_vector.h lifting.h mpc_helpers.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dolfinx_mpc COMPONENT Development)
# Add source files to the target
target_sources(dolfinx_mpc PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/assemble_matrix.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/assemble_vector.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/assemble_utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mpi_utils.cpp
  )

# Set target include location (for build and installed)
target_include_directories(dolfinx_mpc PUBLIC
                           $<INSTALL_INTERFACE:include>
                           "$<BUILD_INTERFACE:${DOLFINX_MPC_SOURCE_DIR};${DOLFINX_MPC_SOURCE_DIR}/dolfinx_mpc>")

# Set target properties
set_target_properties(dolfinx_mpc PROPERTIES
  VERSION ${DOLFINX_MPC_VERSION}
  SOVERSION ${DOLFINX_MPC_VERSION_MAJOR}.${DOLFINX_MPC_VERSION_MINOR})


# Add version to definitions (public)
target_compile_definitions(dolfinx_mpc PUBLIC DOLFINX_MPC_VERSION="${DOLFINX_MPC_VERSION}")


#------------------------------------------------------------------------------
# Install dolfinx library and header files
install(TARGETS dolfinx_mpc
  EXPORT DOLFINX_MPCTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT RuntimeExecutables
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Development
  )

# Generate DOLFINTargets.cmake
install(EXPORT DOLFINX_MPCTargets DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/dolfinx_mpc)

include(CMakePackageConfigHelpers)

configure_package_config_file(${DOLFINX_MPC_SOURCE_DIR}/DOLFINX_MPCConfig.cmake.in
${CMAKE_BINARY_DIR}/dolfinx_mpc/DOLFINX_MPCConfig.cmake
INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/dolfinx_mpc)

# Install CMake helper files
install(
  FILES
  ${CMAKE_BINARY_DIR}/dolfinx_mpc/DOLFINX_MPCConfig.cmake
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/dolfinx_mpc
  COMPONENT Development)

#------------------------------------------------------------------------------
