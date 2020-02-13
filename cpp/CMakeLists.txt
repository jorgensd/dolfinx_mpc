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

# Make sure CMake uses the correct DOLFINConfig.cmake for tests and demos
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${CMAKE_CURRENT_BINARY_DIR}/dolfinx_mpc)

#------------------------------------------------------------------------------
# configurable options for how we want to build

include(FeatureSummary)

option(BUILD_SHARED_LIBS "Build DOLFINX_MPC with shared libraries." ON)
add_feature_info(BUILD_SHARED_LIBS BUILD_SHARED_LIBS "Build DOLFINX_MPC with shared libraries.")

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

# Delcare the library (target)

#------------------------------------------------------------------------------
include(GNUInstallDirs)

#------------------------------------------------------------------------------

# NOTE: This modifies a file in the source directory, which probably isn't good
# configure_file(DolfinxMPC.h.in DolfinxMPC.h)

#------------------------------------------------------------------------------
add_library(dolfinx_mpc "")  # The "" is needed for older CMake. Remove later.

install(FILES dolfinx_mpc.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dolfinx_mpc COMPONENT Development)
install(FILES MultiPointConstraint.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dolfinx_mpc COMPONENT Development)

# Add source files to the target
target_sources(dolfinx_mpc PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/MultiPointConstraint.cpp
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
# Eigen3, MPI and PETSC
# See DOLFINX for explanation of this variable
set(DOLFINX_EIGEN_MAX_ALIGN_BYTES "32" CACHE STRING "\
Minimum alignment in bytes used for Eigen data structures. Set to 32 for \
compatibility with AVX user-compiled code and 64 for AVX-512 user-compiled \
code. Set to 0 for ideal alignment according to -march. Note that if an architecture \
flag (e.g. \"-march=skylake-avx512\") is set for DOLFIN, Eigen will use the \
appropriate ideal alignment instead if it is stricter. Otherwise, the value \
of this variable will be used by Eigen for the alignment of all data structures.\\
")

target_include_directories(dolfinx_mpc SYSTEM PUBLIC ${EIGEN3_INCLUDE_DIR})
target_compile_definitions(dolfinx_mpc PUBLIC "EIGEN_MAX_ALIGN_BYTES=${DOLFINX_EIGEN_MAX_ALIGN_BYTES}")


# MPI
target_link_libraries(dolfinx_mpc PUBLIC MPI::MPI_CXX)

# petsc
target_link_libraries(dolfinx_mpc PUBLIC PETSC::petsc)
target_link_libraries(dolfinx_mpc PRIVATE PETSC::petsc_static)

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

# Generate CMake config files (DOLFINX_MPCConfig{,Version}.cmake)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(${CMAKE_BINARY_DIR}/dolfinx_mpc/DOLFINX_MPCConfigVersion.cmake
  VERSION ${DOLFINX_MPC_VERSION}
  COMPATIBILITY ExactVersion)

configure_package_config_file(${DOLFINX_MPC_SOURCE_DIR}/DOLFINX_MPCConfig.cmake.in
${CMAKE_BINARY_DIR}/dolfinx_mpc/DOLFINX_MPCConfig.cmake
INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/dolfinx_mpc)

# Install CMake helper files
install(
  FILES
  ${CMAKE_BINARY_DIR}/dolfinx_mpc/DOLFINX_MPCConfig.cmake
  ${CMAKE_BINARY_DIR}/dolfinx_mpc/DOLFINX_MPCConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/dolfinx_mpc
  COMPONENT Development)

#------------------------------------------------------------------------------
# Generate pkg-config file and install it

# Define packages that should be required by pkg-config file
set(PKG_REQUIRES "")

# Get link libraries and includes
get_target_property(PKGCONFIG_DOLFINX_TARGET_LINK_LIBRARIES dolfinx_mpc INTERFACE_LINK_LIBRARIES)
get_target_property(PKGCONFIG_DOLFINX_INCLUDE_DIRECTORIES dolfinx_mpc INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)

# Add imported targets to lists for creating pkg-config file
set(PKGCONFIG_DOLFINX_LIBS)
foreach(_target ${PKGCONFIG_DOLFINX_TARGET_LINK_LIBRARIES})

  if ("${_target}" MATCHES "^[^<>]+$")  # Skip "$<foo...>", which we get with static libs

    if ("${_target}" MATCHES "^.*::.*$")
      # Get include paths
      get_target_property(_inc_dirs ${_target} INTERFACE_INCLUDE_DIRECTORIES)
      if (_inc_dirs)
        list(APPEND PKGCONFIG_DOLFINX_INCLUDE_DIRECTORIES ${_inc_dirs})
      endif()

      # Get libraries
      get_target_property(_libs ${_target} INTERFACE_LINK_LIBRARIES)
      if (_libs)
        list(APPEND PKGCONFIG_DOLFINX_LIBS ${_libs})
      endif()

    else()
      # 'regular' libs, i.e. not imported targets
      list(APPEND PKGCONFIG_DOLFINX_LIBS ${_target})
    endif()

    # Special handling for compiled Boost imported targets
    if (("${_target}" MATCHES "^.*Boost::.*$") AND NOT "${_target}" STREQUAL "Boost::boost")
      get_target_property(_libs ${_target} IMPORTED_LOCATION_RELEASE)
      if (_libs)
        list(APPEND PKGCONFIG_DOLFINX_LIBS ${_libs})
      endif()
    endif()

  endif()

endforeach()

# Join include lists and remove duplicates
list(REMOVE_DUPLICATES PKGCONFIG_DOLFINX_INCLUDE_DIRECTORIES)
list(REMOVE_DUPLICATES PKGCONFIG_DOLFINX_LIBS)

# Convert include dirs to -I<incdir> form
foreach(_inc_dir ${PKGCONFIG_DOLFINX_INCLUDE_DIRECTORIES})
  set(PKG_INCLUDES "-I${_inc_dir} ${PKG_INCLUDES}")
endforeach()

# Get dolfinx definitions
get_target_property(PKG_DOLFINX_DEFINITIONS dolfinx_mpc INTERFACE_COMPILE_DEFINITIONS)
set(PKG_DEFINITIONS)
foreach(_def ${PKG_DOLFINX_DEFINITIONS})
  set(PKG_DEFINITIONS "${PKG_DEFINITIONS} -D${_def}")
endforeach()

# Convert compiler flags and definitions into space separated strings
string(REPLACE ";" " " PKG_CXXFLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE ";" " " PKG_LINKFLAGS "${CMAKE_EXE_LINKER_FLAGS}")

# Convert libraries to -L<libdir> -l<lib> form
foreach(_lib ${PKGCONFIG_DOLFINX_LIBS})
  # Add -Wl,option directives
  if ("${_lib}" MATCHES "-Wl,[^ ]*")
    set(PKG_LINKFLAGS "${_lib} ${PKG_LINKFLAGS}")
  else()
    string(REGEX REPLACE "(.?:?/[^ ]*)/lib([^ ]*)\\.(a|so|dylib|dll)" "-L\\1 -l\\2"
      _linkflags
      "${_lib}"
      )

    # Add libraries that matches the form -L<libdir> -l<lib>
    if ("${_linkflags}" MATCHES "-L.+ -l.+")
      set(PKG_LINKFLAGS "${_linkflags} ${PKG_LINKFLAGS}")
    endif()
  endif()
endforeach()

# Remove duplicated link flags
separate_arguments(PKG_LINKFLAGS)
list(REMOVE_DUPLICATES PKG_LINKFLAGS)
string(REPLACE ";" " " PKG_LINKFLAGS "${PKG_LINKFLAGS}")

# Add additional link flags
foreach(_linkflag ${DOLFINX_LINK_FLAGS})
  set(PKG_LINKFLAGS "${PKG_LINKFLAGS} ${_linkflag}")
endforeach()

# Configure and install pkg-config file
configure_file(${DOLFINX_MPC_SOURCE_DIR}/dolfinx_mpc.pc.in ${CMAKE_BINARY_DIR}/dolfinx_mpc.pc @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/dolfinx_mpc.pc
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
  COMPONENT Development
  )
#------------------------------------------------------------------------------
