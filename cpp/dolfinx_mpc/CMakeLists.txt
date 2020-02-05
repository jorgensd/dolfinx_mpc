#------------------------------------------------------------------------------
include(GNUInstallDirs)

#------------------------------------------------------------------------------
# Configure the dolfinx_mpc/common/version.h file

# NOTE: This modifies a file in the source directory, which probably isn't good
configure_file(DolfinxMPC.h.in DolfinxMPC.h)

#------------------------------------------------------------------------------
# Delcare the library (target)

add_library(dolfinx_mpc "")  # The "" is needed for older CMake. Remove later.

#------------------------------------------------------------------------------
# Add source files to the target

set(DOLFINX_MPC_DIRS mpc)

install(FILES dolfinx_mpc.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT Development)

# Add source to dolfinx target, and get sets of header files
foreach(DIR ${DOLFINX_MPC_DIRS})
  add_subdirectory(${DIR})
endforeach()

# Set target include location (for build and installed)
target_include_directories(dolfinx_mpc PUBLIC
                           $<INSTALL_INTERFACE:include>
                           "$<BUILD_INTERFACE:${DOLFINX_MPC_SOURCE_DIR};${DOLFINX_MPC_SOURCE_DIR}/dolfinx_mpc>")


#------------------------------------------------------------------------------
# Set target properties

set_target_properties(dolfinx_mpc PROPERTIES
  VERSION ${DOLFINX_MPC_VERSION}
  SOVERSION ${DOLFINX_MPC_VERSION_MAJOR}.${DOLFINX_MPC_VERSION_MINOR})

# Add git revision flag to the one affected file
set_source_files_properties(common/defines.cpp PROPERTIES
  COMPILE_DEFINITIONS "UFC_SIGNATURE=\"${UFC_SIGNATURE}\";DOLFINX_GIT_COMMIT_HASH=\"${GIT_COMMIT_HASH}\"")

#------------------------------------------------------------------------------
# Set compiler options and definitions

# Set 'Developer' build type flags
# target_compile_options(dolfinx PRIVATE $<$<CONFIG:Developer>:${DOLFINX_CXX_DEVELOPER_FLAGS}>)

# Set debug definitions (private)
# target_compile_definitions(dolfinx PRIVATE $<$<OR:$<CONFIG:Debug>,$<CONFIG:Developer>>:DEBUG>)

# Add version to definitions (public)
target_compile_definitions(dolfinx_mpc PUBLIC DOLFINX_MPC_VERSION="${DOLFINX_MPC_VERSION}")

#------------------------------------------------------------------------------
# Add include directories and libraries of required packages

# UFC
target_include_directories(dolfinx_mpc SYSTEM PUBLIC ${UFC_INCLUDE_DIRS})

# DOLFIN uses Eigen data structures for dense linear algebra operations. Eigen
# performs 'ideal' memory alignment based around the -march flag passed to the
# compiler.  However, because Python DOLFIN JIT compiles code at runtime, it is
# possible for the user to build shared objects with incompatible alignment
# (ABI) if they use a different -march flag than that used to originally build
# DOLFIN. DOLFINX_EIGEN_MAX_ALIGN_BYTES can be used to force alignment.
# See: https://eigen.tuxfamily.org/dox/TopicPreprocessorDirectives.html
# See: https://github.com/FEniCS/dolfinx/pull/143

# Note: The name EIGEN_MAX_ALIGN_BYTES is confusing. In practice, Eigen
# computes the ideal alignment based around -march.  If the ideal alignment is
# greater than EIGEN_MAX_ALIGN_BYTES, the ideal alignment is used. If the ideal
# alignment is less, then EIGEN_MAX_ALIGN_BYTES is used for alignment.
set(DOLFINX_EIGEN_MAX_ALIGN_BYTES "32" CACHE STRING "\
Minimum alignment in bytes used for Eigen data structures. Set to 32 for \
compatibility with AVX user-compiled code and 64 for AVX-512 user-compiled \
code. Set to 0 for ideal alignment according to -march. Note that if an architecture \
flag (e.g. \"-march=skylake-avx512\") is set for DOLFIN, Eigen will use the \
appropriate ideal alignment instead if it is stricter. Otherwise, the value \
of this variable will be used by Eigen for the alignment of all data structures.\\
")

# Eigen3
target_include_directories(dolfinx_mpc SYSTEM PUBLIC ${EIGEN3_INCLUDE_DIR})
target_compile_definitions(dolfinx_mpc PUBLIC "EIGEN_MAX_ALIGN_BYTES=${DOLFINX_EIGEN_MAX_ALIGN_BYTES}")

# Boost
target_link_libraries(dolfinx_mpc PUBLIC Boost::boost)
foreach (BOOST_PACKAGE ${DOLFINX_BOOST_COMPONENTS_PUBLIC})
  target_link_libraries(dolfinx_mpc PUBLIC "Boost::${BOOST_PACKAGE}")
endforeach()
foreach (BOOST_PACKAGE ${DOLFINX_BOOST_COMPONENTS_PRIVATE})
  target_link_libraries(dolfinx_mpc PRIVATE "Boost::${BOOST_PACKAGE}")
endforeach()

# MPI
target_link_libraries(dolfinx_mpc PUBLIC MPI::MPI_CXX)

# PETSc
target_link_libraries(dolfinx_mpc PUBLIC PETSC::petsc)
target_link_libraries(dolfinx_mpc PRIVATE PETSC::petsc_static)

# HDF5
target_compile_definitions(dolfinx_mpc PUBLIC ${HDF5_DEFINITIONS})
target_link_libraries(dolfinx_mpc PUBLIC ${HDF5_C_LIBRARIES})
target_include_directories(dolfinx_mpc SYSTEM PUBLIC ${HDF5_INCLUDE_DIRS})

# SCOTCH
target_link_libraries(dolfinx_mpc PRIVATE ${SCOTCH_LIBRARIES})
target_include_directories(dolfinx_mpc SYSTEM PRIVATE ${SCOTCH_INCLUDE_DIRS})

#------------------------------------------------------------------------------
# Optional packages

# SLEPC
if (DOLFINX_ENABLE_SLEPC AND SLEPC_FOUND)
  target_compile_definitions(dolfinx_mpc PUBLIC HAS_SLEPC)
  target_link_libraries(dolfinx_mpc PUBLIC SLEPC::slepc)
  target_link_libraries(dolfinx_mpc PRIVATE SLEPC::slepc_static)
endif()

# ParMETIS
if (DOLFINX_ENABLE_PARMETIS AND PARMETIS_FOUND)
  target_compile_definitions(dolfinx_mpc PUBLIC HAS_PARMETIS)
  target_link_libraries(dolfinx_mpc PRIVATE ${PARMETIS_LIBRARIES})
  target_include_directories(dolfinx_mpc SYSTEM PRIVATE ${PARMETIS_INCLUDE_DIRS})
endif()

# KaHIP
if (DOLFINX_ENABLE_KAHIP AND KAHIP_FOUND)
  target_compile_definitions(dolfinx_mpc PUBLIC HAS_KAHIP)
  target_link_libraries(dolfinx_mpc PRIVATE ${KAHIP_LIBRARIES})
  target_include_directories(dolfinx_mpc SYSTEM PRIVATE ${KAHIP_INCLUDE_DIRS})
endif()

#------------------------------------------------------------------------------
# Install dolfinx library and header files

install(TARGETS dolfinx_mpc
  EXPORT DOLFINX_MPCTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT RuntimeExecutables
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Development
  )

# Generate DOLFINTargets.cmake
install(EXPORT DOLFINX_MPCTargets DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/dolfinx_mpc/cmake)

# Install the header files
foreach(DIR ${DOLFINX_MPC_DIRS})
  install(FILES ${HEADERS_${DIR}} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dolfinx_mpc/${DIR}
    COMPONENT Development)
endforeach()

#------------------------------------------------------------------------------
# Generate CMake config files (DOLFINX_MPCConfig{,Version}.cmake)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(${CMAKE_BINARY_DIR}/dolfinx_mpc/DOLFINX_MPCConfigVersion.cmake
  VERSION ${DOLFINX_MPC_VERSION}
  COMPATIBILITY ExactVersion)

configure_package_config_file(${DOLFINX_MPC_SOURCE_DIR}/cmake/templates/DOLFINX_MPCConfig.cmake.in
${CMAKE_BINARY_DIR}/dolfinx_mpc/DOLFINX_MPCConfig.cmake
INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/dolfinx_mpc/cmake)

# Install CMake helper files
install(
  FILES
  ${CMAKE_SOURCE_DIR}/cmake/modules/FindPETSc.cmake
  ${CMAKE_SOURCE_DIR}/cmake/modules/FindSLEPc.cmake
  ${CMAKE_BINARY_DIR}/dolfinx_mpc/DOLFINX_MPCConfig.cmake
  ${CMAKE_BINARY_DIR}/dolfinx_mpc/DOLFINX_MPCConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/dolfinx_mpc/cmake
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

# Boost include dir (used as pkg-config variable)
get_target_property(BOOST_INCLUDE_DIR Boost::boost INTERFACE_INCLUDE_DIRECTORIES)

# Configure and install pkg-config file
configure_file(${DOLFINX_MPC_SOURCE_DIR}/cmake/templates/dolfinx_mpc.pc.in ${CMAKE_BINARY_DIR}/dolfinx_mpc.pc @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/dolfinx_mpc.pc
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
  COMPONENT Development
  )
#------------------------------------------------------------------------------
