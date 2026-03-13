cmake_minimum_required(VERSION 3.21)

PROJECT(dolfinx_mpc_nanobind)

find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Detect the installed nanobind package and import it into CMake
option(DETECT_NANOBIND_DIR "Disable if python3 -mnanobind doesn't work (e.g. cross compilation). If OFF, nanobind must be on CMAKE_PREFIX_PATH." ON)
if (DETECT_NANOBIND_DIR)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
  list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
endif()
find_package(nanobind CONFIG REQUIRED)

option(BASIX_HINT "Hint for finding BASIX; skips auto-detection where it might not work (cross-compiling)." "")
if(NOT BASIX_HINT)
  execute_process(
    COMMAND
    ${Python3_EXECUTABLE} -c "import os, sys, basix; sys.stdout.write(os.path.dirname(basix.__file__))"
    OUTPUT_VARIABLE BASIX_PY_DIR
    RESULT_VARIABLE BASIX_PY_COMMAND_RESULT
    ERROR_VARIABLE BASIX_ERROR_OUT OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  set(BASIX_HINT ${BASIX_PY_DIR})
endif()

find_package(Basix REQUIRED CONFIG HINTS ${BASIX_HINT})
if (Basix_FOUND)
  message(STATUS "Found Basix at ${Basix_DIR}")
endif()
find_package(DOLFINX REQUIRED CONFIG)
if (DOLFINX_FOUND)
  message(STATUS "Found DOLFINx at ${DOLFINX_DIR}")
endif()


# Add DOLFINx_mpc libraries
find_package(DOLFINX_MPC REQUIRED CONFIG)
if (DOLFINX_MPC_FOUND)
  message(STATUS "Found DOLFINx_MPC at ${DOLFINX_MPC_DIR}")
endif()

# Create the binding library nanobind handles its own calls to
# target_link_libraries
nanobind_add_module(
  cpp
  NOMINSIZE
  MODULE
  src/dolfinx_mpc/dolfinx_mpc.cpp
  src/dolfinx_mpc/mpc.cpp
)

target_compile_definitions(cpp PRIVATE cxx_std_20)
target_link_libraries(cpp PRIVATE dolfinx_mpc)

# Check for petsc4py
option(PETSC4PY_INCLUDE_DIR "Manually set include directory for petsc4py; skips auto-detection where it might not work (cross-compiling)." "")
if (NOT PETSC4PY_INCLUDE_DIR)
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c
            "import petsc4py; print(petsc4py.get_include())"
    OUTPUT_VARIABLE PETSC4PY_INCLUDE_DIR
    RESULT_VARIABLE PETSC4PY_COMMAND_RESULT
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )
  
  if(NOT PETSC4PY_COMMAND_RESULT)
    message(STATUS "Found petsc4py include directory at ${PETSC4PY_INCLUDE_DIR}")
  else()
    message(FATAL_ERROR "petsc4py not found.")
  endif()
else()
    message(STATUS "Using petsc4py include directory at ${PETSC4PY_INCLUDE_DIR}")
endif()
target_include_directories(cpp PRIVATE ${PETSC4PY_INCLUDE_DIR})
target_compile_definitions(cpp PRIVATE HAS_PETSC4PY)


# Get python include-dirs
option(DOLFINX_PY_DIR "Include directory for dolfinx python wrappers; skips auto-detection where it might not work (cross-compiling)." "")
if (NOT DOLFINX_PY_DIR)
  find_package(MPI 3 REQUIRED)
  execute_process(
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${Python3_EXECUTABLE} -c "from mpi4py import MPI; import dolfinx, sys; sys.stdout.write(str(dolfinx.get_include()))"
    OUTPUT_VARIABLE DOLFINX_PY_DIR
    RESULT_VARIABLE DOLFINX_PY_COMMAND_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )
endif()

if (DOLFINX_PY_DIR)
  message(STATUS "Adding ${DOLFINX_PY_DIR} to include directories")
  target_include_directories(cpp PRIVATE ${DOLFINX_PY_DIR})
endif()


set_target_properties(cpp PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

install(TARGETS cpp DESTINATION dolfinx_mpc)
