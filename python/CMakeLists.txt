cmake_minimum_required(VERSION 3.19)

PROJECT(dolfinx_mpc_pybind11)

get_filename_component(_parent_exe "$ENV{_}" NAME)
if ("${_parent_exe}" MATCHES "^python")
    # $_ is generally the parent command
    # which is the Python interpreter that launched the build
    # this lets us skip out of build isolation for our
    # detection of dolfinx, petsc4py
    # I have no idea how stable this is!
    MESSAGE(STATUS "Found unisolated Python ${parent_python}")
    set(parent_python "$ENV{_}")
else()
    # default path:
    set(parent_python ${Python3_EXECUTABLE})
endif()
MESSAGE(STATUS "Using ${parent_python} to detect dolfinx, petsc4py")

# We can use newer FindPython because scikit-build-core backports it
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED CONFIG HINTS ${PYBIND11_DIR} ${PYBIND11_ROOT}
  $ENV{PYBIND11_DIR} $ENV{PYBIND11_ROOT})

find_package(DOLFINX_MPC REQUIRED)
MESSAGE(STATUS "Found dolfinx_mpc ${DOLFINX_MPC_VERSION}")
MESSAGE(STATUS "Found dolfinx ${DOLFINX_VERSION}")

# Set C++ standard before finding pybind11
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Create the binding library
pybind11_add_module(cpp SHARED
   dolfinx_mpc/mpc.cpp
   dolfinx_mpc/dolfinx_mpc.cpp)

target_compile_definitions(cpp PRIVATE VERSION_INFO=${SKBUILD_PROJECT_VERSION})
target_link_libraries(cpp PRIVATE pybind11::module)
target_link_libraries(cpp PUBLIC dolfinx_mpc)

# Get python include-dirs
execute_process(
  COMMAND ${parent_python} -c "import dolfinx.wrappers, sys; sys.stdout.write(str(dolfinx.wrappers.get_include_path()))"
  OUTPUT_VARIABLE DOLFINX_PY_DIR
  RESULT_VARIABLE DOLFINX_PY_COMMAND_RESULT OUTPUT_STRIP_TRAILING_WHITESPACE)

if (DOLFINX_PY_DIR)
  message(STATUS "Adding ${DOLFINX_PY_DIR} to include directories")
  target_include_directories(cpp PRIVATE ${DOLFINX_PY_DIR})
endif()

# Add strict compiler flags
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-Wall -Werror -pedantic" HAVE_PEDANTIC)
if (HAVE_PEDANTIC)
  target_compile_options(cpp PRIVATE -Wall;-Werror;-pedantic)
endif()

# Find petsc4py through python
execute_process(
  COMMAND ${parent_python} -c "import petsc4py; print(petsc4py.get_include())"
  OUTPUT_VARIABLE PETSC4PY_INCLUDE_DIR
  RESULT_VARIABLE PETSC4PY_NOT_FOUND
  ERROR_QUIET
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )
target_include_directories(cpp PRIVATE ${PETSC4PY_INCLUDE_DIR})

install(TARGETS cpp LIBRARY DESTINATION "dolfinx_mpc")
