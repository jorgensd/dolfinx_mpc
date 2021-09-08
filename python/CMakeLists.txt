cmake_minimum_required(VERSION 3.10)

PROJECT(dolfinx_mpc_pybind11)

find_package(pybind11 REQUIRED CONFIG HINTS ${PYBIND11_DIR} ${PYBIND11_ROOT}
  $ENV{PYBIND11_DIR} $ENV{PYBIND11_ROOT})

find_package(DOLFINX_MPC REQUIRED)
find_package(Basix REQUIRED)

# Set C++ standard before finding pybind11
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Create the binding library
pybind11_add_module(cpp SHARED
   dolfinx_mpc/mpc.cpp
   dolfinx_mpc/dolfinx_mpc.cpp)

target_link_libraries(cpp PRIVATE pybind11::module dolfinx_mpc)
target_link_libraries(cpp PUBLIC dolfinx)
target_include_directories(cpp PUBLIC dolfinx)

# Add strict compiler flags
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-Wall -Werror -pedantic" HAVE_PEDANTIC)
if (HAVE_PEDANTIC)
  target_compile_options(cpp PRIVATE -Wall;-Werror;-pedantic)
endif()


# Add to CMake search path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})

# Add DOLFINX libraries and other config
find_package(DOLFINX REQUIRED)
if (DOLFINX_FOUND)
   target_include_directories(cpp PRIVATE ${DOLFINX_INCLUDE_DIR})

# Find petsc4py through python
execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c "import petsc4py; print(petsc4py.get_include())"
  OUTPUT_VARIABLE PETSC4PY_INCLUDE_DIR
  RESULT_VARIABLE PETSC4PY_NOT_FOUND
  ERROR_QUIET
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )
target_include_directories(cpp PRIVATE ${PETSC4PY_INCLUDE_DIR})

endif()

