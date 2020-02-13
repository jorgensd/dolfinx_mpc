cmake_minimum_required(VERSION 3.10)

PROJECT(dolfinx_mpc_pybind11)

find_package(DOLFINX_MPC REQUIRED)

# Set C++ standard before finding pybind11
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(pybind11 REQUIRED CONFIG HINTS ${PYBIND11_DIR} ${PYBIND11_ROOT}
  $ENV{PYBIND11_DIR} $ENV{PYBIND11_ROOT})

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
endif()

# Check for petsc4py
if(PETSC4PY_INCLUDE_DIRS)
  set(PETSC4PY_FIND_QUIETLY TRUE)
endif(PETSC4PY_INCLUDE_DIRS)

execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c "import petsc4py; print(petsc4py.get_include())"
  OUTPUT_VARIABLE PETSC4PY_INCLUDE_DIRS
  RESULT_VARIABLE PETSC4PY_NOT_FOUND
  ERROR_QUIET
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )

if(PETSC4PY_INCLUDE_DIRS)
  set(PETSC4PY_FOUND TRUE)
  set(PETSC4PY_INCLUDE_DIRS ${PETSC4PY_INCLUDE_DIRS} CACHE STRING "petsc4py include path")
else(PETSC4PY_INCLUDE_DIRS)
  set(PETSC4PY_FOUND FALSE)
endif(PETSC4PY_INCLUDE_DIRS)

if(PETSC4PY_FOUND)
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import petsc4py; print(petsc4py.__version__)"
    OUTPUT_VARIABLE PETSC4PY_VERSION
    RESULT_VARIABLE PETSC4PY_NOT_FOUND
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  string(REPLACE "." ";" PETSC4PY_VERSION_LIST ${PETSC4PY_VERSION})
  list(GET PETSC4PY_VERSION_LIST 0 PETSC4PY_VERSION_MAJOR)
  list(GET PETSC4PY_VERSION_LIST 1 PETSC4PY_VERSION_MINOR)
  if(NOT PETSC4PY_FIND_QUIETLY)
    message(STATUS "petsc4py version ${PETSC4PY_VERSION} found")
  endif(NOT PETSC4PY_FIND_QUIETLY)
else(PETSC4PY_FOUND)
  if(PETSC4PY_FIND_REQUIRED)
    message(FATAL_ERROR "petsc4py missing")
  endif(PETSC4PY_FIND_REQUIRED)
endif(PETSC4PY_FOUND)

mark_as_advanced(PETSC4PY_INCLUDE_DIRS, PETSC4PY_VERSION, PETSC4PY_VERSION_MAJOR, PETSC4PY_VERSION_MINOR)

if (PETSc4py_FIND_VERSION)
  # Check if version found is >= required version
  if (NOT "${PETSC4PY_VERSION}" VERSION_LESS "${PETSc4py_FIND_VERSION}")
    set(PETSC4PY_VERSION_OK TRUE)
  endif()
else()
  # No specific version requested
  set(PETSC4PY_VERSION_OK TRUE)
endif()
mark_as_advanced(PETSC4PY_VERSION_OK)

# Standard package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PETSc4py
  "PETSc4py could not be found. Be sure to set PYTHONPATH appropriately."
  PETSC4PY_INCLUDE_DIRS PETSC4PY_VERSION PETSC4PY_VERSION_OK)
if (PETSc4py_FOUND)
    target_include_directories(cpp PRIVATE ${PETSC4PY_INCLUDE_DIRS})
endif()
