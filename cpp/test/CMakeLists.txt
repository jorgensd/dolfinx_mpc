# C++ tests of DOLFINx-MPC, built against an installed DOLFINx and DOLFINx-MPC:
#
#   cmake -G Ninja -B build-test -S cpp/test
#   cmake --build build-test
#   mpirun -n 2 build-test/unittests
cmake_minimum_required(VERSION 3.21)

project(dolfinx_mpc_tests LANGUAGES C CXX)

find_package(DOLFINX REQUIRED)
find_package(DOLFINX_MPC REQUIRED)
find_package(Catch2 3 REQUIRED)

# The tests use double precision throughout, whatever the PETSc scalar type
add_custom_command(
  OUTPUT forms.c
  COMMAND ffcx ${CMAKE_CURRENT_SOURCE_DIR}/forms.py --scalar_type=float64
  VERBATIM
  DEPENDS forms.py
  COMMENT "Compile forms.py using FFCx"
)
set_source_files_properties(
  ${CMAKE_CURRENT_BINARY_DIR}/forms.c
  PROPERTIES COMPILE_OPTIONS -Wno-unused-parameter
)

add_executable(
  unittests
  main.cpp
  test_cross_block.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/forms.c
)
target_link_libraries(unittests PRIVATE dolfinx_mpc dolfinx::dolfinx Catch2::Catch2)
target_include_directories(unittests PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_compile_features(unittests PRIVATE cxx_std_20)
set_target_properties(unittests PROPERTIES CXX_EXTENSIONS OFF)

enable_testing()
add_test(NAME unittests COMMAND unittests)
