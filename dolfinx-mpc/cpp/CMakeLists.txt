add_library(cpp MultiPointConstraint.h)

# MPI
target_link_libraries(cpp PUBLIC MPI::MPI_CXX)

# PETSc
target_link_libraries(cpp PUBLIC PETSC::petsc)
target_link_libraries(cpp PRIVATE PETSC::petsc_static)


# state that anybody linking to us needs to include the current source dir
# to find MathFunctions.h, while we don't.
target_include_directories(cpp
          INTERFACE ${CMAKE_CURRENT_SOURCE_DIR} SYSTEM PUBLIC ${EIGEN3_INCLUDE_DIR}
          )
