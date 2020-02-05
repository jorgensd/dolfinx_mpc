set(HEADERS_mpc
  ${CMAKE_CURRENT_SOURCE_DIR}/dolfinx_mpc.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MultiPointConstraint.h
  PARENT_SCOPE)

target_sources(dolfinx_mpc PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/MultiPointConstraint.cpp
)
