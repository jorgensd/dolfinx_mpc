// Copyright (C) 2020 Jørgen S. Dokken
//
// This file is part of DOLFINX-MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "caster_petsc.h"
#include <Eigen/Dense>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/geometry/utils.h>
#include <dolfinx/la/PETScMatrix.h>
#include <dolfinx_mpc/MultiPointConstraint.h>
#include <dolfinx_mpc/assembly.h>
#include <dolfinx_mpc/utils.h>
#include <memory>
#include <petscmat.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dolfinx_mpc_wrappers
{
void mpc(py::module& m)
{

  m.def("get_basis_functions", &dolfinx_mpc::get_basis_functions);
  m.def("compute_shared_indices", &dolfinx_mpc::compute_shared_indices);
  m.def("add_pattern_diagonal", &dolfinx_mpc::add_pattern_diagonal);

  // dolfinx_mpc::MultiPointConstraint
  py::class_<dolfinx_mpc::MultiPointConstraint,
             std::shared_ptr<dolfinx_mpc::MultiPointConstraint>>
      multipointconstraint(
          m, "MultiPointConstraint",
          "Object for representing contact (non-penetrating) conditions");
  multipointconstraint
      .def(py::init<std::shared_ptr<const dolfinx::function::FunctionSpace>,
                    Eigen::Array<std::int32_t, Eigen::Dynamic, 1>,
                    std::int32_t>())
      .def("slaves", &dolfinx_mpc::MultiPointConstraint::slaves)
      .def("slave_cells", &dolfinx_mpc::MultiPointConstraint::slave_cells)
      .def("slave_to_cells", &dolfinx_mpc::MultiPointConstraint::slave_to_cells)
      .def("add_masters", &dolfinx_mpc::MultiPointConstraint::add_masters)
      .def("cell_to_slaves", &dolfinx_mpc::MultiPointConstraint::cell_to_slaves)
      .def("masters_local", &dolfinx_mpc::MultiPointConstraint::masters_local)
      .def("coefficients", &dolfinx_mpc::MultiPointConstraint::coefficients)
      .def("create_sparsity_pattern",
           &dolfinx_mpc::MultiPointConstraint::create_sparsity_pattern)
      .def("num_local_slaves",
           &dolfinx_mpc::MultiPointConstraint::num_local_slaves)
      .def("index_map", &dolfinx_mpc::MultiPointConstraint::index_map)
      .def("dofmap", &dolfinx_mpc::MultiPointConstraint::dofmap)
      .def("owners", &dolfinx_mpc::MultiPointConstraint::owners);

  m.def("compute_process_collisions",
        py::overload_cast<const dolfinx::geometry::BoundingBoxTree&,
                          const Eigen::Vector3d&>(
            &dolfinx::geometry::compute_process_collisions));
  m.def("assemble_matrix",
        [](Mat A, const dolfinx::fem::Form<PetscScalar>& a,
           dolfinx_mpc::MultiPointConstraint& mpc,
           const std::vector<std::shared_ptr<
               const dolfinx::fem::DirichletBC<PetscScalar>>>& bcs) {
          dolfinx_mpc::assemble_matrix(dolfinx::la::PETScMatrix::add_fn(A), a,
                                       mpc, bcs);
        });
}
} // namespace dolfinx_mpc_wrappers
