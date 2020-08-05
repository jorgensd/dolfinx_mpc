// Copyright (C) 2020 Jørgen S. Dokken
//
// This file is part of DOLFINX-MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <Eigen/Dense>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/geometry/utils.h>
#include <dolfinx_mpc/ContactConstraint.h>
#include <dolfinx_mpc/utils.h>
#include <memory>
#include <pybind11/eigen.h>
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

  // dolfinx_mpc::ContactConstraint
  py::class_<dolfinx_mpc::ContactConstraint,
             std::shared_ptr<dolfinx_mpc::ContactConstraint>>
      contactconstraint(
          m, "ContactConstraint",
          "Object for representing contact (non-penetrating) conditions");
  contactconstraint
      .def(py::init<std::shared_ptr<const dolfinx::function::FunctionSpace>,
                    Eigen::Array<std::int32_t, Eigen::Dynamic, 1>,
                    std::int32_t>())
      .def("slaves", &dolfinx_mpc::ContactConstraint::slaves)
      .def("slave_cells", &dolfinx_mpc::ContactConstraint::slave_cells)
      .def("slave_to_cells", &dolfinx_mpc::ContactConstraint::slave_to_cells)
      .def("add_masters", &dolfinx_mpc::ContactConstraint::add_masters)
      .def("cell_to_slaves", &dolfinx_mpc::ContactConstraint::cell_to_slaves)
      .def("masters_local", &dolfinx_mpc::ContactConstraint::masters_local)
      .def("coefficients", &dolfinx_mpc::ContactConstraint::coefficients)
      .def("create_sparsity_pattern",
           &dolfinx_mpc::ContactConstraint::create_sparsity_pattern)
      .def("num_local_slaves",
           &dolfinx_mpc::ContactConstraint::num_local_slaves)
      .def("index_map", &dolfinx_mpc::ContactConstraint::index_map)
      .def("dofmap", &dolfinx_mpc::ContactConstraint::dofmap)
      .def("owners", &dolfinx_mpc::ContactConstraint::owners);

  m.def("compute_process_collisions",
        py::overload_cast<const dolfinx::geometry::BoundingBoxTree&,
                          const Eigen::Vector3d&>(
            &dolfinx::geometry::compute_process_collisions));
}
} // namespace dolfinx_mpc_wrappers
