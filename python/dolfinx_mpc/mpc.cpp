// Copyright (C) 2020 Jørgen S. Dokken
//
// This file is part of DOLFINX-MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "array.h"
#include "caster_petsc.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/geometry/utils.h>
#include <dolfinx/la/PETScMatrix.h>
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx_mpc/ContactConstraint.h>
#include <dolfinx_mpc/MultiPointConstraint.h>
#include <dolfinx_mpc/SlipConstraint.h>
#include <dolfinx_mpc/assemble_matrix.h>
#include <dolfinx_mpc/utils.h>
#include <memory>
#include <petscmat.h>
#include <petscvec.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <xtensor/xtensor.hpp>
#include <xtl/xspan.hpp>
namespace py = pybind11;

namespace dolfinx_mpc_wrappers
{
void mpc(py::module& m)
{

  m.def("get_basis_functions",
        [](std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
           const std::array<double, 3>& x, const int index)
        {
          const xt::xtensor<double, 2> basis_function
              = dolfinx_mpc::get_basis_functions(V, x, index);
          return py::array_t<double>(basis_function.shape(),
                                     basis_function.data());
        });
  m.def("compute_shared_indices", &dolfinx_mpc::compute_shared_indices);

  // dolfinx_mpc::MultiPointConstraint
  py::class_<dolfinx_mpc::MultiPointConstraint<PetscScalar>,
             std::shared_ptr<dolfinx_mpc::MultiPointConstraint<PetscScalar>>>
      multipointconstraint(
          m, "MultiPointConstraint",
          "Object for representing contact (non-penetrating) conditions");
  multipointconstraint
      .def(py::init<std::shared_ptr<const dolfinx::fem::FunctionSpace>,
                    std::vector<std::int32_t>, std::vector<std::int64_t>,
                    std::vector<PetscScalar>, std::vector<std::int32_t>,
                    std::vector<std::int32_t>>())
      .def_property_readonly(
          "masters", &dolfinx_mpc::MultiPointConstraint<PetscScalar>::masters)
      .def("coefficients",
           [](dolfinx_mpc::MultiPointConstraint<PetscScalar>& self)
           {
             const std::shared_ptr<dolfinx::graph::AdjacencyList<PetscScalar>>&
                 adj
                 = self.coefficients();
             const std::vector<std::int32_t>& offsets = adj->offsets();
             const std::vector<PetscScalar>& data = adj->array();

             return std::pair<py::array_t<PetscScalar>,
                              py::array_t<std::int32_t>>(
                 py::array_t<PetscScalar>(data.size(), data.data(),
                                          py::cast(self)),
                 py::array_t<std::int32_t>(offsets.size(), offsets.data(),
                                           py::cast(self)));
           })
      .def_property_readonly(
          "owners", &dolfinx_mpc::MultiPointConstraint<PetscScalar>::owners)
      .def_property_readonly(
          "slaves",
          [](dolfinx_mpc::MultiPointConstraint<PetscScalar>& self)
          {
            const std::vector<std::int32_t>& slaves = self.slaves();
            return py::array_t<std::int32_t>(slaves.size(), slaves.data(),
                                             py::cast(self));
          })
      .def_property_readonly(
          "is_slave",
          [](dolfinx_mpc::MultiPointConstraint<PetscScalar>& self)
          {
            const std::vector<std::int8_t>& slaves = self.is_slave();
            return py::array_t<std::int8_t>(slaves.size(), slaves.data(),
                                            py::cast(self));
          })

      .def_property_readonly(
          "cell_to_slaves",
          &dolfinx_mpc::MultiPointConstraint<PetscScalar>::cell_to_slaves)
      .def_property_readonly(
          "num_local_slaves",
          &dolfinx_mpc::MultiPointConstraint<PetscScalar>::num_local_slaves)
      .def_property_readonly(
          "index_map",
          &dolfinx_mpc::MultiPointConstraint<PetscScalar>::index_map)
      .def_property_readonly(
          "dofmap", &dolfinx_mpc::MultiPointConstraint<PetscScalar>::dofmap)
      .def_property_readonly(
          "owners", &dolfinx_mpc::MultiPointConstraint<PetscScalar>::owners)
      .def(
          "backsubstitution",
          [](dolfinx_mpc::MultiPointConstraint<PetscScalar>& self,
             py::array_t<PetscScalar, py::array::c_style> u) {
            self.backsubstitution(
                xtl::span<PetscScalar>(u.mutable_data(), u.size()));
          },
          py::arg("u"), "Backsubstitute slave values into vector");

  py::class_<dolfinx_mpc::mpc_data, std::shared_ptr<dolfinx_mpc::mpc_data>>
      mpc_data(m, "mpc_data", "Object with data arrays for mpc");
  mpc_data
      .def_property_readonly(
          "slaves",
          [](dolfinx_mpc::mpc_data& self)
          {
            const std::vector<std::int32_t>& slaves = self.slaves;
            return py::array_t<std::int32_t>(slaves.size(), slaves.data(),
                                             py::cast(self));
          })
      .def_property_readonly(
          "masters",
          [](dolfinx_mpc::mpc_data& self)
          {
            const std::vector<std::int64_t>& masters = self.masters;
            return py::array_t<std::int64_t>(masters.size(), masters.data(),
                                             py::cast(self));
          })
      .def_property_readonly(
          "coeffs",
          [](dolfinx_mpc::mpc_data& self)
          {
            const std::vector<PetscScalar>& coeffs = self.coeffs;
            return py::array_t<PetscScalar>(coeffs.size(), coeffs.data(),
                                            py::cast(self));
          })
      .def_property_readonly(
          "owners",
          [](dolfinx_mpc::mpc_data& self)
          {
            const std::vector<std::int32_t>& owners = self.owners;
            return py::array_t<std::int32_t>(owners.size(), owners.data(),
                                             py::cast(self));
          })
      .def_property_readonly(
          "offsets",
          [](dolfinx_mpc::mpc_data& self)
          {
            const std::vector<std::int32_t>& offsets = self.offsets;
            return py::array_t<std::int32_t>(offsets.size(), offsets.data(),
                                             py::cast(self));
          });

  //   .def("ghost_masters", &dolfinx_mpc::mpc_data::ghost_masters);
  m.def("create_sparsity_pattern", &dolfinx_mpc::create_sparsity_pattern);

  m.def("assemble_matrix",
        [](Mat A, const dolfinx::fem::Form<PetscScalar>& a,
           const std::shared_ptr<
               const dolfinx_mpc::MultiPointConstraint<PetscScalar>>& mpc,
           const std::vector<std::shared_ptr<
               const dolfinx::fem::DirichletBC<PetscScalar>>>& bcs,
           const PetscScalar diagval)
        {
          dolfinx_mpc::assemble_matrix(
              dolfinx::la::PETScMatrix::set_block_fn(A, ADD_VALUES),
              dolfinx::la::PETScMatrix::set_fn(A, ADD_VALUES), a, mpc, bcs,
              diagval);
        });

  m.def(
      "create_matrix",
      [](const dolfinx::fem::Form<PetscScalar>& a,
         const std::shared_ptr<dolfinx_mpc::MultiPointConstraint<PetscScalar>>&
             mpc)
      {
        auto A = dolfinx_mpc::create_matrix(a, mpc);
        Mat _A = A.mat();
        PetscObjectReference((PetscObject)_A);
        return _A;
      },
      py::return_value_policy::take_ownership,
      "Create a PETSc Mat for bilinear form.");
  m.def("create_contact_slip_condition",
        &dolfinx_mpc::create_contact_slip_condition);
  m.def("create_slip_condition",
        [](std::vector<std::shared_ptr<dolfinx::fem::FunctionSpace>> spaces,
           dolfinx::mesh::MeshTags<std::int32_t> meshtags, std::int32_t marker,
           std::shared_ptr<dolfinx::fem::Function<PetscScalar>> v,
           py::array_t<std::int32_t, py::array::c_style> sub_map,
           std::vector<
               std::shared_ptr<const dolfinx::fem::DirichletBC<PetscScalar>>>
               bcs)
        {
          return dolfinx_mpc::create_slip_condition(
              spaces, meshtags, marker, v,
              xtl::span<const std::int32_t>(sub_map.data(), sub_map.size()),
              bcs);
        });
  m.def("create_contact_inelastic_condition",
        &dolfinx_mpc::create_contact_inelastic_condition);
  m.def("create_dof_to_facet_map", &dolfinx_mpc::create_dof_to_facet_map);
  m.def("create_average_normal", &dolfinx_mpc::create_average_normal);
  m.def("create_normal_approximation",
        [](std::shared_ptr<dolfinx::fem::FunctionSpace> V,
           const py::array_t<std::int32_t, py::array::c_style>& entities,
           py::array_t<PetscScalar, py::array::c_style> vector)
        {
          return dolfinx_mpc::create_normal_approximation(
              V,
              xtl::span<const std::int32_t>(entities.data(), entities.size()),
              xtl::span<PetscScalar>(vector.mutable_data(), vector.size()));
        });
}
} // namespace dolfinx_mpc_wrappers
