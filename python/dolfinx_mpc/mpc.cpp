// Copyright (C) 2020 Jørgen S. Dokken
//
// This file is part of DOLFINX-MPC
//
// SPDX-License-Identifier:    MIT

#include <array.h>
#include <caster_petsc.h>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/geometry/utils.h>
#include <dolfinx/la/petsc.h>
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx_mpc/ContactConstraint.h>
#include <dolfinx_mpc/MultiPointConstraint.h>
#include <dolfinx_mpc/SlipConstraint.h>
#include <dolfinx_mpc/assemble_matrix.h>
#include <dolfinx_mpc/assemble_vector.h>
#include <dolfinx_mpc/lifting.h>
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
           const py::array_t<double, py::array::c_style>& x, const int index)
        {
          const std::size_t x_s0 = x.ndim() == 1 ? 1 : x.shape(0);
          const std::int32_t gdim = V->mesh()->geometry().dim();
          xt::xtensor<double, 2> _x
              = xt::zeros<double>({x_s0, static_cast<std::size_t>(gdim)});
          auto xx = x.unchecked();
          if (xx.ndim() == 1)
          {
            for (py::ssize_t i = 0; i < gdim; i++)
              _x(0, i) = xx(i);
          }
          else if (xx.ndim() == 2)
          {
            for (py::ssize_t i = 0; i < xx.shape(0); i++)
              for (py::ssize_t j = 0; j < gdim; j++)
                _x(i, j) = xx(i, j);
          }
          const xt::xtensor<double, 2> basis_function
              = dolfinx_mpc::get_basis_functions(V, _x, index);
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
          "constants",
          [](dolfinx_mpc::MultiPointConstraint<PetscScalar>& self)
          {
            const std::vector<PetscScalar>& consts = self.constant_values();
            return py::array_t<PetscScalar>(consts.size(), consts.data(),
                                            py::cast(self));
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
          "function_space",
          &dolfinx_mpc::MultiPointConstraint<PetscScalar>::function_space)
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
              dolfinx::la::petsc::Matrix::set_block_fn(A, ADD_VALUES),
              dolfinx::la::petsc::Matrix::set_fn(A, ADD_VALUES), a, mpc, bcs,
              diagval);
        });
  m.def(
      "assemble_vector",
      [](py::array_t<PetscScalar, py::array::c_style> b,
         const dolfinx::fem::Form<PetscScalar>& L,
         const std::shared_ptr<
             const dolfinx_mpc::MultiPointConstraint<PetscScalar>>& mpc)
      {
        dolfinx_mpc::assemble_vector(xtl::span(b.mutable_data(), b.size()), L,
                                     mpc);
      },
      py::arg("b"), py::arg("L"), py::arg("mpc"),
      "Assemble linear form into an existing vector");

  m.def(
      "apply_lifting",
      [](py::array_t<PetscScalar, py::array::c_style> b,
         std::vector<std::shared_ptr<const dolfinx::fem::Form<PetscScalar>>>& a,
         const std::vector<std::vector<std::shared_ptr<
             const dolfinx::fem::DirichletBC<PetscScalar>>>>& bcs1,
         const std::vector<py::array_t<PetscScalar, py::array::c_style>>& x0,
         double scale,
         std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<PetscScalar>>&
             mpc)
      {
        std::vector<xtl::span<const PetscScalar>> _x0;
        for (const auto& x : x0)
          _x0.emplace_back(x.data(), x.size());

        dolfinx_mpc::apply_lifting(xtl::span(b.mutable_data(), b.size()), a,
                                   bcs1, _x0, scale, mpc);
      },
      py::arg("b"), py::arg("a"), py::arg("bcs"), py::arg("x0"),
      py::arg("scale"), py::arg("mpc"),
      "Assemble apply lifting from form a on vector b");

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
  m.def("create_normal_approximation",
        [](std::shared_ptr<dolfinx::fem::FunctionSpace> V, std::int32_t dim,
           const py::array_t<std::int32_t, py::array::c_style>& entities)
        {
          return dolfinx_mpc::create_normal_approximation(
              V, dim,
              xtl::span<const std::int32_t>(entities.data(), entities.size()));
        });
}
} // namespace dolfinx_mpc_wrappers
