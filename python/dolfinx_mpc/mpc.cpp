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
#include <dolfinx_mpc/PeriodicConstraint.h>
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

namespace py = pybind11;

namespace

{

// Templating over mesh resolution
template <typename T, std::floating_point U>
void declare_mpc(py::module& m, std::string type)
{

  std::string pyclass_name = "MultiPointConstraint_" + type;
  // dolfinx_mpc::MultiPointConstraint
  py::class_<dolfinx_mpc::MultiPointConstraint<T, U>,
             std::shared_ptr<dolfinx_mpc::MultiPointConstraint<T, U>>>
      multipointconstraint(m, pyclass_name.c_str(),
                           "Object for representing contact "
                           "(non-penetrating) conditions");
  multipointconstraint
      .def(py::init<std::shared_ptr<const dolfinx::fem::FunctionSpace<U>>,
                    std::vector<std::int32_t>, std::vector<std::int64_t>,
                    std::vector<T>, std::vector<std::int32_t>,
                    std::vector<std::int32_t>>())
      .def_property_readonly("masters",
                             &dolfinx_mpc::MultiPointConstraint<T, U>::masters)
      .def("coefficients",
           [](dolfinx_mpc::MultiPointConstraint<T, U>& self)
           {
             std::shared_ptr<const dolfinx::graph::AdjacencyList<T>> adj
                 = self.coefficients();
             const std::vector<std::int32_t>& offsets = adj->offsets();
             const std::vector<T>& data = adj->array();

             return std::pair<py::array_t<T>, py::array_t<std::int32_t>>(
                 py::array_t<T>(data.size(), data.data(), py::cast(self)),
                 py::array_t<std::int32_t>(offsets.size(), offsets.data(),
                                           py::cast(self)));
           })
      .def_property_readonly(
          "constants",
          [](dolfinx_mpc::MultiPointConstraint<T, U>& self)
          {
            const std::vector<T>& consts = self.constant_values();
            return py::array_t<T>(consts.size(), consts.data(), py::cast(self));
          })
      .def_property_readonly("owners",
                             &dolfinx_mpc::MultiPointConstraint<T, U>::owners)
      .def_property_readonly(
          "slaves",
          [](dolfinx_mpc::MultiPointConstraint<T, U>& self)
          {
            const std::vector<std::int32_t>& slaves = self.slaves();
            return py::array_t<std::int32_t>(slaves.size(), slaves.data(),
                                             py::cast(self));
          })
      .def_property_readonly(
          "is_slave",
          [](dolfinx_mpc::MultiPointConstraint<T, U>& self)
          {
            std::span<const std::int8_t> slaves = self.is_slave();
            return py::array_t<std::int8_t>(slaves.size(), slaves.data(),
                                            py::cast(self));
          })

      .def_property_readonly(
          "cell_to_slaves",
          &dolfinx_mpc::MultiPointConstraint<T, U>::cell_to_slaves)
      .def_property_readonly(
          "num_local_slaves",
          &dolfinx_mpc::MultiPointConstraint<T, U>::num_local_slaves)
      .def_property_readonly(
          "function_space",
          &dolfinx_mpc::MultiPointConstraint<T, U>::function_space)
      .def_property_readonly("owners",
                             &dolfinx_mpc::MultiPointConstraint<T, U>::owners)
      .def(
          "backsubstitution",
          [](dolfinx_mpc::MultiPointConstraint<T, U>& self,
             py::array_t<T, py::array::c_style> u)
          { self.backsubstitution(std::span<T>(u.mutable_data(), u.size())); },
          py::arg("u"), "Backsubstitute slave values into vector")
      .def(
          "homogenize",
          [](dolfinx_mpc::MultiPointConstraint<T, U>& self,
             py::array_t<T, py::array::c_style> u)
          { self.homogenize(std::span<T>(u.mutable_data(), u.size())); },
          py::arg("u"), "Homogenize (set to zero) values at slave DoF indices");

  //   .def("ghost_masters", &dolfinx_mpc::mpc_data::ghost_masters);
}

template <typename T, std::floating_point U>
void declare_functions(py::module& m)
{
  m.def("compute_shared_indices", &dolfinx_mpc::compute_shared_indices<U>);

  m.def("create_sparsity_pattern", &dolfinx_mpc::create_sparsity_pattern<T, U>);

  m.def("create_contact_slip_condition",
        &dolfinx_mpc::create_contact_slip_condition<T, U>);
  m.def("create_slip_condition", &dolfinx_mpc::create_slip_condition<T, U>);
  m.def("create_contact_inelastic_condition",
        &dolfinx_mpc::create_contact_inelastic_condition<T, U>);

  m.def(
      "create_periodic_constraint_geometrical",
      [](const std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> V,
         const std::function<py::array_t<bool>(const py::array_t<U>&)>&
             indicator,
         const std::function<py::array_t<U>(const py::array_t<U>&)>& relation,
         const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>>&
             bcs,
         T scale, bool collapse)
      {
        auto _indicator
            = [&indicator](MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
                           const U,
                           MDSPAN_IMPL_STANDARD_NAMESPACE::extents<
                               std::size_t, 3,
                               MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent>>
                               x) -> std::vector<std::int8_t>
        {
          assert(x.size() % 3 == 0);
          const std::array<std::size_t, 2> shape = {3, x.size() / 3};
          const py::array_t _x(shape, x.data_handle(), py::none());
          py::array_t m = indicator(_x);
          std::vector<std::int8_t> s(m.data(), m.data() + m.size());
          return s;
        };

        auto _relation = [&relation](std::span<const U> x) -> std::vector<U>
        {
          assert(x.size() % 3 == 0);
          const std::array<std::size_t, 2> shape = {3, x.size() / 3};
          const py::array_t _x(shape, x.data(), py::none());
          py::array_t v = relation(_x);
          std::vector<U> output(v.data(), v.data() + v.size());
          return output;
        };
        return dolfinx_mpc::create_periodic_condition_geometrical(
            V, _indicator, _relation, bcs, scale, collapse);
      });

  m.def(
      "create_periodic_constraint_topological",
      [](const std::shared_ptr<const dolfinx::fem::FunctionSpace<U>>& V,
         const std::shared_ptr<const dolfinx::mesh::MeshTags<std::int32_t>>&
             meshtags,
         const int dim,
         const std::function<py::array_t<U>(const py::array_t<U>&)>& relation,
         const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>>&
             bcs,
         T scale, bool collapse)
      {
        auto _relation = [&relation](std::span<const U> x) -> std::vector<U>
        {
          const std::array<std::size_t, 2> shape = {3, x.size() / 3};
          py::array_t _x(shape, x.data(), py::none());
          py::array_t v = relation(_x);
          std::vector<U> output(v.data(), v.data() + v.size());
          return output;
        };
        return dolfinx_mpc::create_periodic_condition_topological(
            V, meshtags, dim, _relation, bcs, scale, collapse);
      });
}

template <typename T, std::floating_point U>
void declare_mpc_data(py::module& m, std::string type)
{
  std::string pyclass_name = "mpc_data_" + type;
  py::class_<dolfinx_mpc::mpc_data<T>,
             std::shared_ptr<dolfinx_mpc::mpc_data<T>>>
      mpc_data(m, pyclass_name.c_str(), "Object with data arrays for mpc");
  mpc_data
      .def_property_readonly(
          "slaves",
          [](dolfinx_mpc::mpc_data<T>& self)
          {
            const std::vector<std::int32_t>& slaves = self.slaves;
            return py::array_t<std::int32_t>(slaves.size(), slaves.data(),
                                             py::cast(self));
          })
      .def_property_readonly(
          "masters",
          [](dolfinx_mpc::mpc_data<T>& self)
          {
            const std::vector<std::int64_t>& masters = self.masters;
            return py::array_t<std::int64_t>(masters.size(), masters.data(),
                                             py::cast(self));
          })
      .def_property_readonly(
          "coeffs",
          [](dolfinx_mpc::mpc_data<T>& self)
          {
            const std::vector<T>& coeffs = self.coeffs;
            return py::array_t<T>(coeffs.size(), coeffs.data(), py::cast(self));
          })
      .def_property_readonly(
          "owners",
          [](dolfinx_mpc::mpc_data<T>& self)
          {
            const std::vector<std::int32_t>& owners = self.owners;
            return py::array_t<std::int32_t>(owners.size(), owners.data(),
                                             py::cast(self));
          })
      .def_property_readonly(
          "offsets",
          [](dolfinx_mpc::mpc_data<T>& self)
          {
            const std::vector<std::int32_t>& offsets = self.offsets;
            return py::array_t<std::int32_t>(offsets.size(), offsets.data(),
                                             py::cast(self));
          });
}

template <typename T = PetscScalar, std::floating_point U>
void declare_petsc_functions(py::module& m)
{
  m.def("create_normal_approximation",
        [](std::shared_ptr<dolfinx::fem::FunctionSpace<U>> V, std::int32_t dim,
           const py::array_t<std::int32_t, py::array::c_style>& entities)
        {
          return dolfinx_mpc::create_normal_approximation(
              V, dim,
              std::span<const std::int32_t>(entities.data(), entities.size()));
        });
  m.def(
      "assemble_matrix",
      [](Mat A, const dolfinx::fem::Form<T>& a,
         const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T, U>>&
             mpc0,
         const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T, U>>&
             mpc1,
         const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>>&
             bcs,
         const T diagval)
      {
        dolfinx_mpc::assemble_matrix(
            dolfinx::la::petsc::Matrix::set_block_fn(A, ADD_VALUES),
            dolfinx::la::petsc::Matrix::set_fn(A, ADD_VALUES), a, mpc0, mpc1,
            bcs, diagval);
      });
  m.def(
      "assemble_vector",
      [](py::array_t<T, py::array::c_style> b, const dolfinx::fem::Form<T>& L,
         const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T, U>>&
             mpc)
      {
        dolfinx_mpc::assemble_vector(std::span(b.mutable_data(), b.size()), L,
                                     mpc);
      },
      py::arg("b"), py::arg("L"), py::arg("mpc"),
      "Assemble linear form into an existing vector");

  m.def(
      "apply_lifting",
      [](py::array_t<T, py::array::c_style> b,
         std::vector<std::shared_ptr<const dolfinx::fem::Form<T>>>& a,
         const std::vector<std::vector<
             std::shared_ptr<const dolfinx::fem::DirichletBC<T>>>>& bcs1,
         const std::vector<py::array_t<T, py::array::c_style>>& x0, U scale,
         std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T, U>>& mpc)
      {
        std::vector<std::span<const T>> _x0;
        for (const auto& x : x0)
          _x0.emplace_back(x.data(), x.size());

        dolfinx_mpc::apply_lifting(std::span(b.mutable_data(), b.size()), a,
                                   bcs1, _x0, scale, mpc);
      },
      py::arg("b"), py::arg("a"), py::arg("bcs"), py::arg("x0"),
      py::arg("scale"), py::arg("mpc"),
      "Assemble apply lifting from form a on vector b");

  m.def(
      "create_matrix",
      [](const dolfinx::fem::Form<T>& a,
         const std::shared_ptr<dolfinx_mpc::MultiPointConstraint<T, U>>& mpc)
      {
        auto A = dolfinx_mpc::create_matrix(a, mpc);
        Mat _A = A.mat();
        PetscObjectReference((PetscObject)_A);
        return _A;
      },
      py::return_value_policy::take_ownership,
      "Create a PETSc Mat for bilinear form.");
  m.def(
      "create_matrix",
      [](const dolfinx::fem::Form<T>& a,
         const std::shared_ptr<dolfinx_mpc::MultiPointConstraint<T, U>>& mpc0,
         const std::shared_ptr<dolfinx_mpc::MultiPointConstraint<T, U>>& mpc1)
      {
        auto A = dolfinx_mpc::create_matrix(a, mpc0, mpc1);
        Mat _A = A.mat();
        PetscObjectReference((PetscObject)_A);
        return _A;
      },
      py::return_value_policy::take_ownership,
      "Create a PETSc Mat for bilinear form.");
}

} // namespace

namespace dolfinx_mpc_wrappers
{

void mpc(py::module& m)
{

  declare_mpc<float, float>(m, "float");
  declare_mpc<std::complex<float>, float>(m, "complex_float");
  declare_mpc<double, double>(m, "double");
  declare_mpc<std::complex<double>, double>(m, "complex_double");

  declare_functions<float, float>(m);
  declare_functions<std::complex<float>, float>(m);
  declare_functions<double, double>(m);
  declare_functions<std::complex<double>, double>(m);

  declare_mpc_data<float, float>(m, "float");
  declare_mpc_data<std::complex<float>, float>(m, "complex_float");
  declare_mpc_data<double, double>(m, "double");
  declare_mpc_data<std::complex<double>, double>(m, "complex_double");

  declare_petsc_functions<PetscScalar, PetscReal>(m);
}
} // namespace dolfinx_mpc_wrappers
