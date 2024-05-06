// Copyright (C) 2020 Jørgen S. Dokken
//
// This file is part of DOLFINX-MPC
//
// SPDX-License-Identifier:    MIT

#include <array>
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
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <petsc4py/petsc4py.h>
#include <petscmat.h>
#include <petscvec.h>
namespace nb = nanobind;
using namespace nb::literals;

namespace

{

// Templating over mesh resolution
template <typename T, std::floating_point U>
void declare_mpc(nb::module_& m, std::string type)
{

  std::string nbclass_name = "MultiPointConstraint_" + type;
  // dolfinx_mpc::MultiPointConstraint
  nb::class_<dolfinx_mpc::MultiPointConstraint<T, U>>(
      m, nbclass_name.c_str(),
      "Object for representing contact (non-penetrating) conditions")
      .def("__init__",
           [](dolfinx_mpc::MultiPointConstraint<T, U>* mpc,
              std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> V,
              nb::ndarray<nb::numpy, std::int32_t, nb::ndim<1>>& slaves,
              nb::ndarray<nb::numpy, std::int64_t, nb::ndim<1>>& masters,
              nb::ndarray<nb::numpy, T, nb::ndim<1>>& coeffs,
              nb::ndarray<nb::numpy, std::int32_t, nb::ndim<1>>& owners,
              nb::ndarray<nb::numpy, std::int32_t, nb::ndim<1>>& offsets)
           {
             new (mpc) dolfinx_mpc::MultiPointConstraint(
                 V, std::span<const std::int32_t>(slaves.data(), slaves.size()),
                 std::span<const std::int64_t>(masters.data(), masters.size()),
                 std::span<const T>(coeffs.data(), coeffs.size()),
                 std::span<const std::int32_t>(owners.data(), owners.size()),
                 std::span<const std::int32_t>(offsets.data(), offsets.size()));
           })
      .def_prop_ro("masters", &dolfinx_mpc::MultiPointConstraint<T, U>::masters)
      .def("coefficients",
           [](dolfinx_mpc::MultiPointConstraint<T, U>& self)
           {
             std::shared_ptr<const dolfinx::graph::AdjacencyList<T>> adj
                 = self.coefficients();
             const std::vector<std::int32_t>& offsets = adj->offsets();
             const std::vector<T>& data = adj->array();

             return std::make_pair(
                 nb::ndarray<nb::numpy, const T, nb::ndim<1>>(data.data(),
                                                              {data.size()}),
                 nb::ndarray<nb::numpy, const std::int32_t, nb::ndim<1>>(
                     offsets.data(), {offsets.size()}));
           })
      .def_prop_ro("constants",
                   [](dolfinx_mpc::MultiPointConstraint<T, U>& self)
                   {
                     const std::vector<T>& consts = self.constant_values();
                     return nb::ndarray<nb::numpy, const T, nb::ndim<1>>(
                         consts.data(), {consts.size()});
                   })
      .def_prop_ro("owners", &dolfinx_mpc::MultiPointConstraint<T, U>::owners)
      .def_prop_ro(
          "slaves",
          [](dolfinx_mpc::MultiPointConstraint<T, U>& self)
          {
            const std::vector<std::int32_t>& slaves = self.slaves();
            return nb::ndarray<nb::numpy, const std::int32_t, nb::ndim<1>>(
                slaves.data(), {slaves.size()});
          })
      .def_prop_ro(
          "is_slave",
          [](dolfinx_mpc::MultiPointConstraint<T, U>& self)
          {
            std::span<const std::int8_t> slaves = self.is_slave();
            return nb::ndarray<nb::numpy, const std::int8_t, nb::ndim<1>>(
                slaves.data(), {slaves.size()});
          })

      .def_prop_ro("cell_to_slaves",
                   &dolfinx_mpc::MultiPointConstraint<T, U>::cell_to_slaves)
      .def_prop_ro("num_local_slaves",
                   &dolfinx_mpc::MultiPointConstraint<T, U>::num_local_slaves)
      .def_prop_ro("function_space",
                   &dolfinx_mpc::MultiPointConstraint<T, U>::function_space)
      .def_prop_ro("owners", &dolfinx_mpc::MultiPointConstraint<T, U>::owners)
      .def(
          "backsubstitution",
          [](dolfinx_mpc::MultiPointConstraint<T, U>& self,
             nb::ndarray<T, nb::ndim<1>, nb::c_contig> u)
          { self.backsubstitution(std::span<T>(u.data(), u.size())); },
          "u"_a, "Backsubstitute slave values into vector")
      .def(
          "homogenize",
          [](dolfinx_mpc::MultiPointConstraint<T, U>& self,
             nb::ndarray<T, nb::ndim<1>, nb::c_contig> u)
          { self.homogenize(std::span<T>(u.data(), u.size())); },
          "u"_a, "Homogenize (set to zero) values at slave DoF indices");

  //   .def("ghost_masters", &dolfinx_mpc::mpc_data::ghost_masters);
}
template <typename T, std::floating_point U>
void declare_functions(nb::module_& m)
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
      [](std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> V,
         const std::function<nb::ndarray<bool, nb::ndim<1>, nb::c_contig>(
             nb::ndarray<const U, nb::ndim<2>, nb::numpy>&)>& indicator,
         const std::function<nb::ndarray<U, nb::ndim<2>, nb::numpy>(
             nb::ndarray<const U, nb::ndim<2>, nb::numpy>&)>& relation,
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
          nb::ndarray<const U, nb::ndim<2>, nb::numpy> x_view(
              x.data_handle(), {3, x.size() / 3});
          auto m = indicator(x_view);
          std::vector<std::int8_t> s(m.data(), m.data() + m.size());
          return s;
        };

        auto _relation = [&relation](std::span<const U> x) -> std::vector<U>
        {
          assert(x.size() % 3 == 0);
          nb::ndarray<const U, nb::ndim<2>, nb::numpy> x_view(
              x.data(), {3, x.size() / 3});
          auto v = relation(x_view);
          std::vector<U> output(v.data(), v.data() + v.size());
          return output;
        };
        return dolfinx_mpc::create_periodic_condition_geometrical(
            V, _indicator, _relation, bcs, scale, collapse);
      },
      "V"_a, "indicator"_a, "relation"_a, "bcs"_a, nb::arg("scale").noconvert(),
      nb::arg("collapse").noconvert());
  m.def(
      "create_periodic_constraint_topological",
      [](std::shared_ptr<const dolfinx::fem::FunctionSpace<U>>& V,
         std::shared_ptr<const dolfinx::mesh::MeshTags<std::int32_t>>& meshtags,
         const int dim,
         const std::function<nb::ndarray<U, nb::ndim<2>, nb::numpy>(
             nb::ndarray<const U, nb::ndim<2>, nb::numpy>&)>& relation,
         const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>>&
             bcs,
         T scale, bool collapse)
      {
        auto _relation = [&relation](std::span<const U> x) -> std::vector<U>
        {
          nb::ndarray<const U, nb::ndim<2>, nb::numpy> x_view(
              x.data(), {3, x.size() / 3});
          auto v = relation(x_view);
          std::vector<U> output(v.data(), v.data() + v.size());
          return output;
        };
        return dolfinx_mpc::create_periodic_condition_topological(
            V, meshtags, dim, _relation, bcs, scale, collapse);
      },
      "V"_a, "meshtags"_a, "dim"_a, "relation"_a, "bcs"_a,
      nb::arg("scale").noconvert(), nb::arg("collapse").noconvert());
}

template <typename T, std::floating_point U>
void declare_mpc_data(nb::module_& m, std::string type)
{
  std::string nbclass_name = "mpc_data_" + type;
  nb::class_<dolfinx_mpc::mpc_data<T>>(m, nbclass_name.c_str(),
                                       "Object with data arrays for mpc")
      .def_prop_ro(
          "slaves",
          [](dolfinx_mpc::mpc_data<T>& self)
          {
            const std::vector<std::int32_t>& slaves = self.slaves;
            return nb::ndarray<nb::numpy, const std::int32_t, nb::ndim<1>>(
                slaves.data(), {slaves.size()});
          })
      .def_prop_ro(
          "masters",
          [](dolfinx_mpc::mpc_data<T>& self)
          {
            const std::vector<std::int64_t>& masters = self.masters;
            return nb::ndarray<nb::numpy, const std::int64_t, nb::ndim<1>>(
                masters.data(), {masters.size()});
          })
      .def_prop_ro("coeffs",
                   [](dolfinx_mpc::mpc_data<T>& self)
                   {
                     const std::vector<T>& coeffs = self.coeffs;
                     return nb::ndarray<nb::numpy, const T, nb::ndim<1>>(
                         coeffs.data(), {coeffs.size()});
                   })
      .def_prop_ro(
          "owners",
          [](dolfinx_mpc::mpc_data<T>& self)
          {
            const std::vector<std::int32_t>& owners = self.owners;
            return nb::ndarray<nb::numpy, const std::int32_t, nb::ndim<1>>(
                owners.data(), {owners.size()});
          })
      .def_prop_ro(
          "offsets",
          [](dolfinx_mpc::mpc_data<T>& self)
          {
            const std::vector<std::int32_t>& offsets = self.offsets;
            return nb::ndarray<nb::numpy, const std::int32_t, nb::ndim<1>>(
                offsets.data(), {offsets.size()});
          });
}

template <typename T = PetscScalar, std::floating_point U>
void declare_petsc_functions(nb::module_& m)
{
  import_petsc4py();
  m.def("create_normal_approximation",
        [](std::shared_ptr<dolfinx::fem::FunctionSpace<U>> V, std::int32_t dim,
           const nb::ndarray<std::int32_t, nb::ndim<1>, nb::c_contig>& entities)
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
      [](nb::ndarray<T, nb::ndim<1>, nb::c_contig> b,
         const dolfinx::fem::Form<T>& L,
         const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T, U>>&
             mpc)
      { dolfinx_mpc::assemble_vector(std::span(b.data(), b.size()), L, mpc); },
      "b"_a, "L"_a, "mpc"_a, "Assemble linear form into an existing vector");

  m.def(
      "apply_lifting",
      [](nb::ndarray<T, nb::ndim<1>, nb::c_contig> b,
         std::vector<std::shared_ptr<const dolfinx::fem::Form<T>>>& a,
         const std::vector<std::vector<
             std::shared_ptr<const dolfinx::fem::DirichletBC<T, U>>>>& bcs1,
         const std::vector<nb::ndarray<const T, nb::ndim<1>, nb::c_contig>>& x0,
         T scale,
         std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T, U>>& mpc)
      {
        std::vector<std::span<const T>> _x0;
        for (const auto& x : x0)
          _x0.emplace_back(x.data(), x.size());

        dolfinx_mpc::apply_lifting(std::span(b.data(), b.size()), a, bcs1, _x0,
                                   scale, mpc);
      },
      nb::arg("b"), nb::arg("a"), nb::arg("bcs"), nb::arg("x0"),
      nb::arg("scale"), nb::arg("mpc"),
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
      nb::rv_policy::take_ownership, "Create a PETSc Mat for bilinear form.");
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
      nb::rv_policy::take_ownership, "Create a PETSc Mat for bilinear form.");
}

} // namespace

namespace dolfinx_mpc_wrappers
{

void mpc(nb::module_& m)
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
