// Copyright (C) 2020 Jørgen S. Dokken
//
// This file is part of DOLFINX-MPC
//
// SPDX-License-Identifier:    MIT

#include <array>
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
#include <dolfinx_mpc/RBE.h>
#include <dolfinx_mpc/SlipConstraint.h>
#include <dolfinx_mpc/assemble_matrix.h>
#include <dolfinx_mpc/assemble_vector.h>
#include <dolfinx_mpc/lifting.h>
#include <dolfinx_mpc/utils.h>
#include <dolfinx_wrappers/array.h>
#include <dolfinx_wrappers/caster_petsc.h>
#include <format>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <petsc4py/petsc4py.h>
#include <petscmat.h>
#include <petscvec.h>
#include <stdexcept>
namespace nb = nanobind;
using namespace nb::literals;

namespace

{

/// Spans over a list of 1D arrays
template <typename V>
std::vector<std::span<const V>>
as_spans(const std::vector<nb::ndarray<const V, nb::ndim<1>, nb::c_contig>>& a)
{
  std::vector<std::span<const V>> spans;
  spans.reserve(a.size());
  for (const auto& a_i : a)
    spans.emplace_back(a_i.data(), a_i.size());
  return spans;
}

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
      .def("__init__",
           [](dolfinx_mpc::MultiPointConstraint<T, U>* mpc,
              std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> V,
              nb::ndarray<nb::numpy, std::int32_t, nb::ndim<1>>& slaves,
              nb::ndarray<nb::numpy, std::int64_t, nb::ndim<1>>& masters,
              nb::ndarray<nb::numpy, T, nb::ndim<1>>& coeffs,
              nb::ndarray<nb::numpy, std::int32_t, nb::ndim<1>>& owners,
              nb::ndarray<nb::numpy, std::int32_t, nb::ndim<1>>& offsets,
              nb::ndarray<nb::numpy, T, nb::ndim<1>>& rhs_coeffs,
              const std::vector<std::shared_ptr<
                  const dolfinx::fem::DirichletBC<T, U>>>& bcs,
              std::optional<U> filter)
           {
             new (mpc) dolfinx_mpc::MultiPointConstraint(
                 V, std::span<const std::int32_t>(slaves.data(), slaves.size()),
                 std::span<const std::int64_t>(masters.data(), masters.size()),
                 std::span<const T>(coeffs.data(), coeffs.size()),
                 std::span<const std::int32_t>(owners.data(), owners.size()),
                 std::span<const std::int32_t>(offsets.data(), offsets.size()),
                 std::span<const T>(rhs_coeffs.data(), rhs_coeffs.size()), bcs,
                 filter);
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
                 nb::ndarray<nb::numpy, const T, nb::ndim<1>>(
                     data.data(), {data.size()}, nb::handle()),
                 nb::ndarray<nb::numpy, const std::int32_t, nb::ndim<1>>(
                     offsets.data(), {offsets.size()}, nb::handle()));
           })
      .def("all_coefficients",
           [](dolfinx_mpc::MultiPointConstraint<T, U>& self)
           {
             auto [coeffs, offsets] = self.all_coefficients();
             return std::make_pair(
                 dolfinx_wrappers::as_nbarray(std::move(coeffs)),
                 dolfinx_wrappers::as_nbarray(std::move(offsets)));
           })
      .def("all_masters",
           [](dolfinx_mpc::MultiPointConstraint<T, U>& self)
           { return dolfinx_wrappers::as_nbarray(self.all_masters()); })
      .def("all_master_blocks",
           [](dolfinx_mpc::MultiPointConstraint<T, U>& self)
           { return dolfinx_wrappers::as_nbarray(self.all_master_blocks()); })
      .def(
          "update_coefficients",
          [](dolfinx_mpc::MultiPointConstraint<T, U>& self,
             nb::ndarray<const T, nb::ndim<1>, nb::c_contig> coeffs)
          {
            self.update_coefficients(
                std::span<const T>(coeffs.data(), coeffs.size()));
          },
          "coeffs"_a, "Replace the coefficients of all masters")
      .def(
          "scale_coefficients",
          [](dolfinx_mpc::MultiPointConstraint<T, U>& self,
             nb::ndarray<const T, nb::ndim<1>, nb::c_contig> factors)
          {
            self.scale_coefficients(
                std::span<const T>(factors.data(), factors.size()));
          },
          "factors"_a, "Multiply the coefficients of each slave by a factor")
      .def_prop_ro("constants",
                   [](dolfinx_mpc::MultiPointConstraint<T, U>& self)
                   {
                     const std::vector<T>& consts = self.constant_values();
                     return nb::ndarray<nb::numpy, const T, nb::ndim<1>>(
                         consts.data(), {consts.size()}, nb::handle());
                   })
      .def_prop_ro("owners", &dolfinx_mpc::MultiPointConstraint<T, U>::owners)
      .def_prop_ro(
          "slaves",
          [](dolfinx_mpc::MultiPointConstraint<T, U>& self)
          {
            const std::vector<std::int32_t>& slaves = self.slaves();
            return nb::ndarray<nb::numpy, const std::int32_t, nb::ndim<1>>(
                slaves.data(), {slaves.size()}, nb::handle());
          })
      .def_prop_ro(
          "is_slave",
          [](dolfinx_mpc::MultiPointConstraint<T, U>& self)
          {
            std::span<const std::int8_t> slaves = self.is_slave();
            return nb::ndarray<nb::numpy, const std::int8_t, nb::ndim<1>>(
                slaves.data(), {slaves.size()}, nb::handle());
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
          "backsubstitution",
          [](dolfinx_mpc::MultiPointConstraint<T, U>& self,
             std::vector<nb::ndarray<T, nb::ndim<1>, nb::c_contig>> u)
          {
            std::vector<std::span<T>> _u;
            for (auto& u_k : u)
              _u.emplace_back(u_k.data(), u_k.size());
            self.backsubstitution(_u);
          },
          "u"_a,
          "Backsubstitute slave values, reading masters from the vector of "
          "their block")
      .def_prop_ro("block", &dolfinx_mpc::MultiPointConstraint<T, U>::block)
      .def_prop_ro(
          "master_blocks",
          [](dolfinx_mpc::MultiPointConstraint<T, U>& self)
          {
            std::span<const std::int32_t> blocks = self.master_blocks();
            return nb::ndarray<nb::numpy, const std::int32_t, nb::ndim<1>>(
                blocks.data(), {blocks.size()}, nb::handle());
          })
      .def_prop_ro(
          "has_cross_block_masters",
          &dolfinx_mpc::MultiPointConstraint<T, U>::has_cross_block_masters)
      .def(
          "homogenize",
          [](dolfinx_mpc::MultiPointConstraint<T, U>& self,
             nb::ndarray<T, nb::ndim<1>, nb::c_contig> u)
          { self.homogenize(std::span<T>(u.data(), u.size())); },
          "u"_a, "Homogenize (set to zero) values at slave DoF indices")
      .def(
          "set_rhs_coeffs",
          [](dolfinx_mpc::MultiPointConstraint<T, U>& self,
             nb::ndarray<const T, nb::ndim<1>, nb::c_contig> g)
          { self.set_rhs_coeffs(std::span<const T>(g.data(), g.size())); },
          "g"_a, "Replace the user supplied constraint inhomogeneity")
      .def("update_constants",
           &dolfinx_mpc::MultiPointConstraint<T, U>::update_constants,
           "Recompute the constraint offsets from the current Dirichlet data")
      .def_prop_ro(
          "has_inhomogeneity",
          &dolfinx_mpc::MultiPointConstraint<T, U>::has_inhomogeneity);

  //   .def("ghost_masters", &dolfinx_mpc::mpc_data::ghost_masters);
}
template <typename T, std::floating_point U>
void declare_functions(nb::module_& m)
{
  m.def("compute_shared_indices", &dolfinx_mpc::compute_shared_indices<U>);
  m.def(
      "locate_spiders",
      [](const dolfinx::fem::FunctionSpace<U>& W,
         nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig> spiders)
      {
        auto [blocks, owners, x] = dolfinx_mpc::locate_spiders<U>(
            W, std::span(spiders.data(), spiders.size()));
        const std::size_t n = blocks.size();
        return nb::make_tuple(
            dolfinx_wrappers::as_nbarray(std::move(blocks)),
            dolfinx_wrappers::as_nbarray(std::move(owners)),
            dolfinx_wrappers::as_nbarray(std::move(x), {n, 3}));
      },
      nb::arg("W"), nb::arg("spiders"),
      "Global block, owner and coordinate of spiders, by input index");
  m.def(
      "update_rbe2",
      [](dolfinx_mpc::MultiPointConstraint<T, U>& mpc,
         const dolfinx::fem::FunctionSpace<U>& V,
         const dolfinx::fem::FunctionSpace<U>& W, int block)
      { dolfinx_mpc::update_rbe2<T, U>(mpc, V, W, block); },
      nb::arg("mpc"), nb::arg("V"), nb::arg("W"), nb::arg("block"),
      "Recompute the RBE2 coefficients from the current dof coordinates");
  m.def(
      "update_rbe3",
      [](dolfinx_mpc::MultiPointConstraint<T, U>& mpc,
         const dolfinx::fem::FunctionSpace<U>& W,
         const std::vector<
             std::shared_ptr<const dolfinx::fem::FunctionSpace<U>>>& V,
         const std::vector<int>& blocks,
         const std::vector<
             nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig>>& dofs,
         const std::vector<nb::ndarray<const std::int64_t, nb::ndim<1>,
                                       nb::c_contig>>& spiders,
         const std::vector<nb::ndarray<const U, nb::ndim<1>, nb::c_contig>>&
             weights)
      {
        dolfinx_mpc::update_rbe3<T, U>(mpc, W, V, blocks, as_spans(dofs),
                                       as_spans(spiders), as_spans(weights));
      },
      nb::arg("mpc"), nb::arg("W"), nb::arg("V"), nb::arg("blocks"),
      nb::arg("dofs"), nb::arg("spiders"), nb::arg("weights"),
      "Recompute the RBE3 coefficients from the current dof coordinates");
  m.def(
      "create_multipointconstraints",
      [](const std::vector<
             std::shared_ptr<const dolfinx::fem::FunctionSpace<U>>>& V,
         const std::vector<nb::ndarray<nb::numpy, std::int32_t, nb::ndim<1>>>&
             slaves,
         const std::vector<nb::ndarray<nb::numpy, std::int64_t, nb::ndim<1>>>&
             masters,
         const std::vector<nb::ndarray<nb::numpy, T, nb::ndim<1>>>& coeffs,
         const std::vector<nb::ndarray<nb::numpy, std::int32_t, nb::ndim<1>>>&
             owners,
         const std::vector<nb::ndarray<nb::numpy, std::int32_t, nb::ndim<1>>>&
             offsets,
         const std::vector<nb::ndarray<nb::numpy, T, nb::ndim<1>>>& rhs_coeffs,
         const std::vector<std::vector<
             std::shared_ptr<const dolfinx::fem::DirichletBC<T, U>>>>& bcs,
         const std::vector<nb::ndarray<nb::numpy, std::int32_t, nb::ndim<1>>>&
             master_blocks,
         std::optional<U> filter)
      {
        const std::size_t nb = V.size();
        if (slaves.size() != nb or masters.size() != nb or coeffs.size() != nb
            or owners.size() != nb or offsets.size() != nb
            or rhs_coeffs.size() != nb or bcs.size() != nb
            or master_blocks.size() != nb)
        {
          throw std::invalid_argument(
              "Every argument must have one entry per function space.");
        }
        std::vector<dolfinx_mpc::mpc_block_view<T>> data;
        data.reserve(nb);
        for (std::size_t k = 0; k < nb; ++k)
        {
          data.push_back(
              {std::span<const std::int32_t>(slaves[k].data(),
                                             slaves[k].size()),
               std::span<const std::int64_t>(masters[k].data(),
                                             masters[k].size()),
               std::span<const T>(coeffs[k].data(), coeffs[k].size()),
               std::span<const std::int32_t>(owners[k].data(),
                                             owners[k].size()),
               std::span<const std::int32_t>(offsets[k].data(),
                                             offsets[k].size()),
               std::span<const T>(rhs_coeffs[k].data(), rhs_coeffs[k].size()),
               bcs[k],
               std::span<const std::int32_t>(master_blocks[k].data(),
                                             master_blocks[k].size())});
        }
        return dolfinx_mpc::create_multipointconstraints<T, U>(V, data, filter);
      },
      nb::arg("V"), nb::arg("slaves"), nb::arg("masters"), nb::arg("coeffs"),
      nb::arg("owners"), nb::arg("offsets"), nb::arg("rhs_coeffs"),
      nb::arg("bcs"), nb::arg("master_blocks"), nb::arg("filter").none(),
      "Create the multi point constraints of several function spaces together");

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
         T scale, bool collapse, std::optional<U> tol, std::size_t num_threads)
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
              x.data_handle(), {3, x.size() / 3}, nb::handle());
          auto m = indicator(x_view);
          std::vector<std::int8_t> s(m.data(), m.data() + m.size());
          return s;
        };

        auto _relation = [&relation](std::span<const U> x) -> std::vector<U>
        {
          assert(x.size() % 3 == 0);
          nb::ndarray<const U, nb::ndim<2>, nb::numpy> x_view(
              x.data(), {3, x.size() / 3}, nb::handle());
          auto v = relation(x_view);
          std::vector<U> output(v.data(), v.data() + v.size());
          return output;
        };
        return dolfinx_mpc::create_periodic_condition_geometrical(
            V, _indicator, _relation, bcs, scale, collapse, tol, num_threads);
      },
      "V"_a, "indicator"_a, "relation"_a, "bcs"_a, nb::arg("scale").noconvert(),
      nb::arg("collapse").noconvert(), nb::arg("tol").noconvert(),
      nb::arg("num_threads").noconvert());
  m.def(
      "create_periodic_constraint_topological",
      [](std::shared_ptr<const dolfinx::fem::FunctionSpace<U>>& V,
         std::shared_ptr<const dolfinx::mesh::MeshTags<std::int32_t>>& meshtags,
         const int dim,
         const std::function<nb::ndarray<U, nb::ndim<2>, nb::numpy>(
             nb::ndarray<const U, nb::ndim<2>, nb::numpy>&)>& relation,
         const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>>&
             bcs,
         T scale, bool collapse, std::optional<U> tol, std::size_t num_threads)
      {
        auto _relation = [&relation](std::span<const U> x) -> std::vector<U>
        {
          nb::ndarray<const U, nb::ndim<2>, nb::numpy> x_view(
              x.data(), {3, x.size() / 3}, nb::handle());
          auto v = relation(x_view);
          std::vector<U> output(v.data(), v.data() + v.size());
          return output;
        };
        return dolfinx_mpc::create_periodic_condition_topological(
            V, meshtags, dim, _relation, bcs, scale, collapse, tol,
            num_threads);
      },
      "V"_a, "meshtags"_a, "dim"_a, "relation"_a, "bcs"_a,
      nb::arg("scale").noconvert(), nb::arg("collapse").noconvert(),
      nb::arg("tol").noconvert(), nb::arg("num_threads").noconvert());
}

template <typename T, std::floating_point U>
void declare_mpc_data(nb::module_& m, std::string type)
{
  // The scalar type cannot be deduced from the arguments, so it is in the name
  m.def(
      ("create_rbe2_" + type).c_str(),
      [](const dolfinx::fem::FunctionSpace<U>& V,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> dofs,
         nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig> spiders,
         const dolfinx::fem::FunctionSpace<U>& W,
         std::optional<nb::ndarray<const U, nb::ndim<2>, nb::c_contig>> x)
      {
        std::span<const std::int32_t> _dofs(dofs.data(), dofs.size());
        std::span<const std::int64_t> _spiders(spiders.data(), spiders.size());
        if (x)
        {
          return dolfinx_mpc::create_rbe2<T, U>(
              V, _dofs, _spiders, W, std::span<const U>(x->data(), x->size()));
        }
        return dolfinx_mpc::create_rbe2<T, U>(V, _dofs, _spiders, W);
      },
      nb::arg("V"), nb::arg("dofs"), nb::arg("spiders"), nb::arg("W"),
      nb::arg("x").none(),
      "Tie blocked dofs to the rigid-body motion of spiders (RBE2)");
  m.def(
      ("create_rbe3_" + type).c_str(),
      [](const dolfinx::fem::FunctionSpace<U>& W,
         const std::vector<
             std::shared_ptr<const dolfinx::fem::FunctionSpace<U>>>& V,
         const std::vector<
             nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig>>& dofs,
         const std::vector<nb::ndarray<const std::int64_t, nb::ndim<1>,
                                       nb::c_contig>>& spiders,
         const std::vector<nb::ndarray<const U, nb::ndim<1>, nb::c_contig>>&
             weights)
      {
        auto [data, spaces] = dolfinx_mpc::create_rbe3<T, U>(
            W, V, as_spans(dofs), as_spans(spiders), as_spans(weights));
        return nb::make_tuple(std::move(data),
                              dolfinx_wrappers::as_nbarray(std::move(spaces)));
      },
      nb::arg("W"), nb::arg("V"), nb::arg("dofs"), nb::arg("spiders"),
      nb::arg("weights"),
      "Tie the dofs of spiders to the motion of their feet (RBE3)");
  std::string nbclass_name = "mpc_data_" + type;
  nb::class_<dolfinx_mpc::mpc_data<T>>(m, nbclass_name.c_str(),
                                       "Object with data arrays for mpc")
      .def(
          "__init__",
          [](dolfinx_mpc::mpc_data<T>* self, std::vector<std::int32_t> slaves,
             std::vector<std::int64_t> masters, std::vector<T> coeffs,
             std::vector<std::int32_t> owners,
             std::vector<std::int32_t> offsets)
          {
            new (self) dolfinx_mpc::mpc_data<T>{
                std::move(slaves), std::move(masters), std::move(coeffs),
                std::move(offsets), std::move(owners)};
          },
          nb::arg("slaves"), nb::arg("masters"), nb::arg("coeffs"),
          nb::arg("owners"), nb::arg("offsets"))
      .def_prop_ro(
          "slaves",
          [](dolfinx_mpc::mpc_data<T>& self)
          {
            const std::vector<std::int32_t>& slaves = self.slaves;
            return nb::ndarray<nb::numpy, const std::int32_t, nb::ndim<1>>(
                slaves.data(), {slaves.size()}, nb::handle());
          })
      .def_prop_ro(
          "masters",
          [](dolfinx_mpc::mpc_data<T>& self)
          {
            const std::vector<std::int64_t>& masters = self.masters;
            return nb::ndarray<nb::numpy, const std::int64_t, nb::ndim<1>>(
                masters.data(), {masters.size()}, nb::handle());
          })
      .def_prop_ro("coeffs",
                   [](dolfinx_mpc::mpc_data<T>& self)
                   {
                     const std::vector<T>& coeffs = self.coeffs;
                     return nb::ndarray<nb::numpy, const T, nb::ndim<1>>(
                         coeffs.data(), {coeffs.size()}, nb::handle());
                   })
      .def_prop_ro(
          "owners",
          [](dolfinx_mpc::mpc_data<T>& self)
          {
            const std::vector<std::int32_t>& owners = self.owners;
            return nb::ndarray<nb::numpy, const std::int32_t, nb::ndim<1>>(
                owners.data(), {owners.size()}, nb::handle());
          })
      .def_prop_ro(
          "offsets",
          [](dolfinx_mpc::mpc_data<T>& self)
          {
            const std::vector<std::int32_t>& offsets = self.offsets;
            return nb::ndarray<nb::numpy, const std::int32_t, nb::ndim<1>>(
                offsets.data(), {offsets.size()}, nb::handle());
          });
}

/// Position of block `i` among `n` vectors, checked
std::size_t block_position(int i, std::size_t n)
{
  if (i < 0 or static_cast<std::size_t>(i) >= n)
    throw std::invalid_argument(
        std::format("Block {} is not one of the {} vectors.", i, n));
  return static_cast<std::size_t>(i);
}

/// @brief Test if A has row and column block size 1, in which case blocked and
/// non-blocked insertion of dof indices are equivalent.
bool unit_block_size(Mat A)
{
  PetscInt bs0 = -1, bs1 = -1;
  dolfinx::common::petsc::check(MatGetBlockSizes(A, &bs0, &bs1),
                                "MatGetBlockSizes");
  return bs0 == 1 and bs1 == 1;
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
         const nb::ndarray<const std::int8_t, nb::ndim<1>, nb::c_contig>&
             dof_marker0,
         const nb::ndarray<const std::int8_t, nb::ndim<1>, nb::c_contig>&
             dof_marker1,
         std::size_t num_threads)
      {
        // Blocked indices are only meaningful to a matrix whose local-to-global
        // map is blocked, as for a stand-alone or nested matrix. The local
        // sub-matrix of a monolithic matrix has block size 1, and the blocked
        // indices must be expanded; with a scalar form they are plain indices.
        std::function<int(std::span<const std::int32_t>,
                          std::span<const std::int32_t>,
                          const std::span<const T>&)>
            set_block;
        if (!unit_block_size(A))
          set_block = dolfinx::la::petsc::Matrix::set_block_fn(A, ADD_VALUES);
        else
        {
          const int bs0 = a.function_spaces()[0]->dofmap()->bs();
          const int bs1 = a.function_spaces()[1]->dofmap()->bs();
          if (bs0 == 1 and bs1 == 1)
            set_block = dolfinx::la::petsc::Matrix::set_fn(A, ADD_VALUES);
          else
          {
            set_block = dolfinx::la::petsc::Matrix::set_block_expand_fn(
                A, bs0, bs1, ADD_VALUES);
          }
        }
        dolfinx_mpc::assemble_matrix(
            set_block, dolfinx::la::petsc::Matrix::set_fn(A, ADD_VALUES), a,
            mpc0, mpc1,
            std::span<const std::int8_t>(dof_marker0.data(),
                                         dof_marker0.size()),
            std::span<const std::int8_t>(dof_marker1.data(),
                                         dof_marker1.size()),
            num_threads);
      },
      nb::arg("A"), nb::arg("a"), nb::arg("mpc0"), nb::arg("mpc1"),
      nb::arg("dof_marker0"), nb::arg("dof_marker1"), nb::arg("num_threads"),
      "Assemble a bilinear form into a matrix, given constrained dof markers");
  m.def(
      "assemble_matrix_blocks",
      [](const std::vector<std::vector<std::optional<Mat>>>& A_, int i, int j,
         const dolfinx::fem::Form<T>& a,
         const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T, U>>&
             mpc0,
         const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T, U>>&
             mpc1,
         const nb::ndarray<const std::int8_t, nb::ndim<1>, nb::c_contig>&
             dof_marker0,
         const nb::ndarray<const std::int8_t, nb::ndim<1>, nb::c_contig>&
             dof_marker1,
         std::size_t num_threads)
      {
        // A block without a matrix is None
        std::vector<std::vector<Mat>> A(A_.size());
        for (std::size_t k = 0; k < A_.size(); ++k)
          for (const std::optional<Mat>& Akl : A_[k])
            A[k].push_back(Akl.value_or(nullptr));
        const std::size_t nr = A.size();
        if (i < 0 or j < 0 or static_cast<std::size_t>(i) >= nr
            or static_cast<std::size_t>(j) >= A[i].size() or !A[i][j])
        {
          throw std::invalid_argument(
              std::format("Block ({}, {}) of the form has no matrix.", i, j));
        }

        // The form's own block, with the blocked insertion of its matrix
        Mat Aij = A[i][j];
        std::function<int(std::span<const std::int32_t>,
                          std::span<const std::int32_t>,
                          const std::span<const T>&)>
            set_block;
        if (!unit_block_size(Aij))
          set_block = dolfinx::la::petsc::Matrix::set_block_fn(Aij, ADD_VALUES);
        else
        {
          const int bs0 = a.function_spaces()[0]->dofmap()->bs();
          const int bs1 = a.function_spaces()[1]->dofmap()->bs();
          if (bs0 == 1 and bs1 == 1)
            set_block = dolfinx::la::petsc::Matrix::set_fn(Aij, ADD_VALUES);
          else
          {
            set_block = dolfinx::la::petsc::Matrix::set_block_expand_fn(
                Aij, bs0, bs1, ADD_VALUES);
          }
        }

        // Every block, with unrolled indices, for the entries of the masters
        std::vector<std::vector<std::function<int(std::span<const std::int32_t>,
                                                  std::span<const std::int32_t>,
                                                  std::span<const T>)>>>
            set(nr);
        for (std::size_t k = 0; k < nr; ++k)
        {
          for (Mat Akl : A[k])
          {
            set[k].emplace_back();
            if (Akl)
              set[k].back()
                  = dolfinx::la::petsc::Matrix::set_fn(Akl, ADD_VALUES);
          }
        }
        auto set_blocks = dolfinx_mpc::make_mat_add_blocks<T, U>(
            std::move(set), i, j, *mpc0, *mpc1);

        dolfinx_mpc::assemble_matrix_blocks<T, U>(
            set_block, set_blocks, a, mpc0, mpc1,
            std::span<const std::int8_t>(dof_marker0.data(),
                                         dof_marker0.size()),
            std::span<const std::int8_t>(dof_marker1.data(),
                                         dof_marker1.size()),
            num_threads);
      },
      nb::arg("A"), nb::arg("i"), nb::arg("j"), nb::arg("a"), nb::arg("mpc0"),
      nb::arg("mpc1"), nb::arg("dof_marker0"), nb::arg("dof_marker1"),
      nb::arg("num_threads"),
      "Assemble the bilinear form of block (i, j) into the matrices `A` of "
      "every block, placing the entries of masters in the block they belong "
      "to");
  m.def(
      "assemble_vector_blocks",
      [](std::vector<nb::ndarray<T, nb::ndim<1>, nb::c_contig>> b, int i,
         const dolfinx::fem::Form<T>& L,
         const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T, U>>&
             mpc,
         std::size_t num_threads)
      {
        std::vector<std::span<T>> _b;
        for (auto& b_k : b)
          _b.emplace_back(b_k.data(), b_k.size());
        dolfinx_mpc::assemble_vector_blocks<T, U>(_b, i, L, mpc, num_threads);
      },
      nb::arg("b"), nb::arg("i"), nb::arg("L"), nb::arg("mpc"),
      nb::arg("num_threads"),
      "Assemble the linear form of block i into the vectors of every block");
  m.def(
      "apply_lifting_blocks",
      [](std::vector<nb::ndarray<T, nb::ndim<1>, nb::c_contig>> b, int i,
         std::vector<std::shared_ptr<const dolfinx::fem::Form<T>>>& a,
         const std::vector<nb::ndarray<const std::int8_t, nb::ndim<1>,
                                       nb::c_contig>>& bc_markers1,
         const std::vector<nb::ndarray<const T, nb::ndim<1>, nb::c_contig>>&
             bc_values1,
         const std::vector<nb::ndarray<const T, nb::ndim<1>, nb::c_contig>>& x0,
         T scale,
         std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T, U>>& mpc,
         std::size_t num_threads)
      {
        std::vector<std::span<T>> _b;
        for (auto& b_k : b)
          _b.emplace_back(b_k.data(), b_k.size());
        std::vector<std::span<const std::int8_t>> _markers;
        for (const auto& m : bc_markers1)
          _markers.emplace_back(m.data(), m.size());
        std::vector<std::span<const T>> _values;
        for (const auto& v : bc_values1)
          _values.emplace_back(v.data(), v.size());
        std::vector<std::span<const T>> _x0;
        for (const auto& x : x0)
          _x0.emplace_back(x.data(), x.size());
        dolfinx_mpc::apply_lifting<T, U>(
            dolfinx_mpc::VectorTarget<T>{_b, block_position(i, _b.size()),
                                         mpc->block()},
            a, _markers, _values, _x0, scale, mpc, num_threads);
      },
      nb::arg("b"), nb::arg("i"), nb::arg("a"), nb::arg("bc_markers1"),
      nb::arg("bc_values1"), nb::arg("x0"), nb::arg("scale"), nb::arg("mpc"),
      nb::arg("num_threads"),
      "Lift Dirichlet values into the vector of block i, and the masters of "
      "its constraint into the vector of their block");
  m.def(
      "apply_mpc_lifting_blocks",
      [](std::vector<nb::ndarray<T, nb::ndim<1>, nb::c_contig>> b, int i,
         std::vector<std::shared_ptr<const dolfinx::fem::Form<T>>>& a, T scale,
         std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T, U>>& mpc0,
         std::vector<std::shared_ptr<
             const dolfinx_mpc::MultiPointConstraint<T, U>>>& mpc1,
         std::size_t num_threads)
      {
        std::vector<std::span<T>> _b;
        for (auto& b_k : b)
          _b.emplace_back(b_k.data(), b_k.size());
        dolfinx_mpc::apply_mpc_lifting<T, U>(
            dolfinx_mpc::VectorTarget<T>{_b, block_position(i, _b.size()),
                                         mpc0->block()},
            a, scale, mpc0, mpc1, num_threads);
      },
      nb::arg("b"), nb::arg("i"), nb::arg("a"), nb::arg("scale"),
      nb::arg("mpc0"), nb::arg("mpc1"), nb::arg("num_threads"),
      "Lift the constraint offsets into the vector of block i, and the masters "
      "of its constraint into the vector of their block");
  m.def(
      "create_matrix_nest",
      [](const std::vector<std::vector<const dolfinx::fem::Form<T>*>>& a,
         const std::vector<
             std::shared_ptr<dolfinx_mpc::MultiPointConstraint<T, U>>>& mpcs0,
         const std::vector<
             std::shared_ptr<dolfinx_mpc::MultiPointConstraint<T, U>>>& mpcs1,
         const std::optional<
             std::vector<std::vector<std::optional<std::string>>>>& types)
      { return dolfinx_mpc::create_matrix_nest<T, U>(a, mpcs0, mpcs1, types); },
      nb::rv_policy::take_ownership, nb::arg("a"), nb::arg("mpcs0"),
      nb::arg("mpcs1"), nb::arg("types").none(),
      "Create a nest PETSc Mat for an array of bilinear forms.");
  m.def(
      "insert_diagonal_slaves",
      [](Mat A, const dolfinx_mpc::MultiPointConstraint<T, U>& mpc,
         const T diagval)
      {
        dolfinx_mpc::insert_slave_diagonal<T, U>(
            dolfinx::la::petsc::Matrix::set_fn(A, ADD_VALUES), mpc, diagval);
      },
      nb::arg("A"), nb::arg("mpc"), nb::arg("diagval"),
      "Add a value on the diagonal of each slave row owned by the process");
  m.def(
      "assemble_vector",
      [](nb::ndarray<T, nb::ndim<1>, nb::c_contig> b,
         const dolfinx::fem::Form<T>& L,
         const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T, U>>&
             mpc,
         std::size_t num_threads)
      {
        dolfinx_mpc::assemble_vector(std::span(b.data(), b.size()), L, mpc,
                                     num_threads);
      },
      "b"_a, "L"_a, "mpc"_a, nb::arg("num_threads"),
      "Assemble linear form into an existing vector");

  m.def(
      "apply_lifting",
      [](nb::ndarray<T, nb::ndim<1>, nb::c_contig> b,
         std::vector<std::shared_ptr<const dolfinx::fem::Form<T>>>& a,
         const std::vector<nb::ndarray<const std::int8_t, nb::ndim<1>,
                                       nb::c_contig>>& bc_markers1,
         const std::vector<nb::ndarray<const T, nb::ndim<1>, nb::c_contig>>&
             bc_values1,
         const std::vector<nb::ndarray<const T, nb::ndim<1>, nb::c_contig>>& x0,
         T scale,
         std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T, U>>& mpc,
         std::size_t num_threads)
      {
        std::vector<std::span<const std::int8_t>> _markers;
        for (const auto& m : bc_markers1)
          _markers.emplace_back(m.data(), m.size());
        std::vector<std::span<const T>> _values;
        for (const auto& v : bc_values1)
          _values.emplace_back(v.data(), v.size());
        std::vector<std::span<const T>> _x0;
        for (const auto& x : x0)
          _x0.emplace_back(x.data(), x.size());

        dolfinx_mpc::apply_lifting<T, U>(std::span(b.data(), b.size()), a,
                                         _markers, _values, _x0, scale, mpc,
                                         num_threads);
      },
      nb::arg("b"), nb::arg("a"), nb::arg("bc_markers1"), nb::arg("bc_values1"),
      nb::arg("x0"), nb::arg("scale"), nb::arg("mpc"), nb::arg("num_threads"),
      "Assemble apply lifting from form a on vector b");
  m.def(
      "apply_mpc_lifting",
      [](nb::ndarray<T, nb::ndim<1>, nb::c_contig> b,
         std::vector<std::shared_ptr<const dolfinx::fem::Form<T>>>& a, T scale,
         std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T, U>>& mpc0,
         std::vector<std::shared_ptr<
             const dolfinx_mpc::MultiPointConstraint<T, U>>>& mpc1,
         std::size_t num_threads)
      {
        dolfinx_mpc::apply_mpc_lifting<T, U>(std::span(b.data(), b.size()), a,
                                             scale, mpc0, mpc1, num_threads);
      },
      nb::arg("b"), nb::arg("a"), nb::arg("scale"), nb::arg("mpc0"),
      nb::arg("mpc1"), nb::arg("num_threads"),
      "Lift the inhomogeneity of a multi point constraint into vector b");

  m.def(
      "create_matrix_block",
      [](const std::vector<std::vector<const dolfinx::fem::Form<T>*>>& a,
         const std::vector<
             std::shared_ptr<dolfinx_mpc::MultiPointConstraint<T, U>>>& mpcs0,
         const std::vector<
             std::shared_ptr<dolfinx_mpc::MultiPointConstraint<T, U>>>& mpcs1,
         const std::optional<std::string>& type)
      { return dolfinx_mpc::create_matrix_block<T, U>(a, mpcs0, mpcs1, type); },
      nb::rv_policy::take_ownership, nb::arg("a"), nb::arg("mpcs0"),
      nb::arg("mpcs1"), nb::arg("type").none(),
      "Create a monolithic PETSc Mat for an array of bilinear forms.");
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
  m.def(
      "create_matrix",
      [](const dolfinx::fem::Form<T>& a,
         const std::shared_ptr<dolfinx_mpc::MultiPointConstraint<T, U>>& mpc0,
         const std::shared_ptr<dolfinx_mpc::MultiPointConstraint<T, U>>& mpc1,
         const std::optional<std::string>& type)
      {
        auto A = dolfinx_mpc::create_matrix(a, mpc0, mpc1,
                                            type.value_or(std::string()));
        Mat _A = A.mat();
        PetscObjectReference((PetscObject)_A);
        return _A;
      },
      nb::rv_policy::take_ownership, nb::arg("a"), nb::arg("mpc0"),
      nb::arg("mpc1"), nb::arg("type").none(),
      "Create a PETSc Mat of the given type for bilinear form.");
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
