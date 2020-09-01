// Copyright (C) 2020 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "assembly.h"
#include <dolfinx/fem/FormIntegrals.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/function/Constant.h>
#include <iostream>

namespace
{
void modify_mpc_cell(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const PetscScalar*)>& mat_set,
    const int num_dofs,
    Eigen::Matrix<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        Ae,
    const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& dofs,
    dolfinx_mpc::MultiPointConstraint& mpc,
    const Eigen::Array<std::int32_t, Eigen::Dynamic, 1> cell_slaves)
{
  // Create copy to use for distribution to master dofs
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Ae_original = Ae;

  for (std::int32_t i = 0; i < cell_slaves.size(); ++i)
  {
    std::cout << cell_slaves[i] << ", ";
  }
  std::cout << "Cell modidified\n";
}
//-----------------------------------------------------------------------------
void assemble_cells_impl(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const PetscScalar*)>& mat_set,
    const dolfinx::mesh::Geometry& geometry,
    const std::vector<std::int32_t>& active_cells,
    const dolfinx::graph::AdjacencyList<std::int32_t>& dofmap0,
    const dolfinx::graph::AdjacencyList<std::int32_t>& dofmap1,
    dolfinx_mpc::MultiPointConstraint& mpc, const std::vector<bool>& bc0,
    const std::vector<bool>& bc1,
    const std::function<void(PetscScalar*, const PetscScalar*,
                             const PetscScalar*, const double*, const int*,
                             const std::uint8_t*, const std::uint32_t)>& kernel,
    const Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                       Eigen::RowMajor>& coeffs,
    const Eigen::Array<PetscScalar, Eigen::Dynamic, 1>& constants,
    const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>& cell_info)
{
  const int gdim = geometry.dim();

  // Prepare cell geometry
  const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
      = geometry.dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = x_dofmap.num_links(0);
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = geometry.x();

  // Data structures used in assembly
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(num_dofs_g, gdim);
  const int num_dofs0 = dofmap0.links(0).size();
  const int num_dofs1 = dofmap1.links(0).size();
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Ae(num_dofs0, num_dofs1);

  // Get slave cells array
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& slave_cells
      = mpc.slave_cells();
  std::vector<bool> is_slave_cell(active_cells.size(), false);
  std::vector<std::int32_t> slave_cell_index(active_cells.size(), -1);
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> cell_to_slaves
      = mpc.cell_to_slaves();
  for (std::int32_t i = 0; i < slave_cells.rows(); ++i)
  {
    is_slave_cell[slave_cells[i]] = true;
    slave_cell_index[slave_cells[i]] = i;
  }

  // Iterate over active cells
  for (std::int32_t k = 0; k < active_cells.size(); ++k)
  {
    // Get cell coordinates/geometry
    const std::int32_t c = active_cells[k];
    auto x_dofs = x_dofmap.links(c);
    for (int i = 0; i < x_dofs.rows(); ++i)
      for (int j = 0; j < gdim; ++j)
        coordinate_dofs(i, j) = x_g(x_dofs[i], j);

    // Tabulate tensor
    std::fill(Ae.data(), Ae.data() + num_dofs0 * num_dofs1, 0);
    kernel(Ae.data(), coeffs.row(c).data(), constants.data(),
           coordinate_dofs.data(), nullptr, nullptr, cell_info[c]);

    // Zero rows/columns for essential bcs
    auto dofs0 = dofmap0.links(c);
    auto dofs1 = dofmap1.links(c);
    if (!bc0.empty())
    {
      for (Eigen::Index i = 0; i < Ae.rows(); ++i)
      {
        if (bc0[dofs0[i]])
          Ae.row(i).setZero();
      }
    }
    if (!bc1.empty())
    {
      for (Eigen::Index j = 0; j < Ae.cols(); ++j)
      {
        if (bc1[dofs1[j]])
          Ae.col(j).setZero();
      }
    }
    if (is_slave_cell[k])
    {
      // Assuming test and trial space has same number of dofs and dofs per
      // cell
      assert(slave_cell_index[k] != -1);
      auto cell_slaves = cell_to_slaves->links(slave_cell_index[k]);
      modify_mpc_cell(mat_set, num_dofs0, Ae, dofs0, mpc, cell_slaves);
    }
    mat_set(dofs0.size(), dofs0.data(), dofs1.size(), dofs1.data(), Ae.data());
  }
}
//-----------------------------------------------------------------------------
void assemble_matrix_impl(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const PetscScalar*)>&
        mat_set_values,
    const dolfinx::fem::Form<PetscScalar>& a,
    dolfinx_mpc::MultiPointConstraint& mpc, const std::vector<bool>& bc0,
    const std::vector<bool>& bc1)
{
  std::shared_ptr<const dolfinx::mesh::Mesh> mesh = a.mesh();
  assert(mesh);
  const int tdim = mesh->topology().dim();
  const std::int32_t num_cells
      = mesh->topology().connectivity(tdim, 0)->num_nodes();

  // Get dofmap data
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap0
      = a.function_space(0)->dofmap();
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap1
      = a.function_space(1)->dofmap();
  assert(dofmap0);
  assert(dofmap1);
  const dolfinx::graph::AdjacencyList<std::int32_t>& dofs0 = dofmap0->list();
  const dolfinx::graph::AdjacencyList<std::int32_t>& dofs1 = dofmap1->list();

  // Prepare constants
  if (!a.all_constants_set())
    throw std::runtime_error("Unset constant in Form");
  const Eigen::Array<PetscScalar, Eigen::Dynamic, 1> constants
      = pack_constants(a);

  // Prepare coefficients
  const Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                     Eigen::RowMajor>
      coeffs = dolfinx::fem::pack_coefficients(a);

  const dolfinx::fem::FormIntegrals<PetscScalar>& integrals = a.integrals();
  const bool needs_permutation_data = integrals.needs_permutation_data();
  if (needs_permutation_data)
    mesh->topology_mutable().create_entity_permutations();
  const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>& cell_info
      = needs_permutation_data
            ? mesh->topology().get_cell_permutation_info()
            : Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>(num_cells);

  for (int i = 0; i < integrals.num_integrals(dolfinx::fem::IntegralType::cell);
       ++i)
  {
    const auto& fn
        = integrals.get_tabulate_tensor(dolfinx::fem::IntegralType::cell, i);
    const std::vector<std::int32_t>& active_cells
        = integrals.integral_domains(dolfinx::fem::IntegralType::cell, i);
    assemble_cells_impl(mat_set_values, mesh->geometry(), active_cells, dofs0,
                        dofs1, mpc, bc0, bc1, fn, coeffs, constants, cell_info);
  }

  if (integrals.num_integrals(dolfinx::fem::IntegralType::exterior_facet) > 0
      or integrals.num_integrals(dolfinx::fem::IntegralType::interior_facet)
             > 0)
  {
    // FIXME: cleanup these calls? Some of the happen internally again.
    mesh->topology_mutable().create_entities(tdim - 1);
    mesh->topology_mutable().create_connectivity(tdim - 1, tdim);

    const int facets_per_cell = dolfinx::mesh::cell_num_entities(
        mesh->topology().cell_type(), tdim - 1);
    const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>& perms
        = needs_permutation_data
              ? mesh->topology().get_facet_permutations()
              : Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>(
                  facets_per_cell, num_cells);

    for (int i = 0; i < integrals.num_integrals(
                        dolfinx::fem::IntegralType::exterior_facet);
         ++i)
    {
      const auto& fn = integrals.get_tabulate_tensor(
          dolfinx::fem::IntegralType::exterior_facet, i);
      const std::vector<std::int32_t>& active_facets
          = integrals.integral_domains(
              dolfinx::fem::IntegralType::exterior_facet, i);
      //   impl::assemble_exterior_facets<T>(mat_set_values, *mesh,
      //   active_facets,
      //                                     dofs0, dofs1, bc0, bc1, fn,
      //                                     coeffs, constants, cell_info,
      //                                     perms);
    }

    const std::vector<int> c_offsets = a.coefficients().offsets();
    for (int i = 0; i < integrals.num_integrals(
                        dolfinx::fem::IntegralType::interior_facet);
         ++i)
    {
      const auto& fn = integrals.get_tabulate_tensor(
          dolfinx::fem::IntegralType::interior_facet, i);
      const std::vector<std::int32_t>& active_facets
          = integrals.integral_domains(
              dolfinx::fem::IntegralType::interior_facet, i);
      //   impl::assemble_interior_facets<T>(
      //       mat_set_values, *mesh, active_facets, *dofmap0, *dofmap1, bc0,
      //       bc1, fn, coeffs, c_offsets, constants, cell_info, perms);
    }
  }
}
} // namespace
//-----------------------------------------------------------------------------

void dolfinx_mpc::assemble_matrix(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const PetscScalar*)>& mat_add,
    const dolfinx::fem::Form<PetscScalar>& a,
    dolfinx_mpc::MultiPointConstraint& mpc,
    const std::vector<
        std::shared_ptr<const dolfinx::fem::DirichletBC<PetscScalar>>>& bcs)
{

  // Index maps for dof ranges
  auto map0 = a.function_space(0)->dofmap()->index_map;
  auto map1 = a.function_space(1)->dofmap()->index_map;

  // Build dof markers
  std::vector<bool> dof_marker0, dof_marker1;
  std::int32_t dim0
      = map0->block_size() * (map0->size_local() + map0->num_ghosts());
  std::int32_t dim1
      = map1->block_size() * (map1->size_local() + map1->num_ghosts());
  for (std::size_t k = 0; k < bcs.size(); ++k)
  {
    assert(bcs[k]);
    assert(bcs[k]->function_space());
    if (a.function_space(0)->contains(*bcs[k]->function_space()))
    {
      dof_marker0.resize(dim0, false);
      bcs[k]->mark_dofs(dof_marker0);
    }
    if (a.function_space(1)->contains(*bcs[k]->function_space()))
    {
      dof_marker1.resize(dim1, false);
      bcs[k]->mark_dofs(dof_marker1);
    }
  }

  // Assemble
  assemble_matrix_impl(mat_add, a, mpc, dof_marker0, dof_marker1);
}