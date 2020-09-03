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
    const Eigen::Array<std::int32_t, Eigen::Dynamic, 1> slave_indices,
    const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>&
        masters,
    const std::shared_ptr<const dolfinx::graph::AdjacencyList<PetscScalar>>&
        coeffs,
    const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& local_indices,
    const std::vector<bool>& is_slave)
{
  dolfinx::common::Timer timer("~MPC: Modify MPC cell (C++)");

  // Arrays for flattened master slave data
  std::vector<std::int32_t> flattened_masters;
  std::vector<std::int32_t> flattened_slaves;
  std::vector<std::int32_t> slaves_loc;
  std::vector<PetscScalar> flattened_coeffs;
  for (std::int32_t i = 0; i < slave_indices.size(); ++i)
  {
    auto local_masters = masters->links(slave_indices[i]);
    auto local_coeffs = coeffs->links(slave_indices[i]);

    for (std::int32_t j = 0; j < local_masters.size(); ++j)
    {
      slaves_loc.push_back(i);
      flattened_slaves.push_back(local_indices[i]);
      flattened_masters.push_back(local_masters[j]);
      flattened_coeffs.push_back(local_coeffs[j]);
    }
  }
  // Matrices used for master insertion
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> m0(1);
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> m1(1);
  Eigen::Matrix<PetscScalar, 1, 1, Eigen::RowMajor> Amaster(1, 1);
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1, Eigen::ColMajor> Arow(num_dofs,
                                                                      1);
  Eigen::Matrix<PetscScalar, 1, Eigen::Dynamic, Eigen::RowMajor> Acol(1,
                                                                      num_dofs);
  Eigen::Matrix<PetscScalar, 1, 1, Eigen::RowMajor> Am0m1;
  // Create copy to use for distribution to master dofs
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Ae_original = Ae;

  // Build matrix where all slave-slave entries are 0 for usage to row and
  // column addition
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Ae_stripped(num_dofs, num_dofs);
  for (std::int32_t i = 0; i < num_dofs; ++i)
  {
    for (std::int32_t j = 0; j < num_dofs; ++j)
    {
      if (is_slave[i] && is_slave[j])
        Ae_stripped(i, j) = 0;
      else
        Ae_stripped(i, j) = Ae(i, j);
    }
  }

  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> mpc_dofs(dofs.size());
  for (std::int32_t i = 0; i < flattened_slaves.size(); ++i)
  {
    const std::int32_t& local_index = flattened_slaves[i];
    const std::int32_t& master = flattened_masters[i];
    const PetscScalar& coeff = flattened_coeffs[i];

    // Remove contributions form Ae
    Ae.col(local_index).setZero();
    Ae.row(local_index).setZero();

    m0(0) = master;
    Arow.col(0) = coeff * Ae_stripped.col(local_index);
    Acol.row(0) = coeff * Ae_stripped.row(local_index);
    Amaster(0, 0) = coeff * coeff * Ae_original(local_index, local_index);
    mpc_dofs = dofs;
    mpc_dofs[local_index] = master;
    mat_set(num_dofs, mpc_dofs.data(), 1, m0.data(), Arow.data());
    mat_set(1, m0.data(), num_dofs, mpc_dofs.data(), Acol.data());
    mat_set(1, m0.data(), 1, m0.data(), Amaster.data());

    // Loop through other masters on the same cell
    std::vector<std::int32_t> other_slaves(flattened_slaves.size() - 1);
    std::iota(std::begin(other_slaves), std::end(other_slaves), 0);
    if (i < flattened_slaves.size() - 1)
      other_slaves[i] = flattened_slaves.size() - 1;
    for (auto j : other_slaves)
    {
      const std::int32_t& other_local_index = flattened_slaves[j];
      const std::int32_t& other_master = flattened_masters[j];
      const PetscScalar& other_coeff = flattened_coeffs[j];
      const PetscScalar c0c1 = coeff * other_coeff;
      m1(0) = other_master;

      if (slaves_loc[i] != slaves_loc[j])
      {
        // Add contributions for masters on the same cell (different slave)
        Am0m1(0, 0) = c0c1 * Ae_original(local_index, other_local_index);
        mat_set(1, m0.data(), 1, m1.data(), Am0m1.data());
      }
      else
      {
        // Add contributions for different masters of the same slave
        Am0m1(0, 0) = c0c1 * Ae_original(local_index, local_index);
        mat_set(1, m0.data(), 1, m1.data(), Am0m1.data());
      }
    }
  }
} // namespace
//-----------------------------------------------------------------------------
void assemble_cells_impl(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const PetscScalar*)>& mat_set,
    const dolfinx::mesh::Geometry& geometry,
    const std::vector<std::int32_t>& active_cells,
    const dolfinx::graph::AdjacencyList<std::int32_t>& dofmap0,
    const dolfinx::graph::AdjacencyList<std::int32_t>& dofmap1,
    const std::vector<bool>& bc0, const std::vector<bool>& bc1,
    const std::function<void(PetscScalar*, const PetscScalar*,
                             const PetscScalar*, const double*, const int*,
                             const std::uint8_t*, const std::uint32_t)>& kernel,
    const Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                       Eigen::RowMajor>& coeffs,
    const Eigen::Array<PetscScalar, Eigen::Dynamic, 1>& constants,
    const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>& cell_info,
    const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& slaves,
    const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
        masters,
    const std::shared_ptr<const dolfinx::graph::AdjacencyList<PetscScalar>>&
        coefficients,
    const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& slave_cells,
    const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>&
        cell_to_slaves)
{
  dolfinx::common::Timer timer("~MPC (C++): Assemble cells");

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

  // Compute local indices for slave cells
  std::vector<bool> is_slave_cell(active_cells.size(), false);
  std::vector<std::int32_t> slave_cell_index(active_cells.size(), -1);

  for (std::int32_t i = 0; i < slave_cells.rows(); ++i)
  {
    for (std::int32_t j = 0; j < active_cells.size(); ++j)
      if (slave_cells[i] == active_cells[j])
      {
        is_slave_cell[j] = true;
        slave_cell_index[j] = i;
        break;
      }
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
      auto slave_indices = cell_to_slaves->links(slave_cell_index[k]);
      // Find local position of every slave
      std::vector<bool> is_slave(num_dofs0, false);
      Eigen::Array<std::int32_t, Eigen::Dynamic, 1> local_indices(
          slave_indices.size());
      for (std::int32_t i = 0; i < slave_indices.size(); ++i)
      {
        for (std::int32_t j = 0; j < dofs0.size(); ++j)
        {
          if (dofs0[j] == slaves[slave_indices[i]])
          {
            local_indices[i] = j;
            is_slave[j] = true;
            break;
          }
        }
      }

      modify_mpc_cell(mat_set, num_dofs0, Ae, dofs0, slave_indices, masters,
                      coefficients, local_indices, is_slave);
    }
    mat_set(dofs0.size(), dofs0.data(), dofs1.size(), dofs1.data(), Ae.data());
  }
}
//-----------------------------------------------------------------------------
void assemble_matrix_impl(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const PetscScalar*)>&
        mat_set_values,
    const dolfinx::fem::Form<PetscScalar>& a, const std::vector<bool>& bc0,
    const std::vector<bool>& bc1,
    const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& slaves,
    const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
        masters,
    const std::shared_ptr<const dolfinx::graph::AdjacencyList<PetscScalar>>&
        coefficients,
    const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& slave_cells,
    const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>&
        cell_to_slaves)
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
                        dofs1, bc0, bc1, fn, coeffs, constants, cell_info,
                        slaves, masters, coefficients, slave_cells,
                        cell_to_slaves);
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
      throw std::runtime_error("Not implemented yet");

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
      throw std::runtime_error("Not implemented yet");

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
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint>& mpc,
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
  const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
      masters = mpc->masters_local();
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& slaves = mpc->slaves();
  const std::shared_ptr<const dolfinx::graph::AdjacencyList<PetscScalar>>&
      coefficients
      = mpc->coeffs();
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& slave_cells
      = mpc->slave_cells();
  const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>&
      cell_to_slaves
      = mpc->cell_to_slaves();

  assemble_matrix_impl(mat_add, a, dof_marker0, dof_marker1, slaves, masters,
                       coefficients, slave_cells, cell_to_slaves);
}