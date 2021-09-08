// Copyright (C) 2020-2021 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "assemble_matrix.h"
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/utils.h>

namespace
{
template <typename T>
void modify_mpc_cell(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const T*)>& mat_set,
    const int num_dofs, xt::xtensor<T, 2>& Ae,
    const xtl::span<const int32_t>& dofs, int bs,
    const xtl::span<const int32_t>& slave_indices,
    const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>&
        masters,
    const std::shared_ptr<const dolfinx::graph::AdjacencyList<T>>& coeffs,
    const std::vector<std::int32_t>& local_indices,
    const std::vector<bool>& is_slave)
{

  // Arrays for flattened master slave data
  std::vector<std::int32_t> flattened_masters;
  std::vector<std::int32_t> flattened_slaves;
  std::vector<std::int32_t> slaves_loc;
  std::vector<T> flattened_coeffs;
  for (std::int32_t i = 0; i < slave_indices.size(); ++i)
  {
    xtl::span<const std::int32_t> local_masters
        = masters->links(slave_indices[i]);
    xtl::span<const T> local_coeffs = coeffs->links(slave_indices[i]);

    for (std::int32_t j = 0; j < local_masters.size(); ++j)
    {
      slaves_loc.push_back(i);
      flattened_slaves.push_back(local_indices[i]);
      flattened_masters.push_back(local_masters[j]);
      flattened_coeffs.push_back(local_coeffs[j]);
    }
  }
  // Matrices used for master insertion
  std::array<std::int32_t, 1> m0;
  std::array<std::int32_t, 1> m1;
  std::array<T, 1> Amaster;
  xt::xarray<T> Arow(bs * num_dofs);
  xt::xarray<T> Acol(bs * num_dofs);
  xt::xtensor_fixed<T, xt::xshape<1>> Am0m1;

  // Create copy to use for distribution to master dofs
  xt::xtensor<T, 2> Ae_original = Ae;
  // Build matrix where all slave-slave entries are 0 for usage to row and
  // column addition
  xt::xtensor<T, 2> Ae_stripped = xt::empty<T>({bs * num_dofs, bs * num_dofs});
  for (std::int32_t i = 0; i < bs * num_dofs; ++i)
    for (std::int32_t j = 0; j < bs * num_dofs; ++j)
      Ae_stripped(i, j) = (!(is_slave[i] && is_slave[j])) * Ae(i, j);

  std::vector<std::int32_t> mpc_dofs(bs * dofs.size());
  for (std::int32_t i = 0; i < flattened_slaves.size(); ++i)
  {
    const std::int32_t& local_index = flattened_slaves[i];
    const std::int32_t& master = flattened_masters[i];
    const T& coeff = flattened_coeffs[i];

    // Remove contributions form Ae
    xt::col(Ae, local_index) = xt::zeros<T>({num_dofs * bs});
    xt::row(Ae, local_index).fill(0);

    m0[0] = master;
    Arow = coeff * xt::col(Ae_stripped, local_index);
    Acol = coeff * xt::row(Ae_stripped, local_index);
    Amaster[0] = coeff * coeff * Ae_original(local_index, local_index);
    // Unroll dof blocks
    for (std::int32_t j = 0; j < dofs.size(); ++j)
      for (std::int32_t k = 0; k < bs; ++k)
        mpc_dofs[j * bs + k] = dofs[j] * bs + k;

    // Insert modified entries
    mpc_dofs[local_index] = master;
    mat_set(bs * num_dofs, mpc_dofs.data(), 1, m0.data(), Arow.data());
    mat_set(1, m0.data(), bs * num_dofs, mpc_dofs.data(), Acol.data());
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
      const T& other_coeff = flattened_coeffs[j];
      const T c0c1 = coeff * other_coeff;
      m1[0] = other_master;

      if (slaves_loc[i] != slaves_loc[j])
      {
        // Add contributions for masters on the same cell (different slave)
        Am0m1(0) = c0c1 * Ae_original(local_index, other_local_index);
        mat_set(1, m0.data(), 1, m1.data(), Am0m1.data());
      }
      else
      {
        // Add contributions for different masters of the same slave
        Am0m1(0) = c0c1 * Ae_original(local_index, local_index);
        mat_set(1, m0.data(), 1, m1.data(), Am0m1.data());
      }
    }
  }
} // namespace

//-----------------------------------------------------------------------------
template <typename T>
void assemble_exterior_facets(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const T*)>&
        mat_add_block_values,
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const T*)>& mat_add_values,
    const dolfinx::mesh::Mesh& mesh,
    const std::vector<std::pair<std::int32_t, int>>& facets,
    const std::function<void(const xtl::span<T>&,
                             const xtl::span<const std::uint32_t>&,
                             std::int32_t, int)>& apply_dof_transformation,
    const dolfinx::graph::AdjacencyList<std::int32_t>& dofmap0, size_t bs0,
    const std::function<
        void(const xtl::span<T>&, const xtl::span<const std::uint32_t>&,
             std::int32_t, int)>& apply_dof_transformation_to_transpose,
    const dolfinx::graph::AdjacencyList<std::int32_t>& dofmap1, size_t bs1,
    const std::vector<bool>& bc0, const std::vector<bool>& bc1,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*)>& kernel,
    const dolfinx::array2d<T>& coeffs, const std::vector<T>& constants,
    const xtl::span<const std::uint32_t>& cell_info,
    const std::function<std::uint8_t(std::size_t)>& get_perm,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T>>& mpc)
{

  // Get MPC data
  const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
      masters = mpc->masters_local();
  const std::shared_ptr<const dolfinx::graph::AdjacencyList<T>>& coefficients
      = mpc->coeffs();
  const xtl::span<const std::int32_t> slaves = mpc->slaves();

  xtl::span<const std::int32_t> slave_cells = mpc->slave_cells();
  const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>&
      cell_to_slaves
      = mpc->cell_to_slaves();

  const int tdim = mesh.topology().dim();
  const int num_cell_facets
      = dolfinx::mesh::cell_num_entities(mesh.topology().cell_type(), tdim - 1);

  // Prepare cell geometry
  const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh.geometry().dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = x_dofmap.num_links(0);
  const xt::xtensor<double, 2>& x_g = mesh.geometry().x();

  // Compute local indices for slave cells
  std::vector<bool> is_slave_facet(facets.size(), false);
  std::vector<std::int32_t> slave_cell_index(facets.size(), -1);
  for (std::int32_t i = 0; i < facets.size(); ++i)
  {
    const std::int32_t cell = facets[i].first;
    for (std::int32_t j = 0; j < slave_cells.size(); ++j)
      if (slave_cells[j] == cell)
      {
        is_slave_facet[i] = true;
        slave_cell_index[i] = j;
        break;
      }
  }

  // Iterate over all facets
  const size_t num_dofs0 = dofmap0.links(0).size();
  const size_t num_dofs1 = dofmap1.links(0).size();
  const std::uint32_t ndim0 = bs0 * num_dofs0;
  const std::uint32_t ndim1 = bs1 * num_dofs1;
  xt::xtensor<T, 2> Ae({ndim0, ndim1});
  const xtl::span<T> _Ae(Ae);
  std::vector<double> coordinate_dofs(3 * num_dofs_g);

  for (std::int32_t l = 0; l < facets.size(); ++l)
  {

    const std::int32_t cell = facets[l].first;
    const int local_facet = facets[l].second;

    // Get cell vertex coordinates
    xtl::span<const std::int32_t> x_dofs = x_dofmap.links(cell);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(xt::row(x_g, x_dofs[i]).begin(), 3,
                  std::next(coordinate_dofs.begin(), 3 * i));
    }
    // Tabulate tensor
    std::uint8_t perm = get_perm(cell * num_cell_facets + local_facet);
    std::fill(Ae.data(), Ae.data() + Ae.size(), 0);
    kernel(Ae.data(), coeffs.row(cell).data(), constants.data(),
           coordinate_dofs.data(), &local_facet, &perm);
    apply_dof_transformation(_Ae, cell_info, cell, ndim1);
    apply_dof_transformation_to_transpose(_Ae, cell_info, cell, ndim0);

    // Zero rows/columns for essential bcs
    xtl::span<const std::int32_t> dmap0 = dofmap0.links(cell);
    xtl::span<const std::int32_t> dmap1 = dofmap1.links(cell);
    if (!bc0.empty())
    {
      for (std::int32_t i = 0; i < num_dofs0; ++i)
      {
        for (std::int32_t k = 0; k < bs0; ++k)
        {
          if (bc0[bs0 * dmap0[i] + k])
            xt::row(Ae, bs0 * i + k).fill(0);
        }
      }
    }

    if (!bc1.empty())
    {
      for (std::int32_t j = 0; j < num_dofs1; ++j)
      {
        for (std::int32_t k = 0; k < bs1; ++k)
        {
          if (bc1[bs1 * dmap1[j] + k])
            xt::col(Ae, bs1 * j + k) = xt::zeros<T>({bs1 * num_dofs1});
        }
      }
    }
    if (is_slave_facet[l])
    {
      // Assuming test and trial space has same number of dofs and dofs per
      // cell
      xtl::span<const std::int32_t> slave_indices
          = cell_to_slaves->links(slave_cell_index[l]);
      // Find local position of every slave
      std::vector<bool> is_slave(num_dofs0, false);
      std::vector<std::int32_t> local_indices(slave_indices.size());
      for (std::int32_t i = 0; i < slave_indices.size(); ++i)
      {
        for (std::int32_t j = 0; j < dmap0.size(); ++j)
        {
          bool found = false;
          for (std::int32_t k = 0; k < bs0; ++k)
          {
            if (bs0 * dmap0[j] + k == slaves[slave_indices[i]])
            {
              local_indices[i] = j * bs0 + k;
              is_slave[j * bs0 + k] = true;
              found = true;
              break;
            }
          }
          if (found)
            break;
        }
      }
      modify_mpc_cell<T>(mat_add_values, num_dofs0, Ae, dmap0, bs0,
                         slave_indices, masters, coefficients, local_indices,
                         is_slave);
    }
    mat_add_block_values(dmap0.size(), dmap0.data(), dmap1.size(), dmap1.data(),
                         Ae.data());
  }
} // namespace
//-----------------------------------------------------------------------------
template <typename T>
void assemble_cells_impl(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const T*)>&
        mat_add_block_values,
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const T*)>& mat_add_values,
    const dolfinx::mesh::Geometry& geometry,
    const std::vector<std::int32_t>& active_cells,
    std::function<void(xtl::span<T>, const xtl::span<const std::uint32_t>,
                       const std::int32_t, const int)>
        apply_dof_transformation,
    const dolfinx::graph::AdjacencyList<std::int32_t>& dofmap0, int bs0,
    std::function<void(xtl::span<T>, const xtl::span<const std::uint32_t>,
                       const std::int32_t, const int)>
        apply_dof_transformation_to_transpose,
    const dolfinx::graph::AdjacencyList<std::int32_t>& dofmap1, int bs1,
    const std::vector<bool>& bc0, const std::vector<bool>& bc1,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*)>& kernel,
    const dolfinx::array2d<T>& coeffs, const std::vector<T>& constants,
    const xtl::span<const std::uint32_t>& cell_info,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T>>& mpc)
{
  dolfinx::common::Timer timer("~MPC (C++): Assemble cells");

  // Get MPC data
  const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
      masters = mpc->masters_local();
  const std::shared_ptr<const dolfinx::graph::AdjacencyList<T>>& coefficients
      = mpc->coeffs();
  const xtl::span<const std::int32_t> slaves = mpc->slaves();

  xtl::span<const std::int32_t> slave_cells = mpc->slave_cells();
  const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>&
      cell_to_slaves
      = mpc->cell_to_slaves();

  // Prepare cell geometry
  const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
      = geometry.dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = x_dofmap.num_links(0);
  const xt::xtensor<double, 2>& x_g = geometry.x();

  // Compute local indices for slave cells
  std::vector<bool> is_slave_cell(active_cells.size(), false);
  std::vector<std::int32_t> slave_cell_index(active_cells.size(), -1);
  for (std::int32_t i = 0; i < slave_cells.size(); ++i)
  {
    for (std::int32_t j = 0; j < active_cells.size(); ++j)
    {
      if (slave_cells[i] == active_cells[j])
      {
        is_slave_cell[j] = true;
        slave_cell_index[j] = i;
        break;
      }
    }
  }

  // Iterate over active cells
  std::vector<double> coordinate_dofs(3 * num_dofs_g);
  const int num_dofs0 = dofmap0.links(0).size();
  const int num_dofs1 = dofmap1.links(0).size();
  const size_t ndim0 = num_dofs0 * bs0;
  const size_t ndim1 = num_dofs1 * bs1;
  xt::xtensor<T, 2> Ae({ndim0, ndim1});
  const xtl::span<T> _Ae(Ae);
  for (std::int32_t l = 0; l < active_cells.size(); ++l)
  {
    // Get cell coordinates/geometry
    const std::int32_t c = active_cells[l];
    xtl::span<const int32_t> x_dofs = x_dofmap.links(c);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(xt::row(x_g, x_dofs[i]).begin(), 3,
                  std::next(coordinate_dofs.begin(), 3 * i));
    }
    // Tabulate tensor
    std::fill(Ae.data(), Ae.data() + Ae.size(), 0);
    kernel(Ae.data(), coeffs.row(c).data(), constants.data(),
           coordinate_dofs.data(), nullptr, nullptr);
    apply_dof_transformation(_Ae, cell_info, c, ndim1);
    apply_dof_transformation_to_transpose(_Ae, cell_info, c, ndim0);

    // Zero rows/columns for essential bcs
    xtl::span<const int32_t> dofs0 = dofmap0.links(c);
    xtl::span<const int32_t> dofs1 = dofmap1.links(c);
    if (!bc0.empty())
    {
      for (std::int32_t i = 0; i < num_dofs0; ++i)
      {
        for (std::int32_t k = 0; k < bs0; ++k)
        {
          if (bc0[bs0 * dofs0[i] + k])
            xt::row(Ae, bs0 * i + k).fill(0);
        }
      }
    }
    if (!bc1.empty())
    {
      for (std::int32_t j = 0; j < num_dofs1; ++j)
      {
        for (std::int32_t k = 0; k < bs1; ++k)
        {
          if (bc1[bs1 * dofs1[j] + k])
            xt::col(Ae, bs1 * j + k) = xt::zeros<T>({num_dofs1 * bs1});
        }
      }
    }
    if (is_slave_cell[l])
    {
      // Assuming test and trial space has same number of dofs and dofs per
      // cell
      xtl::span<const int32_t> slave_indices
          = cell_to_slaves->links(slave_cell_index[l]);
      // Find local position of every slave
      std::vector<bool> is_slave(bs0 * num_dofs0, false);
      std::vector<std::int32_t> local_indices(slave_indices.size());
      for (std::int32_t i = 0; i < slave_indices.size(); ++i)
      {
        bool found = false;
        for (std::int32_t j = 0; j < dofs0.size(); ++j)
        {
          for (std::int32_t k = 0; k < bs0; ++k)
          {
            if (bs0 * dofs0[j] + k == slaves[slave_indices[i]])
            {
              local_indices[i] = bs0 * j + k;
              is_slave[j * bs0 + k] = true;
              break;
            }
          }
          if (found)
            break;
        }
      }
      modify_mpc_cell<T>(mat_add_values, num_dofs0, Ae, dofs0, bs0,
                         slave_indices, masters, coefficients, local_indices,
                         is_slave);
    }
    mat_add_block_values(dofs0.size(), dofs0.data(), dofs1.size(), dofs1.data(),
                         Ae.data());
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void assemble_matrix_impl(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const T*)>&
        mat_add_block_values,
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const T*)>& mat_add_values,
    const dolfinx::fem::Form<T>& a, const std::vector<bool>& bc0,
    const std::vector<bool>& bc1,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T>>& mpc)
{
  std::shared_ptr<const dolfinx::mesh::Mesh> mesh = a.mesh();
  assert(mesh);
  const int tdim = mesh->topology().dim();
  const std::int32_t num_cells
      = mesh->topology().connectivity(tdim, 0)->num_nodes();

  // Get dofmap data
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap0
      = a.function_spaces().at(0)->dofmap();
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap1
      = a.function_spaces().at(1)->dofmap();
  assert(dofmap0);
  assert(dofmap1);
  const dolfinx::graph::AdjacencyList<std::int32_t>& dofs0 = dofmap0->list();
  const int bs0 = dofmap0->bs();
  const dolfinx::graph::AdjacencyList<std::int32_t>& dofs1 = dofmap1->list();
  const int bs1 = dofmap1->bs();
  // Prepare constants
  const std::vector<T> constants = pack_constants(a);

  // Prepare coefficients
  const dolfinx::array2d<T> coeffs = dolfinx::fem::pack_coefficients(a);

  std::shared_ptr<const dolfinx::fem::FiniteElement> element0
      = a.function_spaces().at(0)->element();
  std::shared_ptr<const dolfinx::fem::FiniteElement> element1
      = a.function_spaces().at(1)->element();
  std::function<void(xtl::span<T>, const xtl::span<const std::uint32_t>,
                     const std::int32_t, const int)>
      apply_dof_transformation = element0->get_dof_transformation_function<T>();
  std::function<void(xtl::span<T>, const xtl::span<const std::uint32_t>,
                     const std::int32_t, const int)>
      apply_dof_transformation_to_transpose
      = element1->get_dof_transformation_to_transpose_function<T>();

  const bool needs_transformation_data
      = element0->needs_dof_transformations()
        or element1->needs_dof_transformations()
        or a.needs_facet_permutations();
  xtl::span<const std::uint32_t> cell_info;
  if (needs_transformation_data)
  {
    mesh->topology_mutable().create_entity_permutations();
    cell_info = xtl::span(mesh->topology().get_cell_permutation_info());
  }
  for (int i : a.integral_ids(dolfinx::fem::IntegralType::cell))
  {
    const auto& fn = a.kernel(dolfinx::fem::IntegralType::cell, i);
    const std::vector<std::int32_t>& active_cells = a.cell_domains(i);
    assemble_cells_impl<T>(mat_add_block_values, mat_add_values,
                           mesh->geometry(), active_cells,
                           apply_dof_transformation, dofs0, bs0,
                           apply_dof_transformation_to_transpose, dofs1, bs1,
                           bc0, bc1, fn, coeffs, constants, cell_info, mpc);
  }
  if (a.num_integrals(dolfinx::fem::IntegralType::exterior_facet) > 0
      or a.num_integrals(dolfinx::fem::IntegralType::interior_facet) > 0)
  {
    mesh->topology_mutable().create_connectivity(tdim - 1, tdim);
    mesh->topology_mutable().create_entity_permutations();

    const int facets_per_cell = dolfinx::mesh::cell_num_entities(
        mesh->topology().cell_type(), tdim - 1);
    std::function<std::uint8_t(std::size_t)> get_perm;
    if (a.needs_facet_permutations())
    {
      mesh->topology_mutable().create_entity_permutations();
      const std::vector<std::uint8_t>& perms
          = mesh->topology().get_facet_permutations();
      get_perm = [&perms](std::size_t i) { return perms[i]; };
    }
    else
      get_perm = [](std::size_t) { return 0; };

    for (int i : a.integral_ids(dolfinx::fem::IntegralType::exterior_facet))
    {
      const auto& fn = a.kernel(dolfinx::fem::IntegralType::exterior_facet, i);
      const std::vector<std::pair<std::int32_t, int>>& facets
          = a.exterior_facet_domains(i);
      assemble_exterior_facets<T>(mat_add_block_values, mat_add_values, *mesh,
                                  facets, apply_dof_transformation, dofs0, bs0,
                                  apply_dof_transformation_to_transpose, dofs1,
                                  bs1, bc0, bc1, fn, coeffs, constants,
                                  cell_info, get_perm, mpc);
    }

    const std::vector<int> c_offsets = a.coefficient_offsets();
    for (int i : a.integral_ids(dolfinx::fem::IntegralType::interior_facet))
    {
      const auto& fn = a.kernel(dolfinx::fem::IntegralType::interior_facet, i);
      const std::vector<std::tuple<std::int32_t, int, std::int32_t, int>>&
          active_facets
          = a.interior_facet_domains(i);
      throw std::runtime_error("Not implemented yet");

      //   impl::assemble_interior_facets(
      //       mat_set_values, *mesh, active_facets, *dofmap0, *dofmap1, bc0,
      //       bc1, fn, coeffs, c_offsets, constants, cell_info, perms);
    }
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void _assemble_matrix(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const T*)>& mat_add_block,
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const T*)>& mat_add,
    const dolfinx::fem::Form<T>& a,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T>>& mpc,
    const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>>& bcs,
    const T diagval)
{
  dolfinx::common::Timer timer_s("~MPC: Assembly (C++)");

  // Index maps for dof ranges
  std::shared_ptr<const dolfinx::common::IndexMap> map0
      = a.function_spaces().at(0)->dofmap()->index_map;
  std::shared_ptr<const dolfinx::common::IndexMap> map1
      = a.function_spaces().at(1)->dofmap()->index_map;
  int bs0 = a.function_spaces().at(0)->dofmap()->index_map_bs();
  int bs1 = a.function_spaces().at(1)->dofmap()->index_map_bs();

  // Build dof markers
  std::vector<bool> dof_marker0, dof_marker1;
  std::int32_t dim0 = bs0 * (map0->size_local() + map0->num_ghosts());
  std::int32_t dim1 = bs1 * (map1->size_local() + map1->num_ghosts());
  for (std::size_t k = 0; k < bcs.size(); ++k)
  {
    assert(bcs[k]);
    assert(bcs[k]->function_space());
    if (a.function_spaces().at(0)->contains(*bcs[k]->function_space()))
    {
      dof_marker0.resize(dim0, false);
      bcs[k]->mark_dofs(dof_marker0);
    }
    if (a.function_spaces().at(1)->contains(*bcs[k]->function_space()))
    {
      dof_marker1.resize(dim1, false);
      bcs[k]->mark_dofs(dof_marker1);
    }
  }

  // Assemble
  assemble_matrix_impl<T>(mat_add_block, mat_add, a, dof_marker0, dof_marker1,
                          mpc);

  // Add diagval on diagonal for slave dofs
  const xtl::span<const std::int32_t> slaves = mpc->slaves();
  const std::int32_t num_local_slaves = mpc->num_local_slaves();
  std::vector<std::int32_t> diag_dof(1);
  std::vector<T> diag_value(1);
  diag_value[0] = diagval;
  for (std::int32_t i = 0; i < num_local_slaves; ++i)
  {
    diag_dof[0] = slaves[i];
    mat_add(1, diag_dof.data(), 1, diag_dof.data(), diag_value.data());
  }
}
} // namespace
//-----------------------------------------------------------------------------
void dolfinx_mpc::assemble_matrix(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const double*)>& mat_add_block,
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const double*)>& mat_add,
    const dolfinx::fem::Form<double>& a,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<double>>& mpc,
    const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<double>>>&
        bcs,
    const double diagval)
{
  _assemble_matrix(mat_add_block, mat_add, a, mpc, bcs, diagval);
}
//-----------------------------------------------------------------------------
void dolfinx_mpc::assemble_matrix(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const std::complex<double>*)>&
        mat_add_block,
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const std::complex<double>*)>&
        mat_add,
    const dolfinx::fem::Form<std::complex<double>>& a,
    const std::shared_ptr<
        const dolfinx_mpc::MultiPointConstraint<std::complex<double>>>& mpc,
    const std::vector<
        std::shared_ptr<const dolfinx::fem::DirichletBC<std::complex<double>>>>&
        bcs,
    const std::complex<double> diagval)
{
  _assemble_matrix(mat_add_block, mat_add, a, mpc, bcs, diagval);
}
