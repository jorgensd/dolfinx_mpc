// Copyright (C) 2019-2021 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#include "SlipConstraint.h"
#include <dolfinx/common/MPI.h>
#include <xtensor/xadapt.hpp>
#include <xtensor/xsort.hpp>
using namespace dolfinx_mpc;

mpc_data dolfinx_mpc::create_slip_condition(
    std::vector<std::shared_ptr<dolfinx::fem::FunctionSpace>> spaces,
    dolfinx::mesh::MeshTags<std::int32_t> meshtags, std::int32_t marker,
    std::shared_ptr<dolfinx::fem::Function<PetscScalar>> v,
    tcb::span<const std::int32_t> sub_map,
    std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<PetscScalar>>>
        bcs)
{
  // Determine if v is from a sub space
  bool v_in_sub;
  if (spaces.size() == 1)
    v_in_sub = false;
  else if (spaces.size() == 2)
    v_in_sub = true;
  else
    throw std::runtime_error("Number of input spaces can either be 1 or 2.");

  auto mesh = spaces[0]->mesh();
  MPI_Comm comm = mesh->mpi_comm();
  const int rank = dolfinx::MPI::rank(comm);

  // Create view of v as xtensor
  const tcb::span<const PetscScalar>& v_vec = v->x()->array();
  std::array<std::size_t, 1> shape = {v_vec.size()};
  auto v_xt = xt::adapt(v_vec.data(), v_vec.size(), xt::no_ownership(), shape);

  // Info from parent space
  auto W_imap = spaces[0]->dofmap()->index_map;
  std::vector<std::int32_t> W_ghost_owners = W_imap->ghost_owner_rank();
  const int W_bs = spaces[0]->dofmap()->index_map_bs();
  const int W_local_size = W_imap->size_local();

  // Stack all Dirichlet BC dofs in one array, as MPCs cannot be used on
  // Dirichlet dofs
  std::vector<std::int32_t> bc_dofs;
  for (auto bc : bcs)
  {
    auto dof_indices = bc->dof_indices();
    bc_dofs.insert(bc_dofs.end(), dof_indices.first.begin(),
                   dof_indices.first.end());
  }

  // Extract slave_facets
  std::vector<std::int32_t> slave_facets;
  for (std::size_t i = 0; i < meshtags.indices().size(); ++i)
    if (meshtags.values()[i] == marker)
      slave_facets.push_back(meshtags.indices()[i]);

  // Arrays holding MPC data
  std::vector<std::int32_t> slaves;
  std::vector<std::int64_t> masters;
  std::vector<PetscScalar> coeffs;
  std::vector<std::int32_t> owners;
  std::vector<std::int32_t> offsets(1, 0);

  // Get all degrees of freedom in the vector sub-space on the given facets
  std::vector<std::int32_t> entity_dofs = dolfinx::fem::locate_dofs_topological(
      {*v->function_space().get()}, meshtags.dim(), slave_facets);

  // Create array to hold degrees of freedom from the subspace of v for each
  // block
  std::int32_t num_normal_components;
  if (v_in_sub)
    num_normal_components = v->function_space()->dofmap()->index_map_bs();
  else
    num_normal_components = W_bs;
  std::vector<std::int32_t> normal_dofs(num_normal_components);

  // Create map from a given index in a block of v to the parent space
  std::function<std::int32_t(const std::int32_t)> v_to_parent;
  if (v_in_sub)
  {
    v_to_parent = [sub_map, &normal_dofs](const std::int32_t index)
    { return sub_map[normal_dofs[index]]; };
  }
  else
  {
    v_to_parent = [&normal_dofs](const std::int32_t index)
    { return normal_dofs[index]; };
  }
  for (auto block : entity_dofs)
  {
    // Obtain degrees of freedom for normal vector at slave dof
    for (std::int32_t i = 0; i < num_normal_components; ++i)
      normal_dofs[i] = block * num_normal_components + i;
    // Determine slave dof by component with biggest normal vector (to avoid
    // issues with grids aligned with coordiante system)
    auto normal = xt::view(v_xt, xt::keep(normal_dofs));
    std::int32_t slave_index = xt::argmax(xt::abs(normal))(0, 0);

    // Only add slave if no Dirichlet condition has been imposed on the dof
    std::int32_t parent_slave = v_to_parent(slave_index);
    const auto it = std::find(bc_dofs.begin(), bc_dofs.end(), parent_slave);
    if (it == bc_dofs.end())
    {
      std::vector<std::int32_t> parent_masters;
      std::vector<PetscScalar> pair_c;
      std::vector<std::int32_t> pair_o;
      for (std::int32_t i = 0; i < num_normal_components; ++i)
      {
        if (i != slave_index)
        {
          PetscScalar coeff = -normal[i] / normal[slave_index];
          parent_masters.push_back(v_to_parent(i));
          pair_c.push_back(coeff);
          if (v_to_parent(i) < W_local_size * W_bs)
            pair_o.push_back(rank);
          else
            pair_o.push_back(
                W_ghost_owners[v_to_parent(i) / W_bs - W_local_size]);
        }
      }

      std::vector<std::int64_t> pair_m(parent_masters.size());
      // Convert local parent dof to local parent block
      std::vector<std::int32_t> parent_blocks(parent_masters);
      for (std::size_t i = 0; i < parent_blocks.size(); ++i)
        parent_blocks[i] /= W_bs;
      W_imap->local_to_global(parent_blocks, pair_m);
      // Convert global parent block to the dofs
      for (std::size_t i = 0; i < parent_masters.size(); ++i)
      {
        std::div_t div = std::div(parent_masters[i], W_bs);
        pair_m[i] = pair_m[i] * W_bs + div.rem;
      }
      slaves.push_back(parent_slave);
      masters.insert(masters.end(), pair_m.begin(), pair_m.end());
      coeffs.insert(coeffs.end(), pair_c.begin(), pair_c.end());
      offsets.push_back(masters.size());
      owners.insert(owners.end(), pair_o.begin(), pair_o.end());
    }
  }

  mpc_data data;
  data.slaves = slaves;

  data.masters = masters;
  data.offsets = offsets;
  data.owners = owners;
  data.coeffs = coeffs;
  return data;
}