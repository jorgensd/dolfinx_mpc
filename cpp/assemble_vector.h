// Copyright (C) 2021 Jorgen S. Dokken and Nathan Sime
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "MultiPointConstraint.h"
#include "assemble_utils.h"
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/Form.h>
#include <functional>

namespace dolfinx_mpc
{
template <typename T>
class MultiPointConstraint;

/// Given a local element vector, move all slave contributions to the global
/// (local to process) vector.
/// @param [in, out] b The global (local to process) vector
/// @param [in, out] b_local The local element vector
/// @param [in] b_local_copy Copy of the local element vector
/// @param [in] cell_blocks Dofmap for the blocks in the cell
/// @param [in] num_dofs The number of degrees of freedom in the local vector
/// @param [in] bs The element block size
/// @param [in] is_slave Vector indicating if a dof is a slave
/// @param [in] slaves The slave dofs (local to process)
/// @param [in] masters Adjacency list with master dofs
/// @param [in] coeffs Adjacency list with the master coefficients
template <typename T>
void modify_mpc_vec(
    const std::span<T>& b, const std::span<T>& b_local,
    const std::span<T>& b_local_copy,
    const std::span<const std::int32_t>& cell_blocks, const int num_dofs,
    const int bs, const std::vector<std::int8_t>& is_slave,
    const std::span<const std::int32_t>& slaves,
    const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>&
        masters,
    const std::shared_ptr<const dolfinx::graph::AdjacencyList<T>>& coeffs)
{

  // NOTE: Should this be moved into the MPC constructor?
  // Get local index of slaves in cell
  std::vector<std::int32_t> local_index
      = compute_local_slave_index(slaves, num_dofs, bs, cell_blocks, is_slave);

  // Move contribution from each slave to corresponding master dof
  for (std::size_t i = 0; i < local_index.size(); i++)
  {
    auto masters_i = masters->links(slaves[i]);
    auto coeffs_i = coeffs->links(slaves[i]);
    assert(masters_i.size() == coeffs_i.size());
    for (std::size_t j = 0; j < masters_i.size(); j++)
    {
      b[masters_i[j]] += coeffs_i[j] * b_local_copy[local_index[i]];
      b_local[local_index[i]] = 0;
    }
  }
}

/// Assemble a linear form into a vector
/// @param[in] b The vector to be assembled. It will not be zeroed before
/// assembly.
/// @param[in] L The linear forms to assemble into b
/// @param[in] mpc The multi-point constraint
void assemble_vector(
    std::span<double> b, const dolfinx::fem::Form<double>& L,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<double>>&
        mpc);

/// Assemble a linear form into a vector
/// @param[in] b The vector to be assembled. It will not be zeroed before
/// assembly.
/// @param[in] L The linear forms to assemble into b
/// @param[in] mpc The multi-point constraint
void assemble_vector(
    std::span<std::complex<double>> b,
    const dolfinx::fem::Form<std::complex<double>>& L,
    const std::shared_ptr<
        const dolfinx_mpc::MultiPointConstraint<std::complex<double>>>& mpc);

} // namespace dolfinx_mpc
