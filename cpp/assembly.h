// Copyright (C) 2020 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
#include "MultiPointConstraint.h"
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/Form.h>
#include <functional>
#include <petscmat.h>
namespace dolfinx_mpc
{
/// Assemble bilinear form into a matrix
/// @param[in] mat_add The function for adding values into the matrix
/// @param[in] a The bilinear from to assemble
/// @param[in] bcs Boundary conditions to apply. For boundary condition
///  dofs the row and column are zeroed. The diagonal  entry is not set.
void assemble_matrix(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const PetscScalar*)>& mat_add,
    const dolfinx::fem::Form<PetscScalar>& a,
    const dolfinx_mpc::MultiPointConstraint& mpc,
    const std::vector<
        std::shared_ptr<const dolfinx::fem::DirichletBC<PetscScalar>>>& bcs);
// {

//   // Index maps for dof ranges
//   auto map0 = a.function_space(0)->dofmap()->index_map;
//   auto map1 = a.function_space(1)->dofmap()->index_map;

//   // Build dof markers
//   std::vector<bool> dof_marker0, dof_marker1;
//   std::int32_t dim0
//       = map0->block_size() * (map0->size_local() + map0->num_ghosts());
//   std::int32_t dim1
//       = map1->block_size() * (map1->size_local() + map1->num_ghosts());
//   for (std::size_t k = 0; k < bcs.size(); ++k)
//   {
//     assert(bcs[k]);
//     assert(bcs[k]->function_space());
//     if (a.function_space(0)->contains(*bcs[k]->function_space()))
//     {
//       dof_marker0.resize(dim0, false);
//       bcs[k]->mark_dofs(dof_marker0);
//     }
//     if (a.function_space(1)->contains(*bcs[k]->function_space()))
//     {
//       dof_marker1.resize(dim1, false);
//       bcs[k]->mark_dofs(dof_marker1);
//     }
//   }

//   // Assemble
//   dolfinx::fem::impl::assemble_matrix(mat_add, a, dof_marker0,
//   dof_marker1);
// }
} // namespace dolfinx_mpc