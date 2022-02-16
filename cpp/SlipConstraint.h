// Copyright (C) 2019-2021 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#include "ContactConstraint.h"
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/mesh/MeshTags.h>
namespace dolfinx_mpc
{

//
mpc_data create_slip_condition(
    std::shared_ptr<dolfinx::fem::FunctionSpace>& space,
    const dolfinx::mesh::MeshTags<std::int32_t>& meshtags, std::int32_t marker,
    const dolfinx::fem::Function<PetscScalar>& v,
    std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<PetscScalar>>>
        bcs,
    const bool sub_space);

} // namespace dolfinx_mpc