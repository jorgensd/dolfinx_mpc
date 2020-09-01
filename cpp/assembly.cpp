// Copyright (C) 2020 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "assembly.h"
#include <iostream>

void dolfinx_mpc::assemble_matrix(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const PetscScalar*)>& mat_add,
    const dolfinx::fem::Form<PetscScalar>& a,
    const dolfinx_mpc::MultiPointConstraint& mpc,
    const std::vector<
        std::shared_ptr<const dolfinx::fem::DirichletBC<PetscScalar>>>& bcs)
{
  std::cout << "HERE!!\n\n";
}