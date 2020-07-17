// Copyright (C) 2020 Jorgen S. Dokken
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
//
// Unit tests for Quadrilateral meshes

#include "test.h"
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/generation/BoxMesh.h>
#include <dolfinx/io/cells.h>
#include <iostream>
#include <ufc.h>
using namespace dolfinx;

int main(int argc, char* argv[])
{
  common::SubSystemsManager::init_logging(argc, argv);
  common::SubSystemsManager::init_petsc(argc, argv);

  // Create mesh and define function space
  std::array<Eigen::Vector3d, 2> pt
      = {Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 1, 1)};
  dolfinx::fem::CoordinateElement cmap = dolfinx::fem::create_coordinate_map(
      create_coordinate_mapping_c1881447d6b8c072bffeb3eb309b78481eb4d309);

  auto mesh = std::make_shared<mesh::Mesh>(generation::BoxMesh::create(
      MPI_COMM_WORLD, pt, {{1, 1, 1}}, cmap, mesh::GhostMode::none));
  auto V
      = fem::create_functionspace(create_functionspace_form_test_a, "u", mesh);

  std::shared_ptr<fem::Form> a = fem::create_form(create_form_test_a, {});
  // Attach 'coordinate mapping' to mesh
  //   auto cmap = a->coordinate_mapping();
  //   mesh->geometry().coord_mapping = cmap;
  Eigen::Array<double, Eigen::Dynamic, 3> x = V->tabulate_dof_coordinates();
  std::cout << x;
  // auto u = std::make_shared<function::Function>(V);

  // u->interpolate([](auto x) {
  //   // std::cout << "xcol" << x.col(0) << "--" << std::endl;
  //   return x.col(0);
  // });
  // VecView(u->vector().vec(), PETSC_VIEWER_STDOUT_SELF);
  // a->set_coefficients({{"u", u}});
  // auto intu = dolfinx::fem::assemble_scalar(*a);

  return 0;
} // namespace
