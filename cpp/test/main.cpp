// Copyright (C) 2026 Jørgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#include <catch2/catch_session.hpp>
#include <dolfinx/common/log.h>
#include <dolfinx_mpc/log.h>
#include <mpi.h>

int main(int argc, char* argv[])
{
  // DOLFINx and DOLFINx-MPC log through named loggers that throw if
  // not registered
  dolfinx::init_logging(argc, argv);
  dolfinx_mpc::init_logging(argc, argv);

  MPI_Init(&argc, &argv);

  Catch::Session session;

  // Catch >= 3.9 defaults to random order; MPI tests need the same
  // order on every rank
  session.configData().runOrder = Catch::TestRunOrder::Declared;

  int result = session.run(argc, argv);
  MPI_Finalize();
  return result;
}
