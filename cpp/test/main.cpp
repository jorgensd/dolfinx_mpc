// Copyright (C) 2026 Jørgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#include <catch2/catch_session.hpp>
#include <dolfinx/common/log.h>
#include <mpi.h>

int main(int argc, char* argv[])
{
  // Registers the "dolfinx" logger, without which dolfinx::logger() throws
  dolfinx::init_logging(argc, argv);

  MPI_Init(&argc, &argv);
  int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
