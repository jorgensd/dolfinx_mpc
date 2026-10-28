// Copyright (C) 2026 Jørgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#pragma once

#include <spdlog/spdlog.h>

namespace dolfinx_mpc
{
/// @brief Initialise the DOLFINx-MPC logger.
///
/// Must be called before any DOLFINx-MPC function that logs. The level
/// defaults to warn and can be set from the command line with
/// `SPDLOG_LEVEL=dolfinx_mpc=<level>`.
///
/// @param[in] argc Number of command line arguments.
/// @param[in] argv Command line argument vector.
void init_logging(int argc, char* argv[]);

/// @brief DOLFINx-MPC's logger, used for all logging in the
/// `dolfinx_mpc` namespace.
spdlog::logger& logger();

} // namespace dolfinx_mpc
