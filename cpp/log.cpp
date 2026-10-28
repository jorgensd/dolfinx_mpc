// Copyright (C) 2026 Jørgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#include "log.h"
#include <memory>
#include <spdlog/cfg/argv.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <stdexcept>
#include <string_view>

namespace
{
constexpr std::string_view logger_name = "dolfinx_mpc";
} // namespace

//-----------------------------------------------------------------------------
void dolfinx_mpc::init_logging(int argc, char* argv[])
{
  spdlog::stdout_color_mt(std::string(logger_name));
  logger().set_level(spdlog::level::warn);
  spdlog::cfg::load_argv_levels(argc, argv);
}
//-----------------------------------------------------------------------------
spdlog::logger& dolfinx_mpc::logger()
{
  std::shared_ptr<spdlog::logger> ptr = spdlog::get(std::string(logger_name));
  if (!ptr)
  {
    throw std::runtime_error("DOLFINx-MPC logger not available. "
                             "dolfinx_mpc::init_logging must be called first.");
  }
  return *ptr;
}
//-----------------------------------------------------------------------------
