// Copyright (C) 2026 Jørgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#include <cstddef>
#include <dolfinx_mpc/log.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace dolfinx_mpc_wrappers
{
void log(nb::module_& m)
{
  m.def(
      "init_logging",
      [](std::vector<std::string> args)
      {
        std::vector<char*> argv(args.size() + 1, nullptr);
        for (std::size_t i = 0; i < args.size(); ++i)
          argv[i] = args[i].data();
        dolfinx_mpc::init_logging(args.size(), argv.data());
      },
      nb::arg("args"));

  // spdlog::level::level_enum is bound by dolfinx.cpp.log.LogLevel
  m.def(
      "set_log_level", [](spdlog::level::level_enum level)
      { dolfinx_mpc::logger().set_level(level); }, nb::arg("level"));
  m.def("get_log_level", []() { return dolfinx_mpc::logger().level(); });
}
} // namespace dolfinx_mpc_wrappers
