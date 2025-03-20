// Copyright (C) 2020 Jørgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace dolfinx_mpc_wrappers
{
void mpc(nb::module_& m);
} // namespace dolfinx_mpc_wrappers

NB_MODULE(cpp, m)
{
  // Create module for C++ wrappers
  m.doc() = "DOLFINX MultiPointConstraint Python interface";
  m.attr("__version__") = DOLFINX_MPC_VERSION;

#ifdef NDEBUG
  nanobind::set_leak_warnings(false);
#endif

  // Create mpc submodule [mpc]
  nb::module_ mpc = m.def_submodule("mpc", "General module");
  dolfinx_mpc_wrappers::mpc(mpc);
}
