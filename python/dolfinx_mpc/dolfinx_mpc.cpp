// Copyright (C) 2020 Jørgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <iostream>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace dolfinx_mpc_wrappers
{
void mpc(py::module& m);
} // namespace dolfinx_mpc_wrappers

PYBIND11_MODULE(cpp, m)
{
  // Create module for C++ wrappers
  m.doc() = "DOLFINX MultiPointConstraint Python interface";
  m.attr("__version__") = DOLFINX_MPC_VERSION;

  // Create mpc submodule [mpc]
  py::module mpc = m.def_submodule("mpc", "General module");
  dolfinx_mpc_wrappers::mpc(mpc);
}
