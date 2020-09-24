# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Multi point constraint problem for linear elasticity with slip conditions
# between two cubes.


import pdb
from IPython import embed
import dolfinx.io
import dolfinx.common
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx_mpc

import dolfinx
import dolfinx.io
import dolfinx.la
from create_and_export_mesh import mesh_3D_rot

comm = MPI.COMM_WORLD
ext = "hexahedron"
theta = 0  # np.pi/3
if comm.size == 1:
    mesh_3D_rot(theta, ext)

with dolfinx.io.XDMFFile(comm,
                         "meshes/mesh_{0:s}_{1:.2f}_gmsh.xdmf"
                         .format(ext, theta), "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")
    mesh.name = "mesh_{0:s}_{1:.2f}".format(ext, theta)
    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_connectivity(tdim, tdim)
    mesh.topology.create_connectivity(fdim, tdim)

    ct = xdmf.read_meshtags(mesh, "Grid")

with dolfinx.io.XDMFFile(comm,
                         "meshes/facet_{0:s}_{1:.2f}_gmsh.xdmf"
                         .format(ext, theta), "r") as xdmf:
    mt = xdmf.read_meshtags(mesh, "Grid")
V = dolfinx.VectorFunctionSpace(mesh, ("CG", 1))
nh = dolfinx_mpc.utils.facet_normal_approximation(V, mt, 4)
mpc_data = dolfinx_mpc.cpp.mpc.create_contact_condition(
    V._cpp_object, mt, 4, 9, nh._cpp_object)
mpc = dolfinx_mpc.MultiPointConstraint(V)
mpc.add_constraint_from_mpc_data(V, mpc_data)
mpc.finalize()
# mpc.create_contact_constraint(mt, 4, 9)

#dolfinx.common.list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])
print(comm.rank, "Slaves:", sum(mt.values == 4)
      > 0, "Masters:", sum(mt.values == 9) > 0)
# cell_map = mesh.topology.index_map(tdim)
# num_cells_local = cell_map.size_local
# indices = np.arange(num_cells_local)
# values = MPI.COMM_WORLD.rank*np.ones(num_cells_local, np.int32)
# ct = dolfinx.MeshTags(mesh, mesh.topology.dim, indices, values)
# ct.name = "cells"
# with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "cf.xdmf", "w") as xdmf:
#     xdmf.write_mesh(mesh)
#     xdmf.write_meshtags(ct)
