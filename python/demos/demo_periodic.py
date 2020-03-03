# This demo program solves Poisson's equation
#
#     - div grad u(x, y) = f(x, y)
#
# on the unit square with homogeneous Dirichlet boundary conditions
# at y = 0, 1 and periodic boundary conditions at x = 0, 1.
#
# Original implementation in DOLFIN by Kristian B. Oelgaard and Anders Logg
# This implementation can be found at:
# https://bitbucket.org/fenics-project/dolfin/src/master/python/demo/documented/periodic/demo_periodic.py
#
# Copyright (C) Jørgen S. Dokken 2020.
#
# This file is part of DOLFINX_MPCX.
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


import dolfinx
import dolfinx.io
import dolfinx_mpc
import ufl
import numpy as np
from petsc4py import PETSc

# Create mesh and finite element
N = 50
mesh = dolfinx.UnitSquareMesh(dolfinx.MPI.comm_world, N, N)
V = dolfinx.FunctionSpace(mesh, ("CG", 1))

# Create Dirichlet boundary condition
u_bc = dolfinx.function.Function(V)
with u_bc.vector.localForm() as u_local:
    u_local.set(0.0)


def DirichletBoundary(x):
    return np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 1))


facets = dolfinx.mesh.compute_marked_boundary_entities(mesh, 1,
                                                       DirichletBoundary)
topological_dofs = dolfinx.fem.locate_dofs_topological(V, 1, facets)
bc = dolfinx.fem.DirichletBC(u_bc, topological_dofs)
bcs = [bc]


# Create MPC (map all left dofs to right
dof_at = dolfinx_mpc.dof_close_to


def slave_locater(i, N):
    return lambda x: dof_at(x, [1, float(i/N)])


def master_locater(i, N):
    return lambda x: dof_at(x, [0, float(i/N)])


s_m_c = {}
for i in range(1, N):
    s_m_c[slave_locater(i, N)] = {master_locater(i, N): 1}

(slaves, masters,
 coeffs, offsets) = dolfinx_mpc.slave_master_structure(V, s_m_c)

# Define variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.dot(ufl.grad(u), ufl.grad(v))*ufl.dx

x = ufl.SpatialCoordinate(mesh)
dx = x[0] - 0.9
dy = x[1] - 0.5
f = x[0]*ufl.sin(5.0*ufl.pi*x[1]) \
            + 1.0*ufl.exp(-(dx*dx + dy*dy)/0.02)

lhs = f*v*ufl.dx

# Compute solution
u = dolfinx.Function(V)
u.name = "uh"
mpc = dolfinx_mpc.cpp.mpc.MultiPointConstraint(V._cpp_object, slaves,
                                               masters, coeffs, offsets)
# Setup MPC system
A = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs)
b = dolfinx_mpc.assemble_vector(lhs, mpc)

# Apply boundary conditions
dolfinx.fem.apply_lifting(b, [a], [bcs])
b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
              mode=PETSc.ScatterMode.REVERSE)
dolfinx.fem.set_bc(b, bcs)

# Solve Linear problem
solver = PETSc.KSP().create(dolfinx.MPI.comm_world)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)
solver.setOperators(A)
uh = b.copy()
uh.set(0)
solver.solve(b, uh)
uh.ghostUpdate(addv=PETSc.InsertMode.INSERT,
               mode=PETSc.ScatterMode.FORWARD)

# Back substitute to slave dofs
dolfinx_mpc.backsubstitution(mpc, uh, V.dofmap)

# Create functionspace and function for mpc vector
Vmpc_cpp = dolfinx.cpp.function.FunctionSpace(mesh, V.element,
                                              mpc.mpc_dofmap())
Vmpc = dolfinx.FunctionSpace(None, V.ufl_element(), Vmpc_cpp)

# Write solution to file
u_h = dolfinx.Function(Vmpc)
u_h.vector.setArray(uh.array)
u_h.name = "u_mpc"
dolfinx.io.XDMFFile(dolfinx.MPI.comm_world, "actual_periodic.xdmf").write(u_h)
