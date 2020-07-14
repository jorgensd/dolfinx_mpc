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
import dolfinx_mpc.utils
import ufl
import numpy as np
import time
from mpi4py import MPI
from petsc4py import PETSc


# Create mesh and finite element
N = 12
mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, N, N)
V = dolfinx.FunctionSpace(mesh, ("CG", 1))

# Create Dirichlet boundary condition
u_bc = dolfinx.function.Function(V)
with u_bc.vector.localForm() as u_local:
    u_local.set(0.0)


def DirichletBoundary(x):
    return np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 1))


facets = dolfinx.mesh.locate_entities_boundary(mesh, 1,
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
 coeffs, offsets, owner_ranks) = dolfinx_mpc.slave_master_structure(V, s_m_c)

# Define variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx

x = ufl.SpatialCoordinate(mesh)
dx = x[0] - 0.9
dy = x[1] - 0.5
f = x[0]*ufl.sin(5.0*ufl.pi*x[1]) \
    + 1.0*ufl.exp(-(dx*dx + dy*dy)/0.02)

lhs = ufl.inner(f, v)*ufl.dx

# Compute solution
u = dolfinx.Function(V)
u.name = "uh"
mpc = dolfinx_mpc.cpp.mpc.MultiPointConstraint(V._cpp_object, slaves,
                                               masters, coeffs, offsets,
                                               owner_ranks)
# Setup MPC system

A = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs)
b = dolfinx_mpc.assemble_vector(lhs, mpc)

# Apply boundary conditions
dolfinx.fem.apply_lifting(b, [a], [bcs])
b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
              mode=PETSc.ScatterMode.REVERSE)
dolfinx.fem.set_bc(b, bcs)

# Solve Linear problem
opts = PETSc.Options()

solver = PETSc.KSP().create(MPI.COMM_WORLD)
opts["ksp_type"] = "cg"
opts["ksp_rtol"] = 1.0e-5
opts["pc_type"] = "hypre"
opts['pc_hypre_type'] = 'boomeramg'
opts["pc_hypre_boomeramg_max_iter"] = 1
opts["pc_hypre_boomeramg_cycle_type"] = "v"
# opts["pc_hypre_boomeramg_print_statistics"] = 1

solver.setFromOptions()
solver.setOperators(A)

uh = b.copy()
uh.set(0)
start = time.time()
solver.solve(b, uh)
# solver.view()
end = time.time()
it = solver.getIterationNumber()

print("Solver time: {0:.2e}, Iterations {1:d}".format(end-start, it))

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
outfile = dolfinx.io.XDMFFile(MPI.COMM_WORLD,
                              "results/demo_periodic.xdmf", "w")
outfile.write_mesh(mesh)
outfile.write_function(u_h)


print("----Verification----")
# --------------------VERIFICATION-------------------------
A_org = dolfinx.fem.assemble_matrix(a, bcs)

A_org.assemble()
L_org = dolfinx.fem.assemble_vector(lhs)
dolfinx.fem.apply_lifting(L_org, [a], [bcs])
L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                  mode=PETSc.ScatterMode.REVERSE)
dolfinx.fem.set_bc(L_org, bcs)
solver.setOperators(A_org)
u_ = dolfinx.Function(V)
start = time.time()
solver.solve(L_org, u_.vector)
end = time.time()

it = solver.getIterationNumber()
print("Org solver time: {0:.2e}, Iterations {1:d}".format(end-start, it))
u_.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                      mode=PETSc.ScatterMode.FORWARD)
u_.name = "u_unconstrained"
outfile.write_function(u_)

# Transfer data from the MPC problem to numpy arrays for comparison
A_mpc_np = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A)
mpc_vec_np = dolfinx_mpc.utils.PETScVector_to_global_numpy(b)

# Create global transformation matrix
K = dolfinx_mpc.utils.create_transformation_matrix(V.dim, slaves,
                                                   masters, coeffs,
                                                   offsets)
# Create reduced A
A_global = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A_org)
reduced_A = np.matmul(np.matmul(K.T, A_global), K)


# Created reduced L
vec = dolfinx_mpc.utils.PETScVector_to_global_numpy(L_org)
reduced_L = np.dot(K.T, vec)
# Solve linear system
d = np.linalg.solve(reduced_A, reduced_L)
# Back substitution to full solution vector
uh_numpy = np.dot(K, d)

# compare LHS, RHS and solution with reference values

dolfinx_mpc.utils.compare_vectors(reduced_L, mpc_vec_np, slaves)
dolfinx_mpc.utils.compare_matrices(reduced_A, A_mpc_np, slaves)
assert np.allclose(uh.array, uh_numpy[uh.owner_range[0]:uh.owner_range[1]])
