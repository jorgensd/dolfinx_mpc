# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np

from petsc4py import PETSc
from mpi4py import MPI
from contextlib import ExitStack
import dolfinx
import dolfinx.io
import dolfinx.la
import dolfinx.log
import dolfinx.common
import dolfinx.mesh
import dolfinx_mpc
import dolfinx_mpc.utils
import ufl
import petsc4py
petsc4py.init('-log_view')
opts = PETSc.Options()
opts["ksp_type"] = "cg"
opts["ksp_rtol"] = 1.0e-10
opts["pc_type"] = "gamg"
opts["pc_gamg_type"] = "agg"
opts["pc_gamg_sym_graph"] = True
opts["mg_levels_ksp_type"] = "chebyshev"
opts["mg_levels_pc_type"] = "jacobi"
opts["mg_levels_esteig_ksp_type"] = "cg"
opts["matptap_via"] = "scalable"

import resource

def build_elastic_nullspace(V):
    """Function to build nullspace for 2D/3D elasticity"""

    # Get geometric dim
    gdim = V.mesh.geometry.dim
    assert gdim == 2 or gdim == 3

    # Set dimension of nullspace
    dim = 3 if gdim == 2 else 6

    # Create list of vectors for null space
    nullspace_basis = [dolfinx.cpp.la.create_vector(
        V.dofmap.index_map) for i in range(dim)]

    with ExitStack() as stack:
        vec_local = [stack.enter_context(x.localForm())
                     for x in nullspace_basis]
        basis = [np.asarray(x) for x in vec_local]

        # Build translational null space basis
        for i in range(gdim):
            dofs = V.sub(i).dofmap.list
            basis[i][dofs.array()] = 1.0

        # Build rotational null space basis
        if gdim == 2:
            V.sub(0).set_x(basis[2], -1.0, 1)
            V.sub(1).set_x(basis[2], 1.0, 0)
        elif gdim == 3:
            V.sub(0).set_x(basis[3], -1.0, 1)
            V.sub(1).set_x(basis[3], 1.0, 0)

            V.sub(0).set_x(basis[4], 1.0, 2)
            V.sub(2).set_x(basis[4], -1.0, 0)

            V.sub(2).set_x(basis[5], 1.0, 1)
            V.sub(1).set_x(basis[5], -1.0, 2)

    basis = dolfinx.la.VectorSpaceBasis(nullspace_basis)
    basis.orthonormalize()

    _x = [basis[i] for i in range(6)]
    nsp = PETSc.NullSpace().create(vectors=_x)
    return nsp


def demo_elasticity(r_lvl=1):
    mesh = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, 4, 4, 4)
    for i in range(r_lvl):
        # dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
        mesh = dolfinx.mesh.refine(mesh, redistribute=True)
        # dolfinx.log.set_log_level(dolfinx.log.LogLevel.ERROR)

    fdim = mesh.topology.dim - 1
    V = dolfinx.VectorFunctionSpace(mesh, ("Lagrange", 1))

    # Generate Dirichlet BC on lower boundary (Fixed)
    u_bc = dolfinx.function.Function(V)
    with u_bc.vector.localForm() as u_local:
        u_local.set(0.0)

    def boundaries(x):
        return np.isclose(x[0], np.finfo(float).eps)
    facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim,
                                                   boundaries)
    topological_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, facets)
    bc = dolfinx.fem.DirichletBC(u_bc, topological_dofs)
    bcs = [bc]

    # Create traction meshtag
    def traction_boundary(x):
        return np.isclose(x[0], 1)
    t_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim,
                                                     traction_boundary)
    facet_values = np.ones(len(t_facets), dtype=np.int32)
    mt = dolfinx.MeshTags(mesh, fdim, t_facets, facet_values)

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Elasticity parameters
    E = 1.0e4
    nu = 0.1
    mu = dolfinx.Constant(mesh, E / (2.0 * (1.0 + nu)))
    lmbda = dolfinx.Constant(mesh, E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))
    g = dolfinx.Constant(mesh, (0, 0, -1e2))

    # Stress computation
    def sigma(v):
        return (2.0 * mu * ufl.sym(ufl.grad(v)) +
                lmbda * ufl.tr(ufl.sym(ufl.grad(v))) * ufl.Identity(len(v)))

    # x = ufl.SpatialCoordinate(mesh)
    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(sigma(u), ufl.grad(v)) * ufl.dx
    lhs = ufl.inner(g, v)*ufl.ds(domain=mesh,
                                 subdomain_data=mt, subdomain_id=1)
    if MPI.COMM_WORLD.rank == 0:
        print("Problem size {0:d} ".format(V.dim))
    # Create MPC
    dof_at = dolfinx_mpc.dof_close_to
    s_m_c = {lambda x: dof_at(x, [1, 0, 0]): {
        lambda x: dof_at(x, [1, 0, 1]): 0.5}}  # , lambda x: dof_at(x, [1, 1, 0]): {
    # lambda x: dof_at(x, [1, 1, 1]): 0.5}}
    (slaves, masters,
     coeffs, offsets) = dolfinx_mpc.slave_master_structure(V, s_m_c,
                                                           2, 2)

    mpc = dolfinx_mpc.cpp.mpc.MultiPointConstraint(V._cpp_object, slaves,
                                                   masters, coeffs, offsets)
    # Setup MPC system
    A = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs)
    b = dolfinx_mpc.assemble_vector(lhs, mpc)

    # Create functionspace and function for mpc vector
    Vmpc_cpp = dolfinx.cpp.function.FunctionSpace(mesh, V.element,
                                                  mpc.mpc_dofmap())
    Vmpc = dolfinx.FunctionSpace(None, V.ufl_element(), Vmpc_cpp)
    null_space = build_elastic_nullspace(Vmpc)
    A.setNearNullSpace(null_space)

    # Apply boundary conditions
    dolfinx.fem.apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                  mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b, bcs)

    # def monitor(ksp, its, rnorm, r_lvl=-1):
    #     if MPI.COMM_WORLD.rank == 0:
    #         # print("{}: Iteration: {}, rel. residual: {}"
    #         #      .format(r_lvl, its, rnorm))
    #         pass

    # def pmonitor(ksp, its, rnorm):
    #     return monitor(ksp, its, rnorm, r_lvl=r_lvl)

    # Solve Linear problem

    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setFromOptions()
    # solver.setMonitor(pmonitor)
    solver.setOperators(A)
    uh = b.copy()
    uh.set(0)
    solver.solve(b, uh)
    mem = sum(MPI.COMM_WORLD.allgather(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    it = solver.getIterationNumber()
    if MPI.COMM_WORLD.rank == 0:
        print("Refinement level {0:d}, Iterations {1:d}".format(r_lvl, it))
        print("Max usage {0:d} (kb)".format(mem))
    uh.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                   mode=PETSc.ScatterMode.FORWARD)

    # Back substitute to slave dofs
    dolfinx_mpc.backsubstitution(mpc, uh, V.dofmap)

    # # Write solution to file
    # u_h = dolfinx.Function(Vmpc)
    # u_h.vector.setArray(uh.array)
    # u_h.name = "u_mpc"
    # outfile = dolfinx.io.XDMFFile(MPI.COMM_WORLD,
    #                               "results/bench_elasticity.xdmf", "w")
    # outfile.write_mesh(mesh)
    # outfile.write_function(u_h)

if __name__ == "__main__":
    for i in range(7):
        if MPI.COMM_WORLD.rank == 0:
            dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
            dolfinx.log.log(dolfinx.log.LogLevel.INFO,
                            "Run {0:1d} in progress".format(i))
            dolfinx.log.set_log_level(dolfinx.log.LogLevel.ERROR)
        demo_elasticity(i)
        # dolfinx.common.list_timings(
        #     MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])
