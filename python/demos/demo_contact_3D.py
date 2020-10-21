# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Multi point constraint problem for linear elasticity with slip conditions
# between two cubes.

import argparse

import dolfinx.fem as fem
import dolfinx.io
import dolfinx_mpc
import dolfinx_mpc.utils
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from create_and_export_mesh import mesh_3D_dolfin, gmsh_3D_stacked

comm = MPI.COMM_WORLD


def demo_stacked_cubes(outfile, theta, gmsh=False,
                       ct=dolfinx.cpp.mesh.CellType.tetrahedron, compare=True, res=0.1):
    if ct == dolfinx.cpp.mesh.CellType.hexahedron:
        celltype = "hexahedron"
    else:
        celltype = "tetrahedron"
    dolfinx_mpc.utils.log_info(
        "Run theta:{0:.2f}, Cell: {1:s}, GMSH {2:b}"
        .format(theta, celltype, gmsh))
    # Create rotated mesh
    if comm.size == 1 and not gmsh:
        mesh_3D_dolfin(theta, ct, celltype)

    # Read in mesh
    if gmsh:
        mesh, mt = gmsh_3D_stacked(celltype, theta, res)
        mesh.name = "mesh_{0:s}_{1:.2f}_gmsh".format(celltype, theta)
        tdim = mesh.topology.dim
        fdim = tdim - 1
        mesh.topology.create_connectivity(tdim, tdim)
        mesh.topology.create_connectivity(fdim, tdim)
    else:
        with dolfinx.io.XDMFFile(comm,
                                 "meshes/mesh_{0:s}_{1:.2f}.xdmf".format(
                                     celltype, theta),
                                 "r") as xdmf:
            mesh = xdmf.read_mesh(name="mesh")
            mesh.name = "mesh_{0:s}_{1:.2f}".format(celltype, theta)
            tdim = mesh.topology.dim
            fdim = tdim - 1
            mesh.topology.create_connectivity(tdim, tdim)
            mesh.topology.create_connectivity(fdim, tdim)
            mt = xdmf.read_meshtags(mesh, "facet_tags")

    # Create functionspaces
    V = dolfinx.VectorFunctionSpace(mesh, ("Lagrange", 1))

    # Helper for orienting traction
    r_matrix = dolfinx_mpc.utils.rotation_matrix(
        [1 / np.sqrt(2), 1 / np.sqrt(2), 0], -theta)

    g_vec = np.dot(r_matrix, [0, 0, -4.25e-1])
    g = dolfinx.Constant(mesh, g_vec)

    # Define boundary conditions
    # Bottom boundary is fixed in all directions
    u_bc = dolfinx.function.Function(V)
    with u_bc.vector.localForm() as u_local:
        u_local.set(0.0)

    bottom_facets = mt.indices[np.flatnonzero(mt.values == 5)]
    bottom_dofs = fem.locate_dofs_topological(V, fdim, bottom_facets)
    bc_bottom = fem.DirichletBC(u_bc, bottom_dofs)

    # Top boundary has a given deformation normal to the interface
    g_vec = np.dot(r_matrix, [0, 0, -4.25e-1])
    g = dolfinx.Constant(mesh, g_vec)

    def top_v(x):
        values = np.empty((3, x.shape[1]))
        values[0] = g_vec[0]
        values[1] = g_vec[1]
        values[2] = g_vec[2]
        return values
    u_top = dolfinx.function.Function(V)
    u_top.interpolate(top_v)

    top_facets = mt.indices[np.flatnonzero(mt.values == 3)]
    top_dofs = fem.locate_dofs_topological(V, fdim, top_facets)
    bc_top = fem.DirichletBC(u_top, top_dofs)

    bcs = [bc_bottom, bc_top]

    # Elasticity parameters
    E = 1.0e3
    nu = 0
    mu = dolfinx.Constant(mesh, E / (2.0 * (1.0 + nu)))
    lmbda = dolfinx.Constant(mesh, E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))

    # Stress computation
    def sigma(v):
        return (2.0 * mu * ufl.sym(ufl.grad(v))
                + lmbda * ufl.tr(ufl.sym(ufl.grad(v))) * ufl.Identity(len(v)))

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(sigma(u), ufl.grad(v)) * ufl.dx
    # NOTE: Traction deactivated until we have a way of fixing nullspace
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=mt,
                     subdomain_id=3)
    rhs = ufl.inner(dolfinx.Constant(mesh, (0, 0, 0)), v) * ufl.dx\
        + ufl.inner(g, v) * ds

    mpc = dolfinx_mpc.MultiPointConstraint(V)
    nh = dolfinx_mpc.utils.facet_normal_approximation(V, mt, 4)
    with dolfinx.common.Timer("~Contact: Create contact constraint"):
        mpc_data = dolfinx_mpc.cpp.mpc.create_contact_condition(
            V._cpp_object, mt, 4, 9, nh._cpp_object)
        mpc.add_constraint_from_mpc_data(V, mpc_data)
    with dolfinx.common.Timer("~Contact: Finalize MPC"):
        mpc.finalize()

    with dolfinx.common.Timer("~Contact: Assembly ({0:d})".format(V.dim)):
        A = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs)
        b = dolfinx_mpc.assemble_vector(rhs, mpc)
    with dolfinx.common.Timer("~Contact: Assembly (cached) ({0:d})".format(V.dim)):
        A = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs, A=A)

    fem.apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                  mode=PETSc.ScatterMode.REVERSE)
    fem.set_bc(b, bcs)

    # Solve Linear problem
    opts = PETSc.Options()
    opts["ksp_rtol"] = 1.0e-8
    opts["pc_type"] = "gamg"
    opts["pc_gamg_type"] = "agg"
    opts["pc_gamg_coarse_eq_limit"] = 1000
    opts["pc_gamg_sym_graph"] = True
    opts["mg_levels_ksp_type"] = "chebyshev"
    opts["mg_levels_pc_type"] = "jacobi"
    opts["mg_levels_esteig_ksp_type"] = "cg"
    opts["matptap_via"] = "scalable"
    opts["pc_gamg_square_graph"] = 2
    opts["pc_gamg_threshold"] = 0.02
    # opts["help"] = None # List all available options
    # opts["ksp_view"] = None # List progress of solver

    # Create functionspace and build near nullspace
    null_space = dolfinx_mpc.utils.rigid_motions_nullspace(
        mpc.function_space())
    A.setNearNullSpace(null_space)
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setFromOptions()
    uh = b.copy()
    uh.set(0)
    with dolfinx.common.Timer("~Contact: Solve old"):
        solver.solve(b, uh)
        uh.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                       mode=PETSc.ScatterMode.FORWARD)
        mpc.backsubstitution(uh)

    it = solver.getIterationNumber()
    if MPI.COMM_WORLD.rank == 0:
        print("Number of iterations: {0:d}".format(it))

    unorm = uh.norm()
    if comm.rank == 0:
        print("Norm of u {0:.5e}".format(unorm))

    # Write solution to file
    u_h = dolfinx.Function(mpc.function_space())
    u_h.vector.setArray(uh.array)
    u_h.name = "u_{0:s}_{1:.2f}".format(celltype, theta)
    outfile.write_mesh(mesh)
    outfile.write_function(u_h, 0.0,
                           "Xdmf/Domain/"
                           + "Grid[@Name='{0:s}'][1]"
                           .format(mesh.name))

    # Solve the MPC problem using a global transformation matrix
    # and numpy solvers to get reference values
    if not compare:
        return

    dolfinx_mpc.utils.log_info(
        "Solving reference problem with global matrix (using numpy)")
    with dolfinx.common.Timer("~Contact: Reference problem"):
        A_org = fem.assemble_matrix(a, bcs)
        A_org.assemble()
        L_org = fem.assemble_vector(rhs)
        fem.apply_lifting(L_org, [a], [bcs])
        L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                          mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(L_org, bcs)
        K = dolfinx_mpc.utils.create_transformation_matrix(V, mpc)
        A_global = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A_org)

        reduced_A = np.matmul(np.matmul(K.T, A_global), K)
        vec = dolfinx_mpc.utils.PETScVector_to_global_numpy(L_org)
        reduced_L = np.dot(K.T, vec)
        d = np.linalg.solve(reduced_A, reduced_L)
        uh_numpy = np.dot(K, d)

    with dolfinx.common.Timer("~Contact: Compare LHS, RHS and solution"):
        A_np = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A)
        b_np = dolfinx_mpc.utils.PETScVector_to_global_numpy(b)
        dolfinx_mpc.utils.compare_matrices(reduced_A, A_np, mpc)
        dolfinx_mpc.utils.compare_vectors(reduced_L, b_np, mpc)
        assert np.allclose(uh.array, uh_numpy[uh.owner_range[0]:
                                              uh.owner_range[1]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    comp = parser.add_mutually_exclusive_group(required=False)
    comp.add_argument('--compare', dest='compare', action='store_true',
                      help="Compare with global solution", default=False)
    time = parser.add_mutually_exclusive_group(required=False)
    time.add_argument('--timing', dest='timing', action='store_true',
                      help="List timings", default=False)
    parser.add_argument("--res", default=0.1, type=np.float64, dest="res",
                        help="Resolution of Mesh")

    compare = parser.parse_args().compare
    timing = parser.parse_args().timing
    res = parser.parse_args().res

    outfile = dolfinx.io.XDMFFile(comm,
                                  "results/demo_contact_3D.xdmf", "w")
    # cts = [dolfinx.cpp.mesh.CellType.hexahedron,
    #        dolfinx.cpp.mesh.CellType.tetrahedron]
    # for ct in cts:
    #     demo_stacked_cubes(
    #         outfile, theta=0, gmsh=False, ct=ct, compare=compare)
    #     demo_stacked_cubes(
    #         outfile, theta=np.pi / 3, gmsh=False, ct=ct, compare=compare)

    # NOTE: Unstructured solution experience slight unphysical deformation
    demo_stacked_cubes(
        outfile, theta=np.pi / 5, gmsh=True, ct=dolfinx.cpp.mesh.CellType.tetrahedron,
        compare=compare, res=res)
    # demo_stacked_cubes(
    #     outfile, theta=np.pi / 5,gmsh=True, ct=dolfinx.cpp.mesh.CellType.hexahedron,
    #     compare=compare)
    outfile.close()

    dolfinx_mpc.utils.log_info("Simulation finished")
    if timing:
        dolfinx.common.list_timings(
            comm, [dolfinx.common.TimingType.wall])
