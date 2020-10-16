# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# This demo demonstrates how to solve a contact problem between
# two stacked cubes.
# The bottom cube is fixed at the bottom surface
# The top cube has a force applied normal to its to surface.
# A slip condition is implemented at the interface of the cube.
# Additional constraints to avoid tangential movement is
# added to the to left corner of the top cube.

import argparse

import dolfinx_mpc
import dolfinx_mpc.utils
import numpy as np
import pygmsh
import ufl
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import dolfinx.fem as fem
import dolfinx.io
import dolfinx.la
import dolfinx.log
from create_and_export_mesh import mesh_2D_dolfin, mesh_2D_gmsh

dolfinx.log.set_log_level(dolfinx.log.LogLevel.ERROR)


comm = MPI.COMM_WORLD


def demo_stacked_cubes(outfile, theta, gmsh=True, triangle=True,
                       compare=False):
    dolfinx_mpc.utils.log_info(
        "Run theta:{0:.2f}, Triangle: {1:b}, Gmsh {2:b}"
        .format(theta, triangle, gmsh))

    if gmsh:
        if triangle:
            celltype = "triangle"
        else:
            celltype = "quad"
        mesh_name = "Grid"
        if MPI.COMM_WORLD.size == 1:
            mesh_2D_gmsh(theta, celltype)
        filename = "meshes/mesh_{0:s}_{1:.2f}_gmsh.xdmf".format(
            celltype, theta)
        facet_file = "meshes/facet_{0:s}_{1:.2f}_rot.xdmf".format(
            celltype, theta)
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD,
                                 filename, "r") as xdmf:
            mesh = xdmf.read_mesh(name=mesh_name)
            mesh.name = "mesh_{0:s}_{1:.2f}_gmsh".format(celltype, theta)
            tdim = mesh.topology.dim
            fdim = tdim - 1
            mesh.topology.create_connectivity(tdim, tdim)
            mesh.topology.create_connectivity(fdim, tdim)

        with dolfinx.io.XDMFFile(MPI.COMM_WORLD,
                                 facet_file, "r") as xdmf:
            mt = xdmf.read_meshtags(mesh, name="Grid")
    else:
        if triangle:
            celltype = "triangle"
        else:
            celltype = "quadrilateral"
        mesh_name = "mesh"
        filename = "meshes/mesh_{0:s}_{1:.2f}.xdmf".format(celltype, theta)

        if MPI.COMM_WORLD.size == 1:
            if triangle:
                mesh_2D_dolfin(celltype, theta)
            else:
                mesh_2D_dolfin(celltype, theta)

        with dolfinx.io.XDMFFile(MPI.COMM_WORLD,
                                 filename, "r") as xdmf:
            mesh = xdmf.read_mesh(name=mesh_name)
            mesh.name = "mesh_{0:s}_{1:.2f}".format(celltype, theta)
            tdim = mesh.topology.dim
            fdim = tdim - 1
            mesh.topology.create_connectivity(tdim, tdim)
            mesh.topology.create_connectivity(fdim, tdim)
            mt = xdmf.read_meshtags(mesh, name="facet_tags")

    # dolfinx.io.VTKFile("results/mesh.pvd").write(mesh)
    # Helper until MeshTags can be read in from xdmf
    V = dolfinx.VectorFunctionSpace(mesh, ("Lagrange", 1))

    r_matrix = pygmsh.helpers.rotation_matrix([0, 0, 1], theta)
    g_vec = np.dot(r_matrix, [0, -1.25e2, 0])
    g = dolfinx.Constant(mesh, g_vec[:2])

    def bottom_corner(x):
        return np.isclose(x, [[0], [0], [0]]).all(axis=0)
    # Fix bottom corner
    u_bc = dolfinx.function.Function(V)
    with u_bc.vector.localForm() as u_local:
        u_local.set(0.0)
    bottom_dofs = fem.locate_dofs_geometrical(V, bottom_corner)
    bc_bottom = fem.DirichletBC(u_bc, bottom_dofs)
    bcs = [bc_bottom]

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
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=mt,
                     subdomain_id=3)
    rhs = ufl.inner(dolfinx.Constant(mesh, (0, 0)), v) * ufl.dx\
        + ufl.inner(g, v) * ds

    def left_corner(x):
        return np.isclose(x.T, np.dot(r_matrix, [0, 2, 0])).all(axis=1)

    mpc = dolfinx_mpc.MultiPointConstraint(V)

    with dolfinx.common.Timer("~Contact: Create contact constraint"):
        nh = dolfinx_mpc.utils.facet_normal_approximation(V, mt, 4)
        mpc_data = dolfinx_mpc.cpp.mpc.create_contact_condition(
            V._cpp_object, mt, 4, 9, nh._cpp_object)
        mpc.add_constraint_from_mpc_data(V, mpc_data)

    with dolfinx.common.Timer(
            "~Contact: Add non-slip condition at bottom interface"):
        bottom_normal = dolfinx_mpc.utils.facet_normal_approximation(V, mt, 5)
        mpc.create_slip_constraint(None, bottom_normal, None, (mt, 5), bcs)

    with dolfinx.common.Timer(
            "~Contact: Add tangential constraint at one point"):
        vertex = dolfinx.mesh.locate_entities_boundary(
            mesh, 0, left_corner)

        tangent = dolfinx_mpc.utils.facet_normal_approximation(
            V, mt, 3, tangent=True)
        mtv = dolfinx.MeshTags(mesh, 0, vertex,
                               np.full(len(vertex), 6, dtype=np.int32))
        mpc.create_slip_constraint(None, tangent, None, (mtv, 6), bcs)

    mpc.finalize()

    with dolfinx.common.Timer("~Contact: Assemble LHS and RHS"):
        A = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs)
        b = dolfinx_mpc.assemble_vector(rhs, mpc)

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

    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setFromOptions()
    solver.setOperators(A)
    uh = b.copy()
    uh.set(0)
    solver.solve(b, uh)
    uh.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                   mode=PETSc.ScatterMode.FORWARD)
    mpc.backsubstitution(uh)
    it = solver.getIterationNumber()
    if MPI.COMM_WORLD.rank == 0:
        print("Number of iterations: {0:d}".format(it))

    unorm = uh.norm()
    if MPI.COMM_WORLD.rank == 0:
        print(unorm)

    # Write solution to file
    ext = "gmsh" if gmsh else ""
    u_h = dolfinx.Function(mpc.function_space())
    u_h.vector.setArray(uh.array)
    u_h.name = "u_mpc_{0:s}_{1:.2f}_{2:s}".format(celltype, theta, ext)

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
    with dolfinx.common.Timer("~MPC: Reference problem"):
        # Generate reference matrices and unconstrained solution
        A_org = fem.assemble_matrix(a, bcs)

        A_org.assemble()
        L_org = fem.assemble_vector(rhs)
        fem.apply_lifting(L_org, [a], [bcs])
        L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                          mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(L_org, bcs)

        # Create global transformation matrix
        K = dolfinx_mpc.utils.create_transformation_matrix(V, mpc)
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

    with dolfinx.common.Timer("~MPC: Compare"):
        # Compare LHS, RHS and solution with reference values
        A_np = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A)
        b_np = dolfinx_mpc.utils.PETScVector_to_global_numpy(b)
        dolfinx_mpc.utils.compare_matrices(reduced_A, A_np, mpc)
        dolfinx_mpc.utils.compare_vectors(reduced_L, b_np, mpc)
        assert np.allclose(uh.array, uh_numpy[uh.owner_range[0]:
                                              uh.owner_range[1]], atol=1e-7)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    comp = parser.add_mutually_exclusive_group(required=False)
    comp.add_argument('--compare', dest='compare', action='store_true',
                      help="Compare with global solution", default=False)
    time = parser.add_mutually_exclusive_group(required=False)
    time.add_argument('--timing', dest='timing', action='store_true',
                      help="List timings", default=False)

    compare = parser.parse_args().compare
    timing = parser.parse_args().timing

    outfile = dolfinx.io.XDMFFile(MPI.COMM_WORLD,
                                  "results/demo_contact_2D.xdmf", "w")
    # Built in meshes aligned with coordinate system
    demo_stacked_cubes(outfile, theta=0, gmsh=False,
                       triangle=True, compare=compare)
    demo_stacked_cubes(outfile, theta=0, gmsh=False,
                       triangle=False, compare=compare)
    # Built in meshes non-aligned
    demo_stacked_cubes(outfile, theta=np.pi / 3, gmsh=False,
                       triangle=True, compare=compare)
    demo_stacked_cubes(outfile, theta=np.pi / 3, gmsh=False,
                       triangle=False, compare=compare)
    # Gmsh aligned
    demo_stacked_cubes(outfile, theta=0, gmsh=True,
                       triangle=False, compare=compare)
    demo_stacked_cubes(outfile, theta=0, gmsh=True,
                       triangle=True, compare=compare)
    # Gmsh non-aligned
    demo_stacked_cubes(outfile, theta=np.pi / 5, gmsh=True,
                       triangle=False, compare=compare)
    demo_stacked_cubes(outfile, theta=np.pi / 5, gmsh=True,
                       triangle=True, compare=compare)

    outfile.close()
    if timing:
        dolfinx.common.list_timings(
            MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])
