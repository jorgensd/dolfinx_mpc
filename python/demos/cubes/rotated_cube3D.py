# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Multi point constraint problem for linear elasticity with slip conditions
# between two cubes.
from contextlib import ExitStack

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
from create_and_export_mesh import mesh_3D_dolfin, mesh_3D_rot
from helpers import find_master_slave_relationship


comm = MPI.COMM_WORLD


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

    _x = [basis[i] for i in range(dim)]
    nsp = PETSc.NullSpace().create(vectors=_x)
    return nsp


def demo_stacked_cubes(outfile, theta, dolfin_mesh=False,
                       ct=dolfinx.cpp.mesh.CellType.tetrahedron):
    if ct == dolfinx.cpp.mesh.CellType.hexahedron:
        ext = "hexahedron"
    else:
        ext = "tetrahedron"
    if comm.rank == 0:
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
        dolfinx.log.log(dolfinx.log.LogLevel.INFO,
                        "Run theta:{0:.2f}, Cell: {1:s}, Gmsh {2:b}"
                        .format(theta, ext, dolfin_mesh))
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.ERROR)
    # Create rotated mesh
    if MPI.COMM_WORLD.size == 1:
        if dolfin_mesh:
            mesh_3D_dolfin(theta, ct, ext)
        else:
            if ext == "tetrahedron":
                mesh_3D_rot(theta, "tetra")
            else:
                mesh_3D_rot(theta, ext)
    # Read in mesh
    if dolfin_mesh:
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD,
                                 "meshes/mesh_{0:s}.xdmf".format(ext),
                                 "r") as xdmf:
            mesh = xdmf.read_mesh(name="mesh")
            mesh.name = "mesh_{0:s}_{1:.2f}".format(ext, theta)
            tdim = mesh.topology.dim
            fdim = tdim - 1
            mesh.topology.create_connectivity(tdim, tdim)
            mesh.topology.create_connectivity(fdim, tdim)
            ct = xdmf.read_meshtags(mesh, "mesh_tags")
            mt = xdmf.read_meshtags(mesh, "facet_tags")
    else:
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD,
                                 "meshes/mesh3D_rot.xdmf", "r") as xdmf:
            mesh = xdmf.read_mesh(name="Grid")
            mesh.name = "mesh_{0:s}_{1:.2f}".format(ext, theta)
            tdim = mesh.topology.dim
            fdim = tdim - 1
            mesh.topology.create_connectivity(tdim, tdim)
            mesh.topology.create_connectivity(fdim, tdim)

            ct = xdmf.read_meshtags(mesh, "Grid")

        with dolfinx.io.XDMFFile(MPI.COMM_WORLD,
                                 "meshes/facet3D_rot.xdmf", "r") as xdmf:
            mt = xdmf.read_meshtags(mesh, "Grid")
    top_cube_marker = 2

    # Create functionspaces
    V = dolfinx.VectorFunctionSpace(mesh, ("Lagrange", 1))

    # Helper for orienting traction
    r_matrix = pygmsh.helpers.rotation_matrix(
        [0, 1/np.sqrt(2), 1/np.sqrt(2)], -theta)
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
        return (2.0 * mu * ufl.sym(ufl.grad(v)) +
                lmbda * ufl.tr(ufl.sym(ufl.grad(v))) * ufl.Identity(len(v)))

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(sigma(u), ufl.grad(v)) * ufl.dx
    # NOTE: Traction deactivated until we have a way of fixing nullspace
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=mt,
                     subdomain_id=3)
    lhs = ufl.inner(dolfinx.Constant(mesh, (0, 0, 0)), v)*ufl.dx\
        + ufl.inner(g, v)*ds

    print(V.dim)
    # Find slave master relationship and initialize MPC class
    (slaves, masters, coeffs, offsets,
     owner_ranks) = find_master_slave_relationship(
        V, (mt, 4, 9), (ct, top_cube_marker))

    mpc = dolfinx_mpc.cpp.mpc.MultiPointConstraint(V._cpp_object, slaves,
                                                   masters, coeffs, offsets,
                                                   owner_ranks)
    # Setup MPC system
    A = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs)
    b = dolfinx_mpc.assemble_vector(lhs, mpc)
    return

    # Apply boundary conditions
    fem.apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                  mode=PETSc.ScatterMode.REVERSE)
    fem.set_bc(b, bcs)

    # Solve Linear problem
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
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
    u_h.name = "u_{0:s}_{1:.2f}".format(ext, theta)
    outfile.write_mesh(mesh)
    outfile.write_function(u_h, 0.0,
                           "Xdmf/Domain/"
                           + "Grid[@Name='{0:s}'][1]"
                           .format(mesh.name))
    # Transfer data from the MPC problem to numpy arrays for comparison
    # A_mpc_np = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A)
    # mpc_vec_np = dolfinx_mpc.utils.PETScVector_to_global_numpy(b)

    # # Solve the MPC problem using a global transformation matrix
    # # and numpy solvers to get reference values
    # print("Solving reference problem with global matrix (using numpy)")
    # # Generate reference matrices and unconstrained solution
    # A_org = fem.assemble_matrix(a, bcs)

    # A_org.assemble()
    # L_org = fem.assemble_vector(lhs)
    # fem.apply_lifting(L_org, [a], [bcs])
    # L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
    #                   mode=PETSc.ScatterMode.REVERSE)
    # fem.set_bc(L_org, bcs)

    # # Create global transformation matrix
    # K = dolfinx_mpc.utils.create_transformation_matrix(V.dim, slaves,
    #                                                    masters, coeffs,
    #                                                    offsets)
    # # Create reduced A
    # A_global = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A_org)

    # reduced_A = np.matmul(np.matmul(K.T, A_global), K)
    # # Created reduced L
    # vec = dolfinx_mpc.utils.PETScVector_to_global_numpy(L_org)
    # reduced_L = np.dot(K.T, vec)
    # # Solve linear system
    # d = np.linalg.solve(reduced_A, reduced_L)
    # # Back substitution to full solution vector
    # uh_numpy = np.dot(K, d)
    # # Compare LHS, RHS and solution with reference values
    # dolfinx_mpc.utils.compare_matrices(reduced_A, A_mpc_np, slaves)
    # dolfinx_mpc.utils.compare_vectors(reduced_L, mpc_vec_np, slaves)
    # assert np.allclose(uh.array, uh_numpy[uh.owner_range[0]:
    #                                       uh.owner_range[1]])
    unorm = uh.norm()
    if comm.rank == 0:
        print(unorm)


if __name__ == "__main__":
    outfile = dolfinx.io.XDMFFile(MPI.COMM_WORLD,
                                  "results/rotated_cube3D.xdmf", "w")
    cts = [dolfinx.cpp.mesh.CellType.hexahedron,
           dolfinx.cpp.mesh.CellType.tetrahedron]
    for ct in cts:
        demo_stacked_cubes(outfile, theta=0, dolfin_mesh=True, ct=ct)
        demo_stacked_cubes(outfile, theta=np.pi/3, dolfin_mesh=True, ct=ct)
    # NOTE: Unstructured hex meshes not working nicely as interfaces
    # do not match.
    # demo_stacked_cubes(
    #     outfile, theta=np.pi/5, ct=dolfinx.cpp.mesh.CellType.tetrahedron)
    if comm.rank == 0:
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
        dolfinx.log.log(dolfinx.log.LogLevel.INFO,
                        "Finished")
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.ERROR)
    dolfinx.common.list_timings(
        MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])
    outfile.close()
