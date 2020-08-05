# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Multi point constraint problem for linear elasticity with slip conditions
# between two cubes.

import dolfinx_mpc
import dolfinx_mpc.utils
import numpy as np
import pygmsh
import ufl
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import dolfinx.io as io
import dolfinx.fem as fem
import meshio


def generate_hex_box(x0, y0, z0, x1, y1, z1, theta, res, facet_markers,
                     volume_marker=None):
    """
    Generate the box [x0,y0,z0]x[y0,y1,z0], rotated theta degrees around
    the origin over axis [0,1,1] with resolution res,
    where markers are is an array containing markers
    array of markers for [back, bottom, right, left, top, front]
    and an optional volume_marker.
    """
    geom = pygmsh.built_in.Geometry()

    rect = geom.add_rectangle(x0, x1, y0, y1, 0.0, res)

    geom.add_raw_code("Mesh.RecombinationAlgorithm = 2;")
    geom.add_raw_code("Recombine Surface {:};")
    bottom, volume, sides = geom.extrude(rect, translation_axis=[0, 0, 1],
                                         num_layers=int(0.5/res),
                                         recombine=True)

    geom.add_physical(bottom, facet_markers[4])
    if volume_marker is None:
        geom.add_physical(volume, 0)
    else:
        geom.add_physical(volume, volume_marker)
    geom.add_physical(rect.surface, facet_markers[1])
    geom.add_physical(sides[0], facet_markers[3])
    geom.add_physical(sides[1], facet_markers[5])
    geom.add_physical(sides[2], facet_markers[2])
    geom.add_physical(sides[3], facet_markers[0])
    msh = pygmsh.generate_mesh(geom, verbose=False)

    msh.points[:, -1] += z0
    r_matrix = pygmsh.helpers.rotation_matrix(
        [1/np.sqrt(2), 1/np.sqrt(2), 0], -theta)
    msh.points[:, :] = np.dot(r_matrix, msh.points.T).T
    return msh


def merge_msh_meshes(msh0, msh1, celltype, facettype):
    """
    Merge two meshio meshes with the same celltype, and create
    the mesh and corresponding facet mesh with domain markers.
    """
    points0 = msh0.points

    cells0 = np.vstack([cells.data for cells in msh0.cells
                        if cells.type == celltype])
    cell_data0 = np.vstack([msh0.cell_data_dict["gmsh:physical"][key]
                            for key in
                            msh0.cell_data_dict["gmsh:physical"].keys()
                            if key == celltype])
    points1 = msh1.points
    points = np.vstack([points0, points1])
    cells1 = np.vstack([cells.data for cells in msh1.cells
                        if cells.type == celltype])
    cell_data1 = np.vstack([msh1.cell_data_dict["gmsh:physical"][key]
                            for key in
                            msh1.cell_data_dict["gmsh:physical"].keys()
                            if key == celltype])
    cells1 += points0.shape[0]
    cells = np.vstack([cells0, cells1])
    cell_data = np.hstack([cell_data0, cell_data1])
    mesh = meshio.Mesh(points=points,
                       cells=[(celltype, cells)],
                       cell_data={"name_to_read": cell_data})

    # Write line mesh
    facet_cells0 = np.vstack([cells.data for cells in msh0.cells
                              if cells.type == facettype])
    facet_data0 = np.vstack([msh0.cell_data_dict["gmsh:physical"][key]
                             for key in
                             msh0.cell_data_dict["gmsh:physical"].keys()
                             if key == facettype])
    points1 = msh1.points
    points = np.vstack([points0, points1])
    facet_cells1 = np.vstack([cells.data for cells in msh1.cells
                              if cells.type == facettype])
    facet_data1 = np.vstack([msh1.cell_data_dict["gmsh:physical"][key]
                             for key in
                             msh1.cell_data_dict["gmsh:physical"].keys()
                             if key == facettype])
    facet_cells1 += points0.shape[0]
    facet_cells = np.vstack([facet_cells0, facet_cells1])

    facet_data = np.hstack([facet_data0, facet_data1])
    facet_mesh = meshio.Mesh(points=points,
                             cells=[(facettype, facet_cells)],
                             cell_data={"name_to_read": facet_data})
    return mesh, facet_mesh


def mesh_3D_rot(theta):
    res = 0.2

    msh0 = generate_hex_box(0, 0, 0, 1, 1, 1, theta, res,
                            [11, 5, 12, 13, 4, 14])
    msh1 = generate_hex_box(0, 0, 1, 1, 1, 2, theta, 2*res,
                            [11, 9, 12, 13, 3, 14], 2)
    mesh, facet_mesh = merge_msh_meshes(msh0, msh1, "hexahedron", "quad")
    meshio.xdmf.write("mesh_hex_cube{0:.2f}.xdmf".format(theta), mesh)
    meshio.xdmf.write("facet_hex_cube{0:.2f}.xdmf".format(theta), facet_mesh)


def test_cube_contact():
    comm = MPI.COMM_WORLD
    theta = np.pi/5
    if comm.size == 1:
        mesh_3D_rot(theta)
    # Read in mesh
    with io.XDMFFile(comm, "mesh_hex_cube{0:.2f}.xdmf".format(theta),
                     "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        mesh.name = "mesh_hex_{0:.2f}".format(theta)
        tdim = mesh.topology.dim
        fdim = tdim - 1
        mesh.topology.create_connectivity(tdim, tdim)
        mesh.topology.create_connectivity(fdim, tdim)

    with io.XDMFFile(comm, "facet_hex_cube{0:.2f}.xdmf".format(theta),
                     "r") as xdmf:
        mt = xdmf.read_meshtags(mesh, "Grid")

    # Create functionspaces
    V = dolfinx.VectorFunctionSpace(mesh, ("Lagrange", 1))
    io.VTKFile("outmesh.pvd").write(mesh)
    # Helper for orienting traction
    r_matrix = pygmsh.helpers.rotation_matrix(
        [1/np.sqrt(2), 1/np.sqrt(2), 0], -theta)
    g_vec = np.dot(r_matrix, [0, 0, -4e-1])

    # Bottom boundary is fixed in all directions
    u_bc = dolfinx.function.Function(V)
    with u_bc.vector.localForm() as u_local:
        u_local.set(0.0)

    bottom_facets = mt.indices[np.flatnonzero(mt.values == 5)]
    bottom_dofs = fem.locate_dofs_topological(V, fdim, bottom_facets)
    bc_bottom = fem.DirichletBC(u_bc, bottom_dofs)

    # Top boundary has a given deformation normal to the interface
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
    rhs = ufl.inner(dolfinx.Constant(mesh, (0, 0, 0)), v)*ufl.dx

    # Create LU solver
    solver = PETSc.KSP().create(comm)
    solver.setType("preonly")
    solver.setTolerances(rtol=1.0e-14)
    solver.getPC().setType("lu")

    # Create MPC contact conditon and assemble matrices
    mpc = dolfinx_mpc.create_contact_condition(V, mt, 4, 9)
    A = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs)
    b = dolfinx_mpc.assemble_vector(rhs, mpc)
    fem.apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                  mode=PETSc.ScatterMode.REVERSE)
    fem.set_bc(b, bcs)

    with dolfinx.common.Timer("~MPC: Solve"):
        solver.setOperators(A)
        uh = b.copy()
        uh.set(0)
        solver.solve(b, uh)
    uh.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                   mode=PETSc.ScatterMode.FORWARD)
    dolfinx_mpc.backsubstitution(mpc, uh)

    # Write solution to file
    V_mpc_cpp = dolfinx.cpp.function.FunctionSpace(mesh, V.element,
                                                   mpc.dofmap())
    V_mpc = dolfinx.FunctionSpace(None, V.ufl_element(), V_mpc_cpp)
    u_h = dolfinx.Function(V_mpc)
    u_h.vector.setArray(uh.array)
    u_h.name = "u_{0:.2f}".format(theta)

    # NOTE: Output for debug
    outfile = io.XDMFFile(comm, "output/rotated_cube3D.xdmf", "w")
    outfile.write_mesh(mesh)
    outfile.write_function(u_h, 0.0,
                           "Xdmf/Domain/"
                           + "Grid[@Name='{0:s}'][1]"
                           .format(mesh.name))
    outfile.close()

    # Solve the MPC problem using a global transformation matrix
    # and numpy solvers to get reference values
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

    # Transfer data from the MPC problem to numpy arrays for comparison
    with dolfinx.common.Timer("~MPC: Petsc->numpy"):
        A_np = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A)
        b_np = dolfinx_mpc.utils.PETScVector_to_global_numpy(b)

    with dolfinx.common.Timer("~MPC: Compare"):
        # Compare LHS, RHS and solution with reference values
        dolfinx_mpc.utils.compare_matrices(reduced_A, A_np, mpc)
        dolfinx_mpc.utils.compare_vectors(reduced_L, b_np, mpc)
        assert np.allclose(uh.array, uh_numpy[uh.owner_range[0]:
                                              uh.owner_range[1]])
    # dolfinx.common.list_timings(
    #     comm, [dolfinx.common.TimingType.wall])
