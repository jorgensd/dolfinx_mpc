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


comm = MPI.COMM_WORLD


def mesh_3D_dolfin(theta=0, ct=dolfinx.cpp.mesh.CellType.tetrahedron,
                   ext="tetrahedron", res=0.1):
    timer = dolfinx.common.Timer("Create mesh")

    def find_plane_function(p0, p1, p2):
        """
        Find plane function given three points:
        http://www.nabla.hr/CG-LinesPlanesIn3DA3.htm
        """
        v1 = np.array(p1) - np.array(p0)
        v2 = np.array(p2) - np.array(p0)

        n = np.cross(v1, v2)
        D = -(n[0] * p0[0] + n[1] * p0[1] + n[2] * p0[2])
        return lambda x: np.isclose(0,
                                    np.dot(n, x) + D)

    def over_plane(p0, p1, p2):
        """
        Returns function that checks if a point is over a plane defined
        by the points p0, p1 and p2.
        """
        v1 = np.array(p1) - np.array(p0)
        v2 = np.array(p2) - np.array(p0)

        n = np.cross(v1, v2)
        D = -(n[0] * p0[0] + n[1] * p0[1] + n[2] * p0[2])
        return lambda x: n[0] * x[0] + n[1] * x[1] + D > -n[2] * x[2]

    N = int(1 / res)
    if MPI.COMM_WORLD.rank == 0:
        mesh0 = dolfinx.UnitCubeMesh(MPI.COMM_SELF, N, N, N, ct)
        mesh1 = dolfinx.UnitCubeMesh(MPI.COMM_SELF, 2 * N, 2 * N, 2 * N, ct)
        mesh0.geometry.x[:, 2] += 1

        # Stack the two meshes in one mesh
        r_matrix = dolfinx_mpc.utils.rotation_matrix(
            [1 / np.sqrt(2), 1 / np.sqrt(2), 0], -theta)
        points = np.vstack([mesh0.geometry.x, mesh1.geometry.x])
        points = np.dot(r_matrix, points.T).T

        # Transform topology info into geometry info
        tdim0 = mesh0.topology.dim
        num_cells0 = mesh0.topology.index_map(tdim0).size_local
        cells0 = dolfinx.cpp.mesh.entities_to_geometry(mesh0, tdim0,
                                                       np.arange(num_cells0, dtype=np.int32).reshape((-1, 1)), False)
        tdim1 = mesh1.topology.dim
        num_cells1 = mesh1.topology.index_map(tdim1).size_local
        cells1 = dolfinx.cpp.mesh.entities_to_geometry(mesh1, tdim1,
                                                       np.arange(num_cells1, dtype=np.int32).reshape((-1, 1)), False)
        cells1 += mesh0.geometry.x.shape[0]

        cells = np.vstack([cells0, cells1])
        cell = ufl.Cell(ext, geometric_dimension=points.shape[1])
        domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, 1))
        mesh = dolfinx.mesh.create_mesh(MPI.COMM_SELF, cells, points, domain)
        tdim = mesh.topology.dim
        fdim = tdim - 1
        # Find information about facets to be used in MeshTags
        bottom_points = np.dot(r_matrix, np.array([[0, 0, 0], [1, 0, 0],
                                                   [0, 1, 0], [1, 1, 0]]).T)
        bottom = find_plane_function(bottom_points[:, 0], bottom_points[:, 1],
                                     bottom_points[:, 2])
        bottom_facets = dolfinx.mesh.locate_entities_boundary(
            mesh, fdim, bottom)
        top_points = np.dot(r_matrix, np.array([[0, 0, 2], [1, 0, 2],
                                                [0, 1, 2], [1, 1, 2]]).T)
        top = find_plane_function(top_points[:, 0], top_points[:, 1],
                                  top_points[:, 2])
        top_facets = dolfinx.mesh.locate_entities_boundary(
            mesh, fdim, top)

        # left_side = find_line_function(top_points[:, 0], top_points[:, 3])
        # left_facets = dolfinx.mesh.locate_entities_boundary(
        #     mesh, fdim, left_side)

        # right_side = find_line_function(top_points[:, 1], top_points[:, 2])
        # right_facets = dolfinx.mesh.locate_entities_boundary(
        #     mesh, fdim, right_side)
        if_points = np.dot(r_matrix, np.array([[0, 0, 1], [1, 0, 1],
                                               [0, 1, 1], [1, 1, 1]]).T)

        interface = find_plane_function(if_points[:, 0], if_points[:, 1],
                                        if_points[:, 2])
        i_facets = dolfinx.mesh.locate_entities_boundary(
            mesh, fdim, interface)
        mesh.topology.create_connectivity(fdim, tdim)
        top_interface = []
        bottom_interface = []
        facet_to_cell = mesh.topology.connectivity(fdim, tdim)
        num_cells = mesh.topology.index_map(tdim).size_local
        cell_midpoints = dolfinx.cpp.mesh.midpoints(mesh, tdim,
                                                    range(num_cells))
        top_cube = over_plane(if_points[:, 0], if_points[:, 1], if_points[:, 2])
        for facet in i_facets:
            i_cells = facet_to_cell.links(facet)
            assert(len(i_cells == 1))
            i_cell = i_cells[0]
            if top_cube(cell_midpoints[i_cell]):
                top_interface.append(facet)
            else:
                bottom_interface.append(facet)

        num_cells = mesh.topology.index_map(tdim).size_local
        cell_midpoints = dolfinx.cpp.mesh.midpoints(mesh, tdim,
                                                    range(num_cells))
        top_cube_marker = 2
        indices = []
        values = []
        for cell_index in range(num_cells):
            if top_cube(cell_midpoints[cell_index]):
                indices.append(cell_index)
                values.append(top_cube_marker)
        ct = dolfinx.mesh.MeshTags(mesh, tdim, np.array(
            indices, dtype=np.intc), np.array(values, dtype=np.intc))

        # Create meshtags for facet data
        markers = {3: top_facets, 4: bottom_interface, 9: top_interface,
                   5: bottom_facets}  # , 6: left_facets, 7: right_facets}
        indices = np.array([], dtype=np.intc)
        values = np.array([], dtype=np.intc)

        for key in markers.keys():
            indices = np.append(indices, markers[key])
            values = np.append(values, np.full(len(markers[key]), key,
                                               dtype=np.intc))
        mt = dolfinx.mesh.MeshTags(mesh, fdim,
                                   indices, values)
        mt.name = "facet_tags"
        fname = "meshes/mesh_{0:s}_{1:.2f}.xdmf".format(ext, theta)

        with dolfinx.io.XDMFFile(MPI.COMM_SELF, fname, "w") as o_f:
            o_f.write_mesh(mesh)
            o_f.write_meshtags(ct)
            o_f.write_meshtags(mt)
    timer.stop()


def demo_stacked_cubes(theta, ct, res, noslip, timings=False):
    if ct == dolfinx.cpp.mesh.CellType.hexahedron:
        celltype = "hexahedron"
    else:
        celltype = "tetrahedron"
    if noslip:
        type_ext = "no_slip"
    else:
        type_ext = "slip"
    dolfinx_mpc.utils.log_info("Run theta:{0:.2f}, Cell: {1:s}, Noslip: {2:b}".format(theta, celltype, noslip))

    # Read in mesh
    mesh_3D_dolfin(theta, ct, celltype, res)
    comm.barrier()
    with dolfinx.io.XDMFFile(comm, "meshes/mesh_{0:s}_{1:.2f}.xdmf".format(celltype, theta), "r") as xdmf:
        mesh = xdmf.read_mesh(name="mesh")
        tdim = mesh.topology.dim
        fdim = tdim - 1
        mesh.topology.create_connectivity(tdim, tdim)
        mesh.topology.create_connectivity(fdim, tdim)
        mt = xdmf.read_meshtags(mesh, "facet_tags")
    mesh.name = "mesh_{0:s}_{1:.2f}{2:s}".format(celltype, theta, type_ext)

    # Create functionspaces
    V = dolfinx.VectorFunctionSpace(mesh, ("Lagrange", 1))

    # Define boundary conditions
    # Bottom boundary is fixed in all directions
    u_bc = dolfinx.function.Function(V)
    with u_bc.vector.localForm() as u_local:
        u_local.set(0.0)

    bottom_facets = mt.indices[np.flatnonzero(mt.values == 5)]
    bottom_dofs = fem.locate_dofs_topological(V, fdim, bottom_facets)
    bc_bottom = fem.DirichletBC(u_bc, bottom_dofs)

    g_vec = [0, 0, -4.25e-1]
    if not noslip:
        # Helper for orienting traction
        r_matrix = dolfinx_mpc.utils.rotation_matrix(
            [1 / np.sqrt(2), 1 / np.sqrt(2), 0], -theta)

        # Top boundary has a given deformation normal to the interface
        g_vec = np.dot(r_matrix, [0, 0, -4.25e-1])

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
        return (2.0 * mu * ufl.sym(ufl.grad(v)) + lmbda * ufl.tr(ufl.sym(ufl.grad(v))) * ufl.Identity(len(v)))

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(sigma(u), ufl.grad(v)) * ufl.dx
    rhs = ufl.inner(dolfinx.Constant(mesh, (0, 0, 0)), v) * ufl.dx

    mpc = dolfinx_mpc.MultiPointConstraint(V)
    if noslip:
        with dolfinx.common.Timer("{0:d}: Create constraint".format(V.dim)):
            mpc_data = dolfinx_mpc.cpp.mpc.create_contact_inelastic_condition(V._cpp_object, mt, 4, 9)
    else:
        with dolfinx.common.Timer("{0:d}: FacetNormal".format(V.dim)):
            nh = dolfinx_mpc.utils.facet_normal_approximation(V, mt, 4)
        with dolfinx.common.Timer("{0:d}: Create-constraint".format(V.dim)):
            mpc_data = dolfinx_mpc.cpp.mpc.create_contact_slip_condition(V._cpp_object, mt, 4, 9, nh._cpp_object)

    with dolfinx.common.Timer("{0:d}: MPC".format(V.dim)):
        mpc.add_constraint_from_mpc_data(V, mpc_data)
        mpc.finalize()
    null_space = dolfinx_mpc.utils.rigid_motions_nullspace(mpc.function_space())

    # with dolfinx.common.Timer("~{0:d}:  Assemble matrix ({0:d})".format(V.dim))):
    #     A = dolfinx_mpc.assemble_matrix_cpp(a, mpc, bcs=bcs)
    # MPI.COMM_WORLD.barrier()
    with dolfinx.common.Timer("{0:d}: Assemble-matrix".format(V.dim)):
        A = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs)

    with dolfinx.common.Timer("{0:d}: Assemble-vector".format(V.dim)):
        b = dolfinx_mpc.assemble_vector(rhs, mpc)
        fem.apply_lifting(b, [a], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(b, bcs)

    # Solve Linear problem
    opts = PETSc.Options()
    # opts["ksp_rtol"] = 1.0e-8
    opts["pc_type"] = "gamg"
    # opts["pc_gamg_type"] = "agg"
    # opts["pc_gamg_coarse_eq_limit"] = 1000
    # opts["pc_gamg_sym_graph"] = True
    # opts["mg_levels_ksp_type"] = "chebyshev"
    # opts["mg_levels_pc_type"] = "jacobi"
    # opts["mg_levels_esteig_ksp_type"] = "cg"
    # opts["matptap_via"] = "scalable"
    # opts["pc_gamg_square_graph"] = 2
    # opts["pc_gamg_threshold"] = 1e-2
    # opts["help"] = None  # List all available options
    opts["ksp_view"] = None  # List progress of solver
    # Create functionspace and build near nullspace

    A.setNearNullSpace(null_space)
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setFromOptions()
    uh = b.copy()
    uh.set(0)
    with dolfinx.common.Timer("{0:d}: Solve".format(V.dim)):
        solver.solve(b, uh)
        uh.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    with dolfinx.common.Timer("{0:d}: Backsubstitution".format(V.dim)):
        mpc.backsubstitution(uh)

    it = solver.getIterationNumber()

    # Write solution to file
    u_h = dolfinx.Function(mpc.function_space())
    u_h.vector.setArray(uh.array)
    u_h.name = "u"
    with dolfinx.io.XDMFFile(comm, "results/bench_contact_{0:d}.xdmf".format(V.dim), "w") as outfile:
        outfile.write_mesh(mesh)
        outfile.write_function(u_h, 0.0, "Xdmf/Domain/Grid[@Name='{0:s}'][1]".format(mesh.name))

    # Write performance data to file
    if timings:
        num_slaves = MPI.COMM_WORLD.allreduce(mpc.num_local_slaves(), op=MPI.SUM)
        results_file = None
        if comm.rank == 0:
            results_file = open("results_bench_{0:d}.txt".format(V.dim), "w")
            print("#Dofs: {0:d}".format(V.dim), file=results_file)
            print("#Slaves: {0:d}".format(num_slaves), file=results_file)
            print("#Iterations: {0:d}".format(it), file=results_file)
        operations = ["Create-constraint", "FacetNormal",
                      "MPC", "Assemble-matrix", "Assemble-vector", "Solve", "Backsubstitution"]
        if comm.rank == 0:
            print("Operation  #Calls Avg Min Max", file=results_file)
        for op in operations:
            op_timing = dolfinx.common.timing("{0:d}: ".format(V.dim) + op)
            num_calls = op_timing[0]
            wall_time = op_timing[1]
            avg_time = comm.allreduce(wall_time, op=MPI.SUM) / comm.size
            min_time = comm.allreduce(wall_time, op=MPI.MIN)
            max_time = comm.allreduce(wall_time, op=MPI.MAX)
            if comm.rank == 0:
                print(op, num_calls, avg_time, min_time, max_time, file=results_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta", default=np.pi / 3, type=np.float64,
                        dest="theta", help="Rotation angle around axis [1, 1, 0]")
    hex = parser.add_mutually_exclusive_group(required=False)
    hex.add_argument('--hex', dest='hex', action='store_true', help="Use hexahedron mesh", default=False)
    slip = parser.add_mutually_exclusive_group(required=False)
    slip.add_argument('--no-slip', dest='noslip', action='store_true',
                      help="Use no-slip constraint", default=False)

    args = parser.parse_args()
    noslip = args.noslip
    theta = args.theta
    hex = args.hex

    ct = dolfinx.cpp.mesh.CellType.hexahedron if hex else dolfinx.cpp.mesh.CellType.tetrahedron

    # Create cache
    demo_stacked_cubes(theta=theta, ct=ct, res=0.5, noslip=noslip, timings=False)
    # Run benchmark
    for res in [0.1, 0.05, 0.025]:
        demo_stacked_cubes(theta=theta, ct=ct, res=res, noslip=noslip, timings=True)
