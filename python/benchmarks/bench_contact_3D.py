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


def mesh_3D_dolfin(theta=0, ct=dolfinx.cpp.mesh.CellType.tetrahedron, ext="tetrahedron", num_refinements=0, N0=5):
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
        return lambda x: np.isclose(0, np.dot(n, x) + D)

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

    tmp_mesh_name = "tmp_mesh.xdmf"
    r_matrix = dolfinx_mpc.utils.rotation_matrix([1 / np.sqrt(2), 1 / np.sqrt(2), 0], -theta)

    if MPI.COMM_WORLD.rank == 0:
        # Create two coarse meshes and merge them
        mesh0 = dolfinx.UnitCubeMesh(MPI.COMM_SELF, N0, N0, N0, ct)
        mesh0.geometry.x[:, 2] += 1
        mesh1 = dolfinx.UnitCubeMesh(MPI.COMM_SELF, 2 * N0, 2 * N0, 2 * N0, ct)

        tdim0 = mesh0.topology.dim
        num_cells0 = mesh0.topology.index_map(tdim0).size_local
        cells0 = dolfinx.cpp.mesh.entities_to_geometry(
            mesh0, tdim0, np.arange(num_cells0, dtype=np.int32).reshape((-1, 1)), False)
        tdim1 = mesh1.topology.dim
        num_cells1 = mesh1.topology.index_map(tdim1).size_local
        cells1 = dolfinx.cpp.mesh.entities_to_geometry(mesh1, tdim1,
                                                       np.arange(num_cells1, dtype=np.int32).reshape((-1, 1)), False)
        cells1 += mesh0.geometry.x.shape[0]

        # Concatenate points and cells
        points = np.vstack([mesh0.geometry.x, mesh1.geometry.x])
        cells = np.vstack([cells0, cells1])
        cell = ufl.Cell(ext, geometric_dimension=points.shape[1])
        domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, 1))
        # Rotate mesh
        points = np.dot(r_matrix, points.T).T

        mesh = dolfinx.mesh.create_mesh(MPI.COMM_SELF, cells, points, domain)
        with dolfinx.io.XDMFFile(MPI.COMM_SELF, tmp_mesh_name, "w") as xdmf:
            xdmf.write_mesh(mesh)

    MPI.COMM_WORLD.barrier()
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, tmp_mesh_name, "r") as xdmf:
        mesh = xdmf.read_mesh()

    # Refine coarse mesh
    for i in range(num_refinements):
        mesh.topology.create_entities(mesh.topology.dim - 2)
        mesh = dolfinx.mesh.refine(mesh, redistribute=True)

    tdim = mesh.topology.dim
    fdim = tdim - 1
    # Find information about facets to be used in MeshTags
    bottom_points = np.dot(r_matrix, np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]).T)
    bottom = find_plane_function(bottom_points[:, 0], bottom_points[:, 1], bottom_points[:, 2])
    bottom_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, bottom)
    top_points = np.dot(r_matrix, np.array([[0, 0, 2], [1, 0, 2], [0, 1, 2], [1, 1, 2]]).T)
    top = find_plane_function(top_points[:, 0], top_points[:, 1], top_points[:, 2])
    top_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, top)

    # left_side = find_line_function(top_points[:, 0], top_points[:, 3])
    # left_facets = dolfinx.mesh.locate_entities_boundary(
    #     mesh, fdim, left_side)

    # right_side = find_line_function(top_points[:, 1], top_points[:, 2])
    # right_facets = dolfinx.mesh.locate_entities_boundary(
    #     mesh, fdim, right_side)

    # Determine interface facets
    if_points = np.dot(r_matrix, np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]).T)
    interface = find_plane_function(if_points[:, 0], if_points[:, 1], if_points[:, 2])
    i_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, interface)
    mesh.topology.create_connectivity(fdim, tdim)
    top_interface = []
    bottom_interface = []
    facet_to_cell = mesh.topology.connectivity(fdim, tdim)
    num_cells = mesh.topology.index_map(tdim).size_local

    # Find top and bottom interface facets
    cell_midpoints = dolfinx.cpp.mesh.midpoints(mesh, tdim, range(num_cells))
    top_cube = over_plane(if_points[:, 0], if_points[:, 1], if_points[:, 2])
    for facet in i_facets:
        i_cells = facet_to_cell.links(facet)
        assert(len(i_cells == 1))
        i_cell = i_cells[0]
        if top_cube(cell_midpoints[i_cell]):
            top_interface.append(facet)
        else:
            bottom_interface.append(facet)

    # Create cell tags
    num_cells = mesh.topology.index_map(tdim).size_local
    cell_midpoints = dolfinx.cpp.mesh.midpoints(mesh, tdim, range(num_cells))
    top_cube_marker = 2
    indices = []
    values = []
    for cell_index in range(num_cells):
        if top_cube(cell_midpoints[cell_index]):
            indices.append(cell_index)
            values.append(top_cube_marker)
    ct = dolfinx.mesh.MeshTags(mesh, tdim, np.array(indices, dtype=np.intc), np.array(values, dtype=np.intc))

    # Create meshtags for facet data
    markers = {3: top_facets, 4: bottom_interface, 9: top_interface,
               5: bottom_facets}  # , 6: left_facets, 7: right_facets}
    indices = np.array([], dtype=np.intc)
    values = np.array([], dtype=np.intc)
    for key in markers.keys():
        indices = np.append(indices, markers[key])
        values = np.append(values, np.full(len(markers[key]), key, dtype=np.intc))
    sorted_indices = np.argsort(indices)
    mt = dolfinx.mesh.MeshTags(mesh, fdim, indices[sorted_indices], values[sorted_indices])
    mt.name = "facet_tags"
    fname = f"meshes/mesh_{ext}_{theta:.2f}.xdmf"

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, fname, "w") as o_f:
        o_f.write_mesh(mesh)
        o_f.write_meshtags(ct)
        o_f.write_meshtags(mt)
    timer.stop()


def demo_stacked_cubes(theta, ct, noslip, num_refinements, N0, timings=False):
    celltype = "hexahedron" if ct == dolfinx.cpp.mesh.CellType.hexahedron else "tetrahedron"
    type_ext = "no_slip" if noslip else "slip"
    dolfinx_mpc.utils.log_info(f"Run theta: {theta:.2f}, Cell: {celltype:s}, Noslip: {noslip:b}")

    # Read in mesh
    mesh_3D_dolfin(theta=theta, ct=ct, ext=celltype, num_refinements=num_refinements, N0=N0)
    comm.barrier()
    with dolfinx.io.XDMFFile(comm, f"meshes/mesh_{celltype}_{theta:.2f}.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="mesh")
        tdim = mesh.topology.dim
        fdim = tdim - 1
        mesh.topology.create_connectivity(tdim, tdim)
        mesh.topology.create_connectivity(fdim, tdim)
        mt = xdmf.read_meshtags(mesh, "facet_tags")
    mesh.name = f"mesh_{celltype}_{theta:.2f}{type_ext:s}"

    # Create functionspaces
    V = dolfinx.VectorFunctionSpace(mesh, ("Lagrange", 1))

    # Define boundary conditions
    # Bottom boundary is fixed in all directions
    u_bc = dolfinx.Function(V)
    with u_bc.vector.localForm() as u_local:
        u_local.set(0.0)

    bottom_facets = mt.indices[np.flatnonzero(mt.values == 5)]
    bottom_dofs = fem.locate_dofs_topological(V, fdim, bottom_facets)
    bc_bottom = fem.DirichletBC(u_bc, bottom_dofs)

    g_vec = [0, 0, -4.25e-1]
    if not noslip:
        # Helper for orienting traction
        r_matrix = dolfinx_mpc.utils.rotation_matrix([1 / np.sqrt(2), 1 / np.sqrt(2), 0], -theta)

        # Top boundary has a given deformation normal to the interface
        g_vec = np.dot(r_matrix, [0, 0, -4.25e-1])

    def top_v(x):
        values = np.empty((3, x.shape[1]))
        values[0] = g_vec[0]
        values[1] = g_vec[1]
        values[2] = g_vec[2]
        return values
    u_top = dolfinx.Function(V)
    u_top.interpolate(top_v)
    u_top.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

    top_facets = mt.indices[mt.values == 3]
    top_dofs = fem.locate_dofs_topological(V, fdim, top_facets)
    bc_top = fem.DirichletBC(u_top, top_dofs)

    bcs = [bc_bottom, bc_top]

    # Elasticity parameters
    E = PETSc.ScalarType(1.0e3)
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
    rhs = ufl.inner(dolfinx.Constant(mesh, PETSc.ScalarType((0, 0, 0))), v) * ufl.dx

    dolfinx_mpc.utils.log_info("Create constraints")

    mpc = dolfinx_mpc.MultiPointConstraint(V)
    num_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    if noslip:
        with dolfinx.common.Timer(f"{num_dofs}: Contact-constraint"):
            mpc_data = dolfinx_mpc.cpp.mpc.create_contact_inelastic_condition(V._cpp_object, mt, 4, 9)
    else:
        with dolfinx.common.Timer(f"{num_dofs}: FacetNormal"):
            nh = dolfinx_mpc.utils.create_normal_approximation(V, mt.indices[mt.values == 4])
        with dolfinx.common.Timer(f"{num_dofs}: Contact-constraint"):
            mpc_data = dolfinx_mpc.cpp.mpc.create_contact_slip_condition(V._cpp_object, mt, 4, 9, nh._cpp_object)

    with dolfinx.common.Timer(f"{num_dofs}: MPC-init"):
        mpc.add_constraint_from_mpc_data(V, mpc_data)
        mpc.finalize()
    null_space = dolfinx_mpc.utils.rigid_motions_nullspace(mpc.function_space())
    dolfinx_mpc.utils.log_info(f"Num dofs: {num_dofs}")

    dolfinx_mpc.utils.log_info("Assemble matrix")
    with dolfinx.common.Timer(f"{num_dofs}: Assemble-matrix"):
        A = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs)
    dolfinx_mpc.utils.log_info("Assemble vector")
    with dolfinx.common.Timer(f"{num_dofs}: Assemble-vector"):
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
    if timings:
        opts["ksp_view"] = None  # List progress of solver
    # Create functionspace and build near nullspace

    A.setNearNullSpace(null_space)
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setFromOptions()
    uh = b.copy()
    uh.set(0)
    dolfinx_mpc.utils.log_info("Solve")
    with dolfinx.common.Timer(f"{num_dofs}: Solve"):
        solver.solve(b, uh)
        uh.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    dolfinx_mpc.utils.log_info("Backsub")
    with dolfinx.common.Timer(f"{num_dofs}: Backsubstitution"):
        mpc.backsubstitution(uh)

    it = solver.getIterationNumber()

    # Write solution to file
    u_h = dolfinx.Function(mpc.function_space())
    u_h.vector.setArray(uh.array)
    u_h.name = "u"
    with dolfinx.io.XDMFFile(comm, f"results/bench_contact_{num_dofs}.xdmf", "w") as outfile:
        outfile.write_mesh(mesh)
        outfile.write_function(u_h, 0.0, f"Xdmf/Domain/Grid[@Name='{mesh.name}'][1]")
    # Write performance data to file
    if timings:
        dolfinx_mpc.utils.log_info("Timings")
        num_slaves = MPI.COMM_WORLD.allreduce(mpc.num_local_slaves(), op=MPI.SUM)
        results_file = None
        num_procs = comm.size
        if comm.rank == 0:
            results_file = open(f"results_bench_{num_dofs}.txt", "w")
            print(f"#Procs: {num_procs}", file=results_file)
            print(f"#Dofs: {num_dofs}", file=results_file)
            print(f"#Slaves: {num_slaves}", file=results_file)
            print(f"#Iterations: {it}", file=results_file)
        operations = ["Solve", "Assemble-matrix", "MPC-init", "Contact-constraint", "FacetNormal",
                      "Assemble-vector", "Backsubstitution"]
        if comm.rank == 0:
            print("Operation  #Calls Avg Min Max", file=results_file)
        for op in operations:
            op_timing = dolfinx.common.timing(f"{num_dofs}: {op}")
            num_calls = op_timing[0]
            wall_time = op_timing[1]
            avg_time = comm.allreduce(wall_time, op=MPI.SUM) / comm.size
            min_time = comm.allreduce(wall_time, op=MPI.MIN)
            max_time = comm.allreduce(wall_time, op=MPI.MAX)
            if comm.rank == 0:
                print(op, num_calls, avg_time, min_time, max_time, file=results_file)
        dolfinx.common.list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta", default=np.pi / 3, type=np.float64,
                        dest="theta", help="Rotation angle around axis [1, 1, 0]")
    parser.add_argument("--ref", default=0, type=np.int32,
                        dest="ref", help="Numer of mesh refinements")
    parser.add_argument("--N0", default=3, type=np.int32,
                        dest="N0", help="Initial mesh resolution")

    hex = parser.add_mutually_exclusive_group(required=False)
    hex.add_argument('--hex', dest='hex', action='store_true', help="Use hexahedron mesh", default=False)
    slip = parser.add_mutually_exclusive_group(required=False)
    slip.add_argument('--no-slip', dest='noslip', action='store_true',
                      help="Use no-slip constraint", default=False)

    args = parser.parse_args()
    noslip = args.noslip
    theta = args.theta
    hex = args.hex
    ref = args.ref
    N0 = args.N0
    ct = dolfinx.cpp.mesh.CellType.hexahedron if hex else dolfinx.cpp.mesh.CellType.tetrahedron

    # Create cache
    demo_stacked_cubes(theta=theta, ct=ct, num_refinements=0, N0=3, noslip=noslip, timings=False)

    # Run benchmark
    demo_stacked_cubes(theta=theta, ct=ct, num_refinements=ref, N0=N0, noslip=noslip, timings=True)
