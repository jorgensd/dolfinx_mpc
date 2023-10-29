# Copyright (C) 2020-2021 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
#
# Multi point constraint problem for linear elasticity with slip conditions
# between two cubes.

import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import basix.ufl
import numpy as np
from basix.ufl import element
from dolfinx import default_real_type, default_scalar_type
from dolfinx.common import Timer, TimingType, list_timings, timing
from dolfinx.cpp.mesh import entities_to_geometry
from dolfinx.fem import (Constant, Function, dirichletbc, form, functionspace,
                         locate_dofs_topological, set_bc)
from dolfinx.io import XDMFFile
from dolfinx.mesh import (CellType, compute_midpoints, create_mesh,
                          create_unit_cube, locate_entities_boundary, meshtags,
                          refine)
from dolfinx_mpc.utils import (create_normal_approximation, log_info,
                               rigid_motions_nullspace, rotation_matrix)
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (Identity, Mesh, TestFunction, TrialFunction, dx, grad, inner,
                 sym, tr)

from dolfinx_mpc import (MultiPointConstraint, apply_lifting, assemble_matrix,
                         assemble_vector)

comm = MPI.COMM_WORLD

if default_real_type == np.float32:
    warnings.warn(
        "Demo not supported in single precision as reading from XDMF only support double precision meshes")
    exit(0)


def mesh_3D_dolfin(theta=0, ct=CellType.tetrahedron, ext="tetrahedron", num_refinements=0, N0=5):
    timer = Timer("Create mesh")

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
    tmp_mesh_name = Path("tmp_mesh.xdmf").absolute()
    tmp_mesh_name.parent.mkdir(exist_ok=True)
    r_matrix = rotation_matrix([1 / np.sqrt(2), 1 / np.sqrt(2), 0], -theta)

    if MPI.COMM_WORLD.rank == 0:
        # Create two coarse meshes and merge them
        mesh0 = create_unit_cube(MPI.COMM_SELF, N0, N0, N0, ct)
        mesh0.geometry.x[:, 2] += 1
        mesh1 = create_unit_cube(MPI.COMM_SELF, 2 * N0, 2 * N0, 2 * N0, ct)

        tdim0 = mesh0.topology.dim
        num_cells0 = mesh0.topology.index_map(tdim0).size_local
        cells0 = entities_to_geometry(mesh0._cpp_object, tdim0, np.arange(
            num_cells0, dtype=np.int32).reshape((-1, 1)), False)
        tdim1 = mesh1.topology.dim
        num_cells1 = mesh1.topology.index_map(tdim1).size_local
        cells1 = entities_to_geometry(mesh1._cpp_object, tdim1, np.arange(
            num_cells1, dtype=np.int32).reshape((-1, 1)), False)
        cells1 += mesh0.geometry.x.shape[0]

        # Concatenate points and cells
        points = np.vstack([mesh0.geometry.x, mesh1.geometry.x])
        cells = np.vstack([cells0, cells1])
        domain = Mesh(element("Lagrange", 1))
        # Rotate mesh
        points = np.dot(r_matrix, points.T).T.astype(default_real_type)

        mesh = create_mesh(MPI.COMM_SELF, cells, points, domain)
        with XDMFFile(MPI.COMM_SELF, tmp_mesh_name, "w") as xdmf:
            xdmf.write_mesh(mesh)

    MPI.COMM_WORLD.barrier()
    with XDMFFile(MPI.COMM_WORLD, tmp_mesh_name, "r") as xdmf:
        mesh = xdmf.read_mesh()
    # Refine coarse mesh
    for i in range(num_refinements):
        mesh.topology.create_entities(mesh.topology.dim - 2)
        mesh = refine(mesh, redistribute=True)

    tdim = mesh.topology.dim
    fdim = tdim - 1
    # Find information about facets to be used in meshtags
    bottom_points = np.dot(r_matrix, np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]).T)
    bottom = find_plane_function(bottom_points[:, 0], bottom_points[:, 1], bottom_points[:, 2])
    bottom_facets = locate_entities_boundary(mesh, fdim, bottom)
    top_points = np.dot(r_matrix, np.array([[0, 0, 2], [1, 0, 2], [0, 1, 2], [1, 1, 2]]).T)
    top = find_plane_function(top_points[:, 0], top_points[:, 1], top_points[:, 2])
    top_facets = locate_entities_boundary(mesh, fdim, top)

    # Determine interface facets
    if_points = np.dot(r_matrix, np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]).T)
    interface = find_plane_function(if_points[:, 0], if_points[:, 1], if_points[:, 2])
    i_facets = locate_entities_boundary(mesh, fdim, interface)
    mesh.topology.create_connectivity(fdim, tdim)
    top_interface = []
    bottom_interface = []
    facet_to_cell = mesh.topology.connectivity(fdim, tdim)
    num_cells = mesh.topology.index_map(tdim).size_local

    # Find top and bottom interface facets
    cell_midpoints = compute_midpoints(mesh, tdim, range(num_cells))
    top_cube = over_plane(if_points[:, 0], if_points[:, 1], if_points[:, 2])
    for facet in i_facets:
        i_cells = facet_to_cell.links(facet)
        assert len(i_cells == 1)
        i_cell = i_cells[0]
        if top_cube(cell_midpoints[i_cell]):
            top_interface.append(facet)
        else:
            bottom_interface.append(facet)

    # Create cell tags
    num_cells = mesh.topology.index_map(tdim).size_local
    cell_midpoints = compute_midpoints(mesh, tdim, range(num_cells))
    top_cube_marker = 2
    indices = []
    values = []
    for cell_index in range(num_cells):
        if top_cube(cell_midpoints[cell_index]):
            indices.append(cell_index)
            values.append(top_cube_marker)
    ct = meshtags(mesh, tdim, np.array(indices, dtype=np.intc), np.array(values, dtype=np.intc))

    # Create meshtags for facet data
    markers = {3: top_facets, 4: bottom_interface, 9: top_interface,
               5: bottom_facets}  # , 6: left_facets, 7: right_facets}
    indices = np.array([], dtype=np.intc)
    values = np.array([], dtype=np.intc)
    for key in markers.keys():
        indices = np.append(indices, markers[key])
        values = np.append(values, np.full(len(markers[key]), key, dtype=np.intc))
    sorted_indices = np.argsort(indices)
    mt = meshtags(mesh, fdim, indices[sorted_indices], values[sorted_indices])
    mt.name = "facet_tags"
    mesh_dir = Path("meshes").absolute()
    mesh_dir.mkdir(exist_ok=True)
    fname = mesh_dir / f"mesh_{ext}_{theta:.2f}.xdmf"

    with XDMFFile(mesh.comm, fname, "w") as o_f:
        o_f.write_mesh(mesh)
        o_f.write_meshtags(ct, x=mesh.geometry)
        o_f.write_meshtags(mt, x=mesh.geometry)
    timer.stop()


def demo_stacked_cubes(theta, ct, noslip, num_refinements, N0, timings=False):
    celltype = "hexahedron" if ct == CellType.hexahedron else "tetrahedron"
    type_ext = "no_slip" if noslip else "slip"
    log_info(f"Run theta: {theta:.2f}, Cell: {celltype:s}, Noslip: {noslip:b}")

    # Read in mesh
    mesh_3D_dolfin(theta=theta, ct=ct, ext=celltype, num_refinements=num_refinements, N0=N0)
    comm.barrier()
    mesh_dir = Path("meshes").absolute()
    with XDMFFile(comm, mesh_dir / f"mesh_{celltype}_{theta:.2f}.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="mesh")
        tdim = mesh.topology.dim
        fdim = tdim - 1
        mesh.topology.create_connectivity(tdim, tdim)
        mesh.topology.create_connectivity(fdim, tdim)
        mt = xdmf.read_meshtags(mesh, "facet_tags")
    mesh.name = f"mesh_{celltype}_{theta:.2f}{type_ext:s}"

    # Create functionspaces
    el = basix.ufl.element("Lagrange", mesh.topology.cell_name(), 1, shape=(mesh.geometry.dim, ))
    V = functionspace(mesh, el)

    # Define boundary conditions
    # Bottom boundary is fixed in all directions
    u_bc = Function(V)
    with u_bc.vector.localForm() as u_local:
        u_local.set(0.0)
    u_bc.vector.destroy()

    bottom_dofs = locate_dofs_topological(V, fdim, mt.find(5))
    bc_bottom = dirichletbc(u_bc, bottom_dofs)

    g_vec = [0, 0, -4.25e-1]
    if not noslip:
        # Helper for orienting traction
        r_matrix = rotation_matrix([1 / np.sqrt(2), 1 / np.sqrt(2), 0], -theta)

        # Top boundary has a given deformation normal to the interface
        g_vec = np.dot(r_matrix, [0, 0, -4.25e-1])

    def top_v(x):
        values = np.empty((3, x.shape[1]))
        values[0] = g_vec[0]
        values[1] = g_vec[1]
        values[2] = g_vec[2]
        return values
    u_top = Function(V)
    u_top.interpolate(top_v)
    u_top.x.scatter_forward()

    top_dofs = locate_dofs_topological(V, fdim, mt.find(3))
    bc_top = dirichletbc(u_top, top_dofs)

    bcs = [bc_bottom, bc_top]

    # Elasticity parameters
    E = default_scalar_type(1.0e3)
    nu = 0
    mu = Constant(mesh, E / (2.0 * (1.0 + nu)))
    lmbda = Constant(mesh, E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))

    # Stress computation
    def sigma(v):
        return (2.0 * mu * sym(grad(v)) + lmbda * tr(sym(grad(v))) * Identity(len(v)))
    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(sigma(u), grad(v)) * dx
    rhs = inner(Constant(mesh, default_scalar_type((0, 0, 0))), v) * dx

    log_info("Create constraints")

    mpc = MultiPointConstraint(V)
    num_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    if noslip:
        with Timer(f"{num_dofs}: Contact-constraint"):
            mpc.create_contact_inelastic_condition(mt, 4, 9)
    else:
        with Timer(f"{num_dofs}: FacetNormal"):
            nh = create_normal_approximation(V, mt, 4)
        with Timer(f"{num_dofs}: Contact-constraint"):
            mpc.create_contact_slip_condition(mt, 4, 9, nh)

    with Timer(f"{num_dofs}: MPC-init"):
        mpc.finalize()
    null_space = rigid_motions_nullspace(mpc.function_space)
    log_info(f"Num dofs: {num_dofs}")

    log_info("Assemble matrix")
    bilinear_form = form(a)
    linear_form = form(rhs)
    with Timer(f"{num_dofs}: Assemble-matrix (C++)"):
        A = assemble_matrix(bilinear_form, mpc, bcs=bcs)
    with Timer(f"{num_dofs}: Assemble-vector (C++)"):
        b = assemble_vector(linear_form, mpc)
    apply_lifting(b, [bilinear_form], [bcs], mpc)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
    set_bc(b, bcs)
    list_timings(MPI.COMM_WORLD, [TimingType.wall])

    # Solve Linear problem
    opts = PETSc.Options()  # type: ignore
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
    solver = PETSc.KSP().create(comm)  # type: ignore
    solver.setOperators(A)
    solver.setFromOptions()
    uh = Function(mpc.function_space)
    uh.x.array[:] = 0
    log_info("Solve")
    with Timer(f"{num_dofs}: Solve"):
        solver.solve(b, uh.vector)
        uh.x.scatter_forward()
    log_info("Backsub")
    with Timer(f"{num_dofs}: Backsubstitution"):
        mpc.backsubstitution(uh)

    it = solver.getIterationNumber()

    # Write solution to file
    results = Path("results").absolute()
    results.mkdir(exist_ok=True)
    with XDMFFile(comm, results / f"bench_contact_{num_dofs}.xdmf", "w") as outfile:
        outfile.write_mesh(mesh)
        outfile.write_function(uh, 0.0, f"Xdmf/Domain/Grid[@Name='{mesh.name}'][1]")
    # Write performance data to file
    if timings:
        log_info("Timings")
        num_slaves = MPI.COMM_WORLD.allreduce(mpc.num_local_slaves, op=MPI.SUM)
        results_file = None
        num_procs = comm.size
        if comm.rank == 0:
            results_file = open(results / f"results_bench_{num_dofs}.txt", "w")
            print(f"#Procs: {num_procs}", file=results_file)
            print(f"#Dofs: {num_dofs}", file=results_file)
            print(f"#Slaves: {num_slaves}", file=results_file)
            print(f"#Iterations: {it}", file=results_file)
        operations = ["Solve", "Assemble-matrix (C++)", "MPC-init", "Contact-constraint", "FacetNormal",
                      "Assemble-vector (C++)", "Backsubstitution"]
        if comm.rank == 0:
            print("Operation  #Calls Avg Min Max", file=results_file)
        for op in operations:
            op_timing = timing(f"{num_dofs}: {op}")
            num_calls = op_timing[0]
            wall_time = op_timing[1]
            avg_time = comm.allreduce(wall_time, op=MPI.SUM) / comm.size
            min_time = comm.allreduce(wall_time, op=MPI.MIN)
            max_time = comm.allreduce(wall_time, op=MPI.MAX)
            if comm.rank == 0:
                print(op, num_calls, avg_time, min_time, max_time, file=results_file)
        list_timings(MPI.COMM_WORLD, [TimingType.wall])


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
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
    ct = CellType.hexahedron if args.hex else CellType.tetrahedron

    # Create cache
    demo_stacked_cubes(theta=args.theta, ct=ct, num_refinements=0, N0=3, noslip=args.noslip, timings=False)

    # Run benchmark
    demo_stacked_cubes(theta=args.theta, ct=ct, num_refinements=args.ref, N0=args.N0, noslip=args.noslip, timings=True)
