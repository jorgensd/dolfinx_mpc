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
import dolfinx.geometry as geometry
import dolfinx.io as io
import dolfinx.fem as fem
import meshio


def find_master_slave_relationship(V, interface_info, cell_info):
    """
    Function return the master, slaves, coefficients and offsets
    for coupling two interfaces.
    Input:
        V - The function space that should be constrained.
        interface_info - Tuple containing a mesh function
                         (for facets) and two ints, one specifying the
                          slave interface, the other the master interface.
        cell_info - Tuple containing a mesh function (for cells)
                    and an int indicating which cells should be
                    master cells.
    """

    # Extract mesh info
    mesh = V.mesh
    tdim = V.mesh.topology.dim
    fdim = tdim - 1

    # Collapse sub spaces
    Vs = [V.sub(i).collapse() for i in range(tdim)]

    # Get dofmap info
    comm = V.mesh.mpi_comm()
    dofs = V.dofmap.list.array()
    dof_coords = V.tabulate_dof_coordinates()
    num_dofs_per_element = V.dofmap.dof_layout.num_dofs
    indexmap = V.dofmap.index_map
    bs = indexmap.block_size
    global_indices = np.array(indexmap.global_indices(False),
                              dtype=np.int64)
    lmin = bs*indexmap.local_range[0]
    lmax = bs*indexmap.local_range[1]
    ghost_owners = indexmap.ghost_owner_rank()

    # Extract interface_info
    (mt, slave_i_value, master_i_value) = interface_info
    (ct, master_value) = cell_info

    # Find all dofs on slave interface
    slave_facets = mt.indices[np.flatnonzero(mt.values == slave_i_value)]
    interface_dofs = []
    for i in range(tdim):
        i_dofs = fem.locate_dofs_topological((V.sub(i), Vs[i]),
                                             fdim, slave_facets)[:, 0]
        global_dofs = global_indices[i_dofs]
        owned = np.logical_and(lmin <= global_dofs, global_dofs < lmax)
        interface_dofs.append(i_dofs[owned])

    # Extract all top cell values from MeshTags
    tree = geometry.BoundingBoxTree(mesh, tdim)
    ct_top_cells = ct.indices[ct.values == master_value]

    # Compute approximate facet normals for the interface
    # (Does not have unit length)
    nh = dolfinx_mpc.facet_normal_approximation(V, mt, slave_i_value)
    n_vec = nh.vector.getArray()
    ff = io.XDMFFile(comm, "nvec.xdmf", "w")
    ff.write_mesh(nh.function_space.mesh)
    ff.write_function(nh)
    ff.close()
    # Create local dictionaries for the master, slaves and coeffs
    master_dict = {}
    master_owner_dict = {}
    coeffs_dict = {}
    local_slaves = []
    local_coordinates = np.zeros((len(interface_dofs[tdim-1]), 3),
                                 dtype=np.float64)
    local_normals = np.zeros((len(interface_dofs[tdim-1]), tdim),
                             dtype=PETSc.ScalarType)

    # Gather all slaves fromm each processor,
    # with its corresponding normal vector and coordinate
    for i in range(len(interface_dofs[tdim-1])):
        local_slave = interface_dofs[tdim-1][i]
        global_slave = global_indices[local_slave]
        if lmin <= global_slave < lmax:
            local_slaves.append(global_slave)
            local_coordinates[i] = dof_coords[local_slave]
            for j in range(tdim):
                local_normals[i, j] = n_vec[interface_dofs[j][i]]

    # Gather info on all processors
    slaves = np.hstack(comm.allgather(local_slaves))
    slave_coordinates = np.concatenate(comm.allgather(local_coordinates))
    slave_normals = np.concatenate(comm.allgather(local_normals))

    # Local range of cells
    cellmap = V.mesh.topology.index_map(tdim)
    [cmin, cmax] = cellmap.local_range
    ltog_cells = np.array(cellmap.global_indices(False), dtype=np.int64)

    # Local dictionaries which will hold data for dofs at same block as the slave
    master_dict = {}
    master_owner_dict = {}
    coeffs_dict = {}
    # Local dictionaries for the interface comming into contact with the slaves
    other_master_dict = {}
    other_master_owner_dict = {}
    other_coeffs_dict = {}
    other_side_dict = {}

    for i, slave in enumerate(slaves):
        master_dict[i] = []
        master_owner_dict[i] = []
        coeffs_dict[i] = []
        other_master_dict[i] = []
        other_coeffs_dict[i] = []
        other_master_owner_dict[i] = []
        other_side_dict[i] = 0
        # Find all masters from same coordinate as the slave (same proc)
        if lmin <= slave < lmax:
            local_index = np.flatnonzero(slave == local_slaves)[0]
            for j in range(tdim-1):
                coeff = -slave_normals[i][j]/slave_normals[i][tdim-1]
                if not np.isclose(coeff, 0):
                    master_dict[i].append(
                        global_indices[interface_dofs[j][local_index]])
                    master_owner_dict[i].append(comm.rank)
                    coeffs_dict[i].append(coeff)

        # Find bb cells for masters on the other interface (no ghosts)
        possible_cells = np.array(
            geometry.compute_collisions_point(tree,
                                              slave_coordinates[i]))
        is_top = np.isin(possible_cells, ct_top_cells)
        is_owned = np.zeros(len(possible_cells), dtype=bool)
        if len(possible_cells) > 0:
            is_owned = np.logical_and(cmin <= ltog_cells[possible_cells],
                                      ltog_cells[possible_cells] < cmax)
        top_cells = possible_cells[np.logical_and(is_owned, is_top)]
        # Check if actual cell in mesh is within a distance of 1e-14
        close_cell = dolfinx.cpp.geometry.select_colliding_cells(
            mesh, list(top_cells), slave_coordinates[i], 1)
        if len(close_cell) > 0:
            master_cell = close_cell[0]
            master_dofs = dofs[num_dofs_per_element * master_cell:
                               num_dofs_per_element * master_cell
                               + num_dofs_per_element]
            global_masters = global_indices[master_dofs]

            # Get basis values, and add coeffs of sub space
            # to masters with corresponding coeff
            basis_values = dolfinx_mpc.cpp.mpc.\
                get_basis_functions(V._cpp_object,
                                    slave_coordinates[i],
                                    master_cell)
            for k in range(tdim):
                for local_idx, dof in enumerate(global_masters):
                    l_coeff = basis_values[local_idx, k]
                    l_coeff *= (slave_normals[i][k] /
                                slave_normals[i][tdim-1])
                    if not np.isclose(l_coeff, 0):
                        # If master i ghost, find its owner
                        if lmin <= dof < lmax:
                            other_master_owner_dict[i].append(comm.rank)
                        else:
                            other_master_owner_dict[i].append(ghost_owners[
                                master_dofs[local_idx] //
                                bs-indexmap.size_local])
                        other_master_dict[i].append(dof)
                        other_coeffs_dict[i].append(l_coeff)
            other_side_dict[i] += 1

    # Gather entries on all processors
    masters = np.array([], dtype=np.int64)
    coeffs = np.array([], dtype=np.int64)
    owner_ranks = np.array([], dtype=np.int64)
    offsets = np.zeros(len(slaves)+1, dtype=np.int64)
    for key in master_dict.keys():
        # Gather masters from same block as the slave
        # NOTE: Should check if concatenate is costly
        master_glob = np.array(np.concatenate(comm.allgather(
            master_dict[key])), dtype=np.int64)
        coeff_glob = np.array(np.concatenate(comm.allgather(
            coeffs_dict[key])), dtype=PETSc.ScalarType)
        owner_glob = np.array(np.concatenate(comm.allgather(
            master_owner_dict[key])), dtype=np.int64)

        # Check if masters on the other interface is appearing
        # on multiple processors and select one other processor at random
        occurances = np.array(comm.allgather(
            other_side_dict[key]), dtype=np.int32)
        locations = np.where(occurances > 0)[0]
        index = np.random.choice(locations.shape[0],
                                 1, replace=False)[0]

        # Get masters from selected processor
        other_masters = comm.allgather(other_master_dict[key])[
            locations[index]]
        other_coeffs = comm.allgather(other_coeffs_dict[key])[locations[index]]
        other_master_owner = comm.allgather(other_master_owner_dict[key])[
            locations[index]]

        # Gather info globally
        masters = np.hstack([masters, master_glob, other_masters])
        coeffs = np.hstack([coeffs, coeff_glob, other_coeffs])
        owner_ranks = np.hstack([owner_ranks, owner_glob, other_master_owner])
        offsets[key+1] = len(masters)

    return (slaves, masters, coeffs, offsets, owner_ranks)


def generate_hex_box(x0, y0, z0, x1, y1, z1, theta, res, facet_markers,
                     volume_marker=None):
    """
    Generate the box [x0,y0,z0]x[y0,y1,z0], rotated theta degrees around
    the origin over axis [0,1,1] with resolution res,
    where markers are is an array containing markers
    array of markers for [left, back, top, bottom, front, right]
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

        ct = xdmf.read_meshtags(mesh, "Grid")

    with io.XDMFFile(comm, "facet_hex_cube{0:.2f}.xdmf".format(theta),
                     "r") as xdmf:
        mt = xdmf.read_meshtags(mesh, "Grid")
    top_cube_marker = 2

    # Create functionspaces
    V = dolfinx.VectorFunctionSpace(mesh, ("Lagrange", 1))
    io.VTKFile("outmesh.pvd").write(mesh)
    # Helper for orienting traction
    r_matrix = pygmsh.helpers.rotation_matrix(
        [1/np.sqrt(2), 1/np.sqrt(2), 0], -theta)
    g_vec = np.dot(r_matrix, [0, 0, -4e-1])

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

    # Apply boundary conditions
    fem.apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                  mode=PETSc.ScatterMode.REVERSE)
    fem.set_bc(b, bcs)

    # Create functionspace and build near nullspace
    Vmpc_cpp = dolfinx.cpp.function.FunctionSpace(mesh, V.element,
                                                  mpc.mpc_dofmap())
    Vmpc = dolfinx.FunctionSpace(None, V.ufl_element(), Vmpc_cpp)
    null_space = dolfinx_mpc.utils.build_elastic_nullspace(Vmpc)
    A.setNearNullSpace(null_space)

    solver = PETSc.KSP().create(comm)
    solver.setType("preonly")
    solver.setTolerances(rtol=1.0e-14)
    solver.getPC().setType("lu")
    solver.setOperators(A)
    uh = b.copy()
    uh.set(0)
    with dolfinx.common.Timer("MPC: Solve"):
        solver.solve(b, uh)
    uh.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                   mode=PETSc.ScatterMode.FORWARD)

    # Back substitute to slave dofs
    with dolfinx.common.Timer("MPC: Backsubstitute"):
        dolfinx_mpc.backsubstitution(mpc, uh, V.dofmap)

    # Write solution to file
    u_h = dolfinx.Function(Vmpc)
    u_h.vector.setArray(uh.array)
    u_h.name = "u_{0:.2f}".format(theta)

    # NOTE: Output for debug
    outfile = io.XDMFFile(comm, "rotated_cube3D.xdmf", "w")
    outfile.write_mesh(mesh)
    outfile.write_function(u_h, 0.0,
                           "Xdmf/Domain/"
                           + "Grid[@Name='{0:s}'][1]"
                           .format(mesh.name))
    outfile.close()

    # Transfer data from the MPC problem to numpy arrays for comparison
    with dolfinx.common.Timer("MPC: Petsc->numpy"):
        A_mpc_np = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A)
        mpc_vec_np = dolfinx_mpc.utils.PETScVector_to_global_numpy(b)

    # Solve the MPC problem using a global transformation matrix
    # and numpy solvers to get reference values
    dolfinx_mpc.utils.log_info(
        "Solving reference problem with global matrix (using numpy)")

    with dolfinx.common.Timer("MPC: Reference problem"):

        # Generate reference matrices and unconstrained solution
        A_org = fem.assemble_matrix(a, bcs)

        A_org.assemble()
        L_org = fem.assemble_vector(lhs)
        fem.apply_lifting(L_org, [a], [bcs])
        L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                          mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(L_org, bcs)

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
    with dolfinx.common.Timer("MPC: Compare"):
        # Compare LHS, RHS and solution with reference values
        dolfinx_mpc.utils.compare_matrices(reduced_A, A_mpc_np, slaves)
        dolfinx_mpc.utils.compare_vectors(reduced_L, mpc_vec_np, slaves)
        assert np.allclose(uh.array, uh_numpy[uh.owner_range[0]:
                                              uh.owner_range[1]])
    # dolfinx.common.list_timings(
    #     comm, [dolfinx.common.TimingType.wall])


test_cube_contact()
