# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import types
import typing

import numba
import numpy
from dolfinx_mpc import cpp
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx.cpp
from dolfinx import fem, function, geometry, log
import ufl
from .assemble_matrix import in_numpy_array


def backsubstitution(mpc, vector, dofmap):

    # Unravel data from MPC
    slave_cells = mpc.slave_cells()
    coefficients = mpc.coefficients()
    masters = mpc.masters_local()
    slave_cell_to_dofs = mpc.slave_cell_to_dofs()
    cell_to_slave = slave_cell_to_dofs.array
    cell_to_slave_offset = slave_cell_to_dofs.offsets
    slaves = mpc.slaves()
    masters_local = masters.array
    offsets = masters.offsets
    mpc_wrapper = (slaves, slave_cells, cell_to_slave, cell_to_slave_offset,
                   masters_local, coefficients, offsets)
    num_dofs_per_element = (dofmap.dof_layout.num_dofs *
                            dofmap.dof_layout.block_size())
    index_map = mpc.index_map()
    global_indices = index_map.indices(True)
    backsubstitution_numba(vector, dofmap.list.array,
                           num_dofs_per_element, mpc_wrapper, global_indices)
    vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                       mode=PETSc.ScatterMode.FORWARD)
    return vector


@numba.njit(cache=True)
def backsubstitution_numba(b, dofmap, num_dofs_per_element, mpc,
                           global_indices):
    """
    Insert mpc values into vector bc
    """
    (slaves, slave_cells, cell_to_slave, cell_to_slave_offset,
     masters_local, coefficients, offsets) = mpc
    slaves_visited = numpy.empty(0, dtype=numpy.float64)

    # Loop through slave cells
    for (index, cell_index) in enumerate(slave_cells):
        cell_slaves = cell_to_slave[cell_to_slave_offset[index]:
                                    cell_to_slave_offset[index+1]]
        local_dofs = dofmap[num_dofs_per_element * cell_index:
                            num_dofs_per_element * cell_index
                            + num_dofs_per_element]

        # Find the global index of the slaves on the cell in the slaves-array
        global_slaves_index = []
        for gi in range(len(slaves)):
            if in_numpy_array(cell_slaves, slaves[gi]):
                global_slaves_index.append(gi)

        for slave_index in global_slaves_index:
            slave = slaves[slave_index]
            k = -1
            # Find local position of slave dof
            for local_dof in local_dofs:
                if global_indices[local_dof] == slave:
                    k = local_dof
            assert k != -1
            # Check if we have already inserted for this slave
            if not in_numpy_array(slaves_visited, slave):
                slaves_visited = numpy.append(slaves_visited, slave)
                slaves_masters = masters_local[offsets[slave_index]:
                                               offsets[slave_index+1]]
                slaves_coeffs = coefficients[offsets[slave_index]:
                                             offsets[slave_index+1]]
                for (master, coeff) in zip(slaves_masters, slaves_coeffs):
                    b[k] += coeff*b[master]


def slave_master_structure(V: function.FunctionSpace, slave_master_dict:
                           typing.Dict[types.FunctionType,
                                       typing.Dict[
                                           types.FunctionType, float]],
                           subspace_slave=None,
                           subspace_master=None):
    """
    Returns the data structures required to build a multi-point constraint.
    Given a nested dictionary, where the first keys are functions for
    geometrically locating the slave degrees of freedom. The values of these
    keys are another dictionary, containing functions for geometrically
    locating the master degree of freedom. The value of the nested dictionary
    is the coefficient the master degree of freedom should be multiplied with
    in the multi point constraint.
    Example:
       If u0 = alpha u1 + beta u2, u3 = beta u4 + gamma u5
       slave_master_dict = {lambda x loc_u0:{lambda x loc_u1: alpha,
                                             lambda x loc_u2: beta},
                            lambda x loc_u3:{lambda x loc_u4: beta,
                                             lambda x loc_u5: gamma}}
    """
    log.log(log.LogLevel.INFO, "Creating master-slave structure")
    slaves = []
    masters = []
    coeffs = []
    offsets = []
    owner_ranks = []
    local_to_global = V.dofmap.index_map.global_indices(False)
    if subspace_slave is not None:
        Vsub_slave = V.sub(subspace_slave).collapse()
    if subspace_master is not None:
        Vsub_master = V.sub(subspace_master).collapse()
    range_min, range_max = V.dofmap.index_map.local_range
    bs = V.dofmap.index_map.block_size
    for slave in slave_master_dict.keys():
        offsets.append(len(masters))
        if subspace_slave is None:
            dof = fem.locate_dofs_geometrical(V, slave)[:, 0]
        else:
            dof = fem.locate_dofs_geometrical((V.sub(subspace_slave),
                                               Vsub_slave),
                                              slave)[:, 0]
        dof_global = []
        if len(dof) > 0:
            for d in dof:
                dof_global.append(local_to_global[d])

        else:
            dof_global = []
        dofs_global = numpy.hstack(MPI.COMM_WORLD.allgather(dof_global))
        assert(len(set(dofs_global)) == 1)
        slaves.append(dofs_global[0])
        for master in slave_master_dict[slave].keys():
            if subspace_master is None:
                dof = fem.locate_dofs_geometrical(V, master)[:, 0]
            else:
                dof = fem.locate_dofs_geometrical((V.sub(subspace_master),
                                                   Vsub_master),
                                                  master)[:, 0]
            owner_m_global = []
            dof_m_global = []
            if len(dof) > 0:
                for d in dof:
                    if bs*range_min <= local_to_global[d] < bs*range_max:
                        dof_m_global.append(local_to_global[d])
                        owner_m_global.append(MPI.COMM_WORLD.rank)
            else:
                dof_global = []
            dofs_m_global = numpy.hstack(
                MPI.COMM_WORLD.allgather(dof_m_global))
            owners_m_global = numpy.hstack(
                MPI.COMM_WORLD.allgather(owner_m_global))

            owner_ranks.append(owners_m_global[0])

            assert(len(set(owners_m_global)) == 1)
            assert(len(set(dofs_m_global)) == 1)

            masters.append(dofs_m_global[0])
            coeffs.append(slave_master_dict[slave][master])
    offsets.append(len(masters))
    slaves = numpy.array(slaves, dtype=numpy.int64)
    masters = numpy.array(masters, dtype=numpy.int64)
    assert(not numpy.all(numpy.isin(masters, slaves)))
    return (slaves, masters, numpy.array(coeffs, dtype=numpy.float64),
            numpy.array(offsets), numpy.array(owner_ranks,
                                              dtype=numpy.int32))


def dof_close_to(x, point):
    """
    Convenience function for locating a dof close to a point use numpy
    and lambda functions.
    """
    if point is None:
        raise ValueError("Point must be supplied")
    if len(point) == 1:
        return numpy.isclose(x[0], point[0])
    elif len(point) == 2:
        return numpy.logical_and(numpy.isclose(x[0], point[0]),
                                 numpy.isclose(x[1], point[1]))
    elif len(point) == 3:
        return numpy.logical_and(
            numpy.logical_and(numpy.isclose(x[0], point[0]),
                              numpy.isclose(x[1], point[1])),
            numpy.isclose(x[2], point[2]))
    else:
        return ValueError("Point has to be 1D, 2D or 3D")


def facet_normal_approximation(V, mt, mt_id):
    timer = dolfinx.common.Timer("~MPC: Facet normal projection")
    comm = V.mesh.mpi_comm()
    n = ufl.FacetNormal(V.mesh)
    nh = dolfinx.Function(V)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    ds = ufl.ds(domain=V.mesh, subdomain_data=mt, subdomain_id=mt_id)
    a = (ufl.inner(u, v)*ufl.ds)
    L = ufl.inner(n, v)*ds

    # Find all dofs that are not boundary dofs
    imap = V.dofmap.index_map
    all_dofs = numpy.array(
        range(imap.size_local*imap.block_size), dtype=numpy.int32)
    top_facets = mt.indices[numpy.flatnonzero(mt.values == mt_id)]
    top_dofs = dolfinx.fem.locate_dofs_topological(
        V, V.mesh.topology.dim-1, top_facets)[:, 0]
    deac_dofs = all_dofs[numpy.isin(all_dofs, top_dofs, invert=True)]

    # Note there should be a better way to do this
    # Create sparsity pattern only for constraint + bc
    cpp_form = dolfinx.Form(a)._cpp_object
    pattern = dolfinx.cpp.fem.create_sparsity_pattern(cpp_form)
    block_size = V.dofmap.index_map.block_size
    blocks = numpy.unique(deac_dofs // block_size)
    cpp.mpc.add_pattern_diagonal(
        pattern, blocks, block_size)
    pattern.assemble()

    u_0 = dolfinx.Function(V)
    u_0.vector.set(0)

    bc_deac = dolfinx.fem.DirichletBC(u_0, deac_dofs)
    A = dolfinx.cpp.la.create_matrix(comm, pattern)
    A.zeroEntries()
    if cpp_form.function_spaces[0].id == cpp_form.function_spaces[1].id:
        dolfinx.cpp.fem.add_diagonal(A, cpp_form.function_spaces[0],
                                     [bc_deac], 1.0)
    # Assemble the matrix with all entries
    dolfinx.cpp.fem.assemble_matrix_petsc(A, cpp_form, [bc_deac])
    A.assemble()

    b = dolfinx.fem.assemble_vector(L)

    dolfinx.fem.apply_lifting(b, [a], [[bc_deac]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                  mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b, [bc_deac])
    opts = PETSc.Options()
    opts["ksp_type"] = "cg"
    opts["ksp_rtol"] = 1.0e-8
    # opts["help"] = None # List all available options
    # opts["ksp_view"] = None # List progress of solver

    # Solve Linear problem
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setFromOptions()
    solver.setOperators(A)
    solver.solve(b, nh.vector)
    timer.stop()
    return nh


def create_collision_constraint(V, interface_info, cell_info):
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
    dofs = V.dofmap.list.array
    dof_coords = V.tabulate_dof_coordinates()
    num_dofs_per_element = (V.dofmap.dof_layout.num_dofs *
                            V.dofmap.dof_layout.block_size())
    indexmap = V.dofmap.index_map
    bs = indexmap.block_size
    global_indices = numpy.array(indexmap.global_indices(False),
                                 dtype=numpy.int64)
    lmin = bs*indexmap.local_range[0]
    lmax = bs*indexmap.local_range[1]
    ghost_owners = indexmap.ghost_owner_rank()

    # Extract interface_info
    (mt, slave_i_value, master_i_value) = interface_info
    (ct, master_value) = cell_info

    # Find all dofs on slave interface
    slave_facets = mt.indices[numpy.flatnonzero(mt.values == slave_i_value)]
    interface_dofs = []
    for i in range(tdim):
        i_dofs = fem.locate_dofs_topological((V.sub(i), Vs[i]),
                                             fdim, slave_facets)[:, 0]
        global_dofs = global_indices[i_dofs]
        owned = numpy.logical_and(lmin <= global_dofs, global_dofs < lmax)
        interface_dofs.append(i_dofs[owned])

    # Extract all top cell values from MeshTags
    tree = geometry.BoundingBoxTree(mesh, tdim)
    ct_top_cells = ct.indices[ct.values == master_value]

    # Compute approximate facet normals for the interface
    # (Does not have unit length)
    nh = facet_normal_approximation(V, mt, slave_i_value)
    n_vec = nh.vector.getArray()

    # Create local dictionaries for the master, slaves and coeffs
    master_dict = {}
    master_owner_dict = {}
    coeffs_dict = {}
    local_slaves = []
    local_coordinates = numpy.zeros((len(interface_dofs[tdim-1]), 3),
                                    dtype=numpy.float64)
    local_normals = numpy.zeros((len(interface_dofs[tdim-1]), tdim),
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
    slaves = numpy.hstack(comm.allgather(local_slaves))
    slave_coordinates = numpy.concatenate(comm.allgather(local_coordinates))
    slave_normals = numpy.concatenate(comm.allgather(local_normals))

    # Local range of cells
    cellmap = V.mesh.topology.index_map(tdim)
    [cmin, cmax] = cellmap.local_range
    ltog_cells = numpy.array(cellmap.global_indices(False), dtype=numpy.int64)

    # Local dictionaries which will hold data for dofs
    #  at same block as the slave
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
            local_index = numpy.flatnonzero(slave == local_slaves)[0]
            for j in range(tdim-1):
                coeff = -slave_normals[i][j]/slave_normals[i][tdim-1]
                if not numpy.isclose(coeff, 0):
                    master_dict[i].append(
                        global_indices[interface_dofs[j][local_index]])
                    master_owner_dict[i].append(comm.rank)
                    coeffs_dict[i].append(coeff)

        # Find bb cells for masters on the other interface (no ghosts)
        possible_cells = numpy.array(
            geometry.compute_collisions_point(tree,
                                              slave_coordinates[i]))
        is_top = numpy.isin(possible_cells, ct_top_cells)
        is_owned = numpy.zeros(len(possible_cells), dtype=bool)
        if len(possible_cells) > 0:
            is_owned = numpy.logical_and(cmin <= ltog_cells[possible_cells],
                                         ltog_cells[possible_cells] < cmax)
        top_cells = possible_cells[numpy.logical_and(is_owned, is_top)]
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
            basis_values = cpp.mpc.\
                get_basis_functions(V._cpp_object,
                                    slave_coordinates[i],
                                    master_cell)
            for k in range(tdim):
                for local_idx, dof in enumerate(global_masters):
                    l_coeff = basis_values[local_idx, k]
                    l_coeff *= (slave_normals[i][k] /
                                slave_normals[i][tdim-1])
                    if not numpy.isclose(l_coeff, 0):
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
    masters = numpy.array([], dtype=numpy.int64)
    coeffs = numpy.array([], dtype=numpy.int64)
    owner_ranks = numpy.array([], dtype=numpy.int64)
    offsets = numpy.zeros(len(slaves)+1, dtype=numpy.int64)
    for key in master_dict.keys():
        # Gather masters from same block as the slave
        # NOTE: Should check if concatenate is costly
        master_glob = numpy.array(numpy.concatenate(comm.allgather(
            master_dict[key])), dtype=numpy.int64)
        coeff_glob = numpy.array(numpy.concatenate(comm.allgather(
            coeffs_dict[key])), dtype=PETSc.ScalarType)
        owner_glob = numpy.array(numpy.concatenate(comm.allgather(
            master_owner_dict[key])), dtype=numpy.int64)

        # Check if masters on the other interface is appearing
        # on multiple processors and select one other processor at random
        occurances = numpy.array(comm.allgather(
            other_side_dict[key]), dtype=numpy.int32)
        locations = numpy.where(occurances > 0)[0]
        index = numpy.random.choice(locations.shape[0],
                                    1, replace=False)[0]

        # Get masters from selected processor
        other_masters = comm.allgather(other_master_dict[key])[
            locations[index]]
        other_coeffs = comm.allgather(other_coeffs_dict[key])[locations[index]]
        other_master_owner = comm.allgather(other_master_owner_dict[key])[
            locations[index]]

        # Gather info globally
        masters = numpy.hstack([masters, master_glob, other_masters])
        coeffs = numpy.hstack([coeffs, coeff_glob, other_coeffs])
        owner_ranks = numpy.hstack(
            [owner_ranks, owner_glob, other_master_owner])
        offsets[key+1] = len(masters)

    return (slaves, masters, coeffs, offsets, owner_ranks)
