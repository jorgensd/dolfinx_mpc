# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import typing
import numba
from petsc4py import PETSc
import types
from dolfinx import function, fem, MPI
from .assemble_matrix import in_numpy_array, add_diagonal
import numpy


def backsubstitution(mpc, vector, dofmap):

    # Unravel data from MPC
    slave_cells = mpc.slave_cells()
    coefficients = mpc.coefficients()
    masters = mpc.masters_local()
    slave_cell_to_dofs = mpc.slave_cell_to_dofs()
    cell_to_slave = slave_cell_to_dofs.array()
    cell_to_slave_offset = slave_cell_to_dofs.offsets()
    slaves = mpc.slaves()
    masters_local = masters.array()
    offsets = masters.offsets()
    mpc_wrapper = (slaves, slave_cells, cell_to_slave, cell_to_slave_offset,
                   masters_local, coefficients, offsets)
    num_dofs_per_element = dofmap.dof_layout.num_dofs
    index_map = mpc.index_map()
    global_indices = index_map.indices(True)
    backsubstitution_numba(vector, dofmap.list.array(),
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
    slaves = []
    masters = []
    coeffs = []
    offsets = []
    local_min = (V.dofmap.index_map.local_range[0]
                 * V.dofmap.index_map.block_size)
    if subspace_slave is not None:
        Vsub_slave = V.sub(subspace_slave).collapse()
    if subspace_master is not None:
        Vsub_master = V.sub(subspace_master).collapse()
    for slave in slave_master_dict.keys():
        offsets.append(len(masters))
        if subspace_slave is None:
            dof = fem.locate_dofs_geometrical(V, slave) + local_min
            dof_global = numpy.vstack(MPI.comm_world.allgather(dof))[0]

        else:
            dof = fem.locate_dofs_geometrical((V.sub(subspace_slave),
                                               Vsub_slave),
                                              slave)
            for (i, d) in enumerate(dof):
                dof[i] += local_min
            dof_global = numpy.vstack(MPI.comm_world.allgather(dof))[0, 0]
        slaves.append(dof_global)
        for master in slave_master_dict[slave].keys():
            if subspace_master is None:
                dof_m = fem.locate_dofs_geometrical(V, master) + local_min
                dof_m = numpy.vstack(MPI.comm_world.allgather(dof_m))[0]
            else:
                dof_m = fem.locate_dofs_geometrical((V.sub(subspace_master),
                                                     Vsub_master),
                                                    master)
                for (i, d) in enumerate(dof_m):
                    dof_m[i] += local_min
                dof_m = numpy.vstack(MPI.comm_world.allgather(dof_m))[0, 0]

            masters.append(dof_m)
            coeffs.append(slave_master_dict[slave][master])
    offsets.append(len(masters))
    return (numpy.array(slaves), numpy.array(masters),
            numpy.array(coeffs, dtype=numpy.float64), numpy.array(offsets))


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
    import dolfinx
    import ufl
    n = dolfinx.FacetNormal(V.mesh)
    nh = dolfinx.Function(V)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = (dolfinx.Constant(V.mesh, 0)*ufl.inner(u, v)*ufl.dx
         + ufl.inner(u, v)*ufl.ds)
    ds = ufl.ds(domain=V.mesh, subdomain_data=mt, subdomain_id=mt_id)
    L = ufl.inner(n, v)*ds

    A = dolfinx.fem.assemble_matrix(a)
    ident_zeros(A)
    A.assemble()
    b = dolfinx.fem.assemble_vector(L)
    ksp = PETSc.KSP().create(V.mesh.mpi_comm())
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.solve(b, nh.vector)

    return nh


def ident_zeros(A):
    """
    Find all rows in a matrix that is zero, and add a 1 on the diagonal
    """
    assert A.size[0] == A.size[1]
    A.assemble()
    o_range = A.getOwnershipRange()
    rows = []
    for i in range(o_range[1]-o_range[0]):
        indices, values = A.getRow(o_range[0]+i)
        absrow = sum(abs(values))
        if absrow < 1e-6:
            rows.append(o_range[0] + i)
    add_diagonal(A.handle, numpy.array(rows))
