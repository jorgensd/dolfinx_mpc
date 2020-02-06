import dolfinx
import dolfinx_mpc
import numpy as np
from numba.typed import List
import ufl
from petsc4py import PETSc

def master_dofs(x):
    logical = np.logical_and(np.logical_or(np.isclose(x[:,0],0),np.isclose(x[:,0],1)),np.isclose(x[:,1],1))
    try:
        return np.argwhere(logical).T[0]
    except IndexError:
        return []

def slave_dofs(x):
    logical = np.logical_and(np.isclose(x[:,0],1),np.isclose(x[:,1],0))
    try:
        return np.argwhere(logical)[0]
    except IndexError:
        return []
import pytest
skip_in_parallel = pytest.mark.skipif(dolfinx.MPI.size(dolfinx.MPI.comm_world) > 1,
                                      reason="This test should only be run in serial.")


@skip_in_parallel
def test_mpc_assembly():

    # Create mesh and function space
    mesh = dolfinx.UnitSquareMesh(dolfinx.MPI.comm_world, 1, 2)
    V = dolfinx.FunctionSpace(mesh, ("Lagrange", 1))
    # Unpack mesh and dofmap data
    c = mesh.topology.connectivity(2, 0).connections()
    pos = mesh.topology.connectivity(2, 0).pos()
    geom = mesh.geometry.points
    dofs = V.dofmap.dof_array

    # Find master and slave dofs from geometrical search
    dof_coords = V.tabulate_dof_coordinates()
    masters = master_dofs(dof_coords)
    indexmap = V.dofmap.index_map
    ghost_info = (indexmap.local_range, indexmap.indices(True))

    # Build slaves and masters in parallel with global dof numbering scheme
    for i, master in enumerate(masters):
        masters[i] = masters[i]+V.dofmap.index_map.local_range[0]
    masters = dolfinx.MPI.comm_world.allgather(np.array(masters, dtype=np.int32))
    masters = np.hstack(masters)
    slaves = slave_dofs(dof_coords)
    for i, slave in enumerate(slaves):
        slaves[i] = slaves[i]+V.dofmap.index_map.local_range[0]
    slaves = dolfinx.MPI.comm_world.allgather(np.array(slaves, dtype=np.int32))
    slaves = np.hstack(slaves)
    slave_master_dict = {}
    coeffs = [0.11,0.43]
    for slave in slaves:
        slave_master_dict[slave] = {}
        for master, coeff in zip(masters,coeffs):
            slave_master_dict[slave][master] = coeff

    # Define bcs
    bcs = np.array([])
    values = np.array([])

    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    f = ufl.sin(2*ufl.pi*x[0])*ufl.sin(ufl.pi*x[1])
    l = ufl.inner(f, v)*ufl.dx

    L1 = dolfinx.fem.assemble_vector(l)

    mpc = dolfinx_mpc.MultiPointConstraint(V, slave_master_dict)
    slave_cells = np.array(mpc.slave_cells())

    # Data from mpc
    masters, coefficients = mpc.masters_and_coefficients()
    cell_to_slave, cell_to_slave_offset = mpc.cell_to_slave_mapping()
    slaves = mpc.slaves()
    offsets = mpc.master_offsets()


    # Wrapping for numba to be able to do "if i in slave_cells"
    if len(slave_cells)==0:
        sc_nb = List.empty_list(numba.types.int64)
    else:
        sc_nb = List()
    [sc_nb.append(sc) for sc in slave_cells]

    # Can be empty list locally, so has to be wrapped to be used with numba
    if len(cell_to_slave)==0:
        c2s_nb = List.empty_list(numba.types.int64)
    else:
        c2s_nb = List()
    [c2s_nb.append(c2s) for c2s in cell_to_slave]
    if len(cell_to_slave_offset)==0:
        c2so_nb = List.empty_list(numba.types.int64)
    else:
        c2so_nb = List()
    [c2so_nb.append(c2so) for c2so in cell_to_slave_offset]

    b_func = dolfinx.Function(V)
    ufc_form = dolfinx.jit.ffcx_jit(l)
    kernel = ufc_form.create_cell_integral(-1).tabulate_tensor
    for i in range(2):
        with b_func.vector.localForm() as b:
            b.set(0.0)
            dolfinx_mpc.assemble_vector_mpc(np.asarray(b), kernel, (c, pos), geom, dofs,
                                            (slaves, masters, coefficients, offsets, sc_nb, c2s_nb, c2so_nb), ghost_info, (bcs, values))

    b_func.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    # Build global K matrix from slave_master_dict
    K = np.zeros((V.dim(), V.dim()-len(slave_master_dict.keys())))
    for i in range(K.shape[0]):
        if i in slave_master_dict.keys():
            for master in slave_master_dict[i].keys():
                l = sum(master>np.array(list(slave_master_dict.keys())))
                K[i, master - l] = slave_master_dict[i][master]
        else:
            l = sum(i>np.array(list(slave_master_dict.keys())))
            K[i, i-l] = 1

            reduced_L = np.dot(K.T,L1.array)
    l = 0
    for i in range(V.dim()):
        if i in slave_master_dict.keys():
            l+= 1
            continue
        assert(np.isclose(reduced_L[i-l], b_func.vector.array[i]))
