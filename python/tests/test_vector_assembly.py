import dolfinx
import dolfinx_mpc
import numpy as np
from numba.typed import List
import ufl
from petsc4py import PETSc
import numba
import pytest

def master_dofs(x):
    dofs, vals = [],[]
    logical = np.logical_and(np.isclose(x[:,0],0),np.isclose(x[:,1],1))
    try:
        dof, val = np.argwhere(logical).T[0][0], 0.43
        dofs.append(dof)
        vals.append(val)
    except IndexError:
        pass
    logical = np.logical_and(np.isclose(x[:,0],1), np.isclose(x[:,1],1))
    try:
        dof, val = np.argwhere(logical).T[0][0], 0.11
        dofs.append(dof)
        vals.append(val)
    except IndexError:
        pass
    return dofs, vals

def slave_dofs(x):
    logical = np.logical_and(np.isclose(x[:,0],1),np.isclose(x[:,1],0))
    try:
        return np.argwhere(logical)[0]
    except IndexError:
        return []

def slaves_2(x):
    logical = np.logical_and(np.isclose(x[:,0],0),np.isclose(x[:,1],0))
    try:
        return np.argwhere(logical)[0]
    except IndexError:
        return []

def masters_2(x):
    logical = np.logical_and(np.isclose(x[:,0],1),np.isclose(x[:,1],1))
    try:
        return np.argwhere(logical).T[0], [0.69]
    except IndexError:
        return [], []

def masters_3(x):
    logical = np.logical_and(np.isclose(x[:,0],0),np.isclose(x[:,1],1))
    try:
        return np.argwhere(logical).T[0], [0.69]
    except IndexError:
        return [], []

@pytest.mark.parametrize("masters_2", [masters_2, masters_3])
def test_mpc_assembly(masters_2):

    # Create mesh and function space
    mesh = dolfinx.UnitSquareMesh(dolfinx.MPI.comm_world, 3, 5)
    V = dolfinx.FunctionSpace(mesh, ("Lagrange", 1))
    # Unpack mesh and dofmap data
    c = mesh.topology.connectivity(2, 0).array()
    pos = mesh.topology.connectivity(2, 0).offsets()
    geom = mesh.geometry.points
    dofs = V.dofmap.dof_array
    x = V.tabulate_dof_coordinates()
    masters, coeffs = master_dofs(x)
    for i, master in enumerate(masters):
        masters[i] = masters[i]+V.dofmap.index_map.local_range[0]
    masters = dolfinx.MPI.comm_world.allgather(np.array(masters, dtype=np.int32))
    coeffs = dolfinx.MPI.comm_world.allgather(np.array(coeffs, dtype=np.float32))
    masters = np.hstack(masters)
    coeffs = np.hstack(coeffs)
    slaves = slave_dofs(x)
    for i, slave in enumerate(slaves):
        slaves[i] = slaves[i]+V.dofmap.index_map.local_range[0]
    slaves = dolfinx.MPI.comm_world.allgather(np.array(slaves, dtype=np.int32))
    slaves = np.hstack(slaves)
    slave_master_dict = {}
    for slave in slaves:
        slave_master_dict[slave] = {}
        for master, coeff in zip(masters,coeffs):
            slave_master_dict[slave][master] = coeff
    # Second constraint
    masters2, coeffs2 = masters_2(x)
    slaves2 = slaves_2(x)
    for i, slave in enumerate(slaves2):
        slaves2[i] = slaves2[i]+V.dofmap.index_map.local_range[0]
    slaves2 = dolfinx.MPI.comm_world.allgather(np.array(slaves2, dtype=np.int32))
    slaves2 = np.hstack(slaves2)
    for i, master in enumerate(masters2):
        masters2[i] = masters2[i]+V.dofmap.index_map.local_range[0]
    masters2 = dolfinx.MPI.comm_world.allgather(np.array(masters2, dtype=np.int32))
    masters2 = np.hstack(masters2)
    coeffs2 = dolfinx.MPI.comm_world.allgather(np.array(coeffs2, dtype=np.float32))
    coeffs2 = np.hstack(coeffs2)

    for slave in slaves2:
        slave_master_dict[slave] = {}
        for master, coeff in zip(masters2,coeffs2):
            slave_master_dict[slave][master] = coeff


    # Define bcs
    bcs = np.array([])
    values = np.array([])
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx
    f = dolfinx.Constant(mesh, 1)
    # f = ufl.sin(2*ufl.pi*x[0])*ufl.sin(ufl.pi*x[1])
    l = ufl.inner(f, v)*ufl.dx

    L1 = dolfinx.fem.assemble_vector(l)
    L1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    mpc = dolfinx_mpc.MultiPointConstraint(V, slave_master_dict)
    slave_cells = np.array(mpc.slave_cells())

    # Data from mpc
    masters, coefficients = mpc.masters_and_coefficients()
    cell_to_slave, cell_to_slave_offset = mpc.cell_to_slave_mapping()
    slaves = mpc.slaves()
    offsets = mpc.master_offsets()

    # Assemble matrix to obtain sparsitypattern
    A2 = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs)

    # Get index map and ghost info
    index_map = mpc.index_map()
    ghost_info = (index_map.local_range, index_map.indices(True),index_map.ghosts)

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
    b_func = dolfinx.cpp.la.create_vector(index_map)

    ufc_form = dolfinx.jit.ffcx_jit(l)
    kernel = ufc_form.create_cell_integral(-1).tabulate_tensor
    for i in range(1):
        with b_func.localForm() as b:
            b.set(0.0)
            dolfinx_mpc.assemble_vector_numba(np.asarray(b), kernel, (c, pos), geom, dofs,
                                            (slaves, masters, coefficients, offsets, sc_nb, c2s_nb, c2so_nb), ghost_info, (bcs, values))

    b_func.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    K = np.zeros((V.dim(), V.dim()-len(slave_master_dict.keys())))
    vec = np.zeros(V.dim())
    mpc_vec = np.zeros(V.dim())
    for i in range(K.shape[0]):
        if i in slave_master_dict.keys():
            for master in slave_master_dict[i].keys():
                l = sum(master>np.array(list(slave_master_dict.keys())))
                K[i, master - l] = slave_master_dict[i][master]
        else:
            l = sum(i>np.array(list(slave_master_dict.keys())))
            K[i, i-l] = 1


    vec[L1.owner_range[0]:L1.owner_range[1]] += L1.array
    vec = sum(dolfinx.MPI.comm_world.allgather(np.array(vec, dtype=np.float32)))
    mpc_vec[b_func.owner_range[0]:b_func.owner_range[1]] += b_func.array
    mpc_vec = sum(dolfinx.MPI.comm_world.allgather(np.array(mpc_vec, dtype=np.float32)))
    reduced_L = np.dot(K.T,vec)

    l = 0
    if dolfinx.MPI.comm_world.rank==0:
        print(slave_master_dict)
    for i in range(V.dim()):
        if i in slave_master_dict.keys():
            l+= 1
            continue

        assert(np.isclose(reduced_L[i-l], mpc_vec[i]))
