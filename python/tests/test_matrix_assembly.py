import dolfinx
import dolfinx_mpc
import numba
from numba.typed import List
import numpy as np
import ufl

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

def test_mpc_assembly():

    # Create mesh and function space
    mesh = dolfinx.UnitSquareMesh(dolfinx.MPI.comm_world, 5,3)
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

    # Test against generated code and general assembler
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx

    # Generating normal matrix for comparison
    A1 = dolfinx.fem.assemble_matrix(a)
    A1.assemble()

    mpc = dolfinx_mpc.MultiPointConstraint(V, slave_master_dict)
    slave_cells = np.array(mpc.slave_cells())

    # Add mpc non-zeros to sparisty pattern#
    A2 = mpc.generate_petsc_matrix(dolfinx.Form(a)._cpp_object)

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


    # No dirichlet condition as this is just assembly
    bcs = np.array([])


    # Assemble using generated tabulate_tensor kernel and Numba assembler
    ufc_form = dolfinx.jit.ffcx_jit(a)
    kernel = ufc_form.create_cell_integral(-1).tabulate_tensor
    for i in range(2):
        A2.zeroEntries()
        dolfinx_mpc.assemble_matrix_mpc(A2.handle, kernel, (c, pos), geom, dofs,
                            (slaves, masters, coefficients, offsets, sc_nb, c2s_nb, c2so_nb), ghost_info,bcs)

        A2.assemble()

    # Transfer A2 to numpy matrix for comparison with globally built matrix
    A_mpc_np = np.zeros((V.dim(), V.dim()))
    for i in range(A2.getOwnershipRange()[0], A2.getOwnershipRange()[1]):
        cols, vals = A2.getRow(i)
        for col, val in zip(cols, vals):
            A_mpc_np[i,col] = val
    A_mpc_np = sum(dolfinx.MPI.comm_world.allgather(A_mpc_np))

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


    # Transfer original matrix to numpy
    A_global = np.zeros((V.dim(), V.dim()))
    for i in range(A1.getOwnershipRange()[0], A1.getOwnershipRange()[1]):
        cols, vals = A1.getRow(i)
        for col, val in zip(cols, vals):
            A_global[i,col] = val
    A_global = sum(dolfinx.MPI.comm_world.allgather(A_global))

    # Compute globally reduced system and compare norms with A2
    reduced_A = np.matmul(np.matmul(K.T,A_global),K)
    assert(np.isclose(np.linalg.norm(reduced_A), A2.norm()))

    # Pad globally reduced system with 0 rows and columns for each slave entry
    A_numpy_padded = np.zeros((V.dim(),V.dim()))
    l = 0
    for i in range(V.dim()):
        if i in slave_master_dict.keys():
            l += 1
            continue
        m = 0
        for j in range(V.dim()):
            if j in slave_master_dict.keys():
                m += 1
                continue
            else:
                A_numpy_padded[i, j] = reduced_A[i-l, j-m]

    # Check that all entities are close
    assert np.allclose(A_mpc_np, A_numpy_padded)
