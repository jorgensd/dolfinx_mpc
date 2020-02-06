import dolfinx
import dolfinx_mpc
from numba.typed import List
import numpy as np
import ufl
import time

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
def bc0(x):
    return np.logical_and(np.isclose(x[:,1], 0),np.isclose(x[:,1], 0))

def bc_value(bcs, x):
    vals = np.zeros(len(bcs))
    for i in range(len(bcs)):
        if bcs[i]:
            vals[i] = 0
    return vals

def test_mpc_assembly():

    # Create mesh and function space
    mesh = dolfinx.UnitSquareMesh(dolfinx.MPI.comm_world, 3, 3)
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
    bcs = np.array([])#bc0(dof_coords)
    values = np.array([]) # values = bc_value(bcs, dof_coords)
    # Test against generated code and general assembler
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    f = ufl.sin(2*ufl.pi*x[0])*ufl.sin(ufl.pi*x[1])
    a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx
    l = ufl.inner(f, v)*ufl.dx

    # Timing assembly of matrix with C++
    start = time.time()
    A1 = dolfinx.fem.assemble_matrix(a)
    # end = time.time()
    A1.assemble()

    # print()
    # print("--"*10)
    # print("Matrix Time (C++, pass 1):", end - start)
    # start = time.time()
    # L1 = dolfinx.fem.assemble_vector(l)
    # print(L1.array)
    # end = time.time()
    # print("Vector Time (C++, pass 1):", end - start)
    # print("--"*10)
    # second pass in C++
    # A1.zeroEntries()
    # start = time.time()
    # dolfinx.fem.assemble_matrix(A1, a)
    # end = time.time()
    # A1.assemble()
    # print("Matrix Time (C++, pass 2):", end - start)
    # start = time.time()
    L1 = dolfinx.fem.assemble_vector(l)
    end = time.time()
    # print("Vector Time (C++, pass 1):", end - start)


    # slave_master_dict = {3:{0:0.3,2:0.5}}
    # slave_master_dict = {0:{0:1}}
    # slave_master_dict = {1:{0:0.1},3:{0:0.3,2:0.73}}
    # slave_master_dict = {1:{7:0.1}}


    mpc = dolfinx_mpc.MultiPointConstraint(V, slave_master_dict)
    slave_cells = np.array(mpc.slave_cells())
    # pattern = dolfinx.cpp.fem.create_sparsity_pattern(dolfinx.Form(a)._cpp_object)
    #pattern.info_statistics()
    #print("-----"*10)
    # dolfinx.cpp.fem.create_matrix(dolfinx.Form(a)._cpp_object)
    # pattern.info_statistics()
    #print("-"*25)
    # Add mpc non-zeros to sparisty pattern#
    print(slave_master_dict)

    A2 = mpc.generate_petsc_matrix(dolfinx.Form(a)._cpp_object)
    A2.zeroEntries()
    A1.assemble()

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

    # Assemble using generated tabulate_tensor kernel and Numba assembler
    ufc_form = dolfinx.jit.ffcx_jit(a)
    kernel = ufc_form.create_cell_integral(-1).tabulate_tensor
    for i in range(2):
        A2.zeroEntries()
        start = time.time()
        dolfinx_mpc.assemble_matrix_mpc(A2.handle, kernel, (c, pos), geom, dofs,
                            (slaves, masters, coefficients, offsets, sc_nb, c2s_nb, c2so_nb), ghost_info,bcs)
        end = time.time()
        print("Matrix Time (numba/cffi, pass {}): {}".format(i, end - start))

        A2.assemble()


    # A2.view()
    assert(A2.norm() == 11.34150823303497)
    return
    print(A2.norm())
    # K = np.array([[1,0,0,0,0,0,0,0],
    #                [0,1,0,0,0,0,0,0],
    #                [0,0,1,0,0,0,0,0],
    #                [0,0,0,1,0,0,0,0],
    #                [0,0,0,0,1,0,0,0],
    #                [0,0,0,0,0,1,0,0],
    #                [0,0,0,0,0,beta,0,0],
    #                [0,0,0,0,0,0,1,0],
    #                [0,0,0,0,0,0,0,1]])
    # global_A = np.matmul(np.matmul(K.T,A1[:,:]),K)
    # print(global_A)
    # print(A2[:,:])
    # for local, glob in zip([0,1,2,3,4,5,7,8],[0,1,2,3,4,5,6,7]):
    #     for local2, glob2 in zip([0,1,2,3,4,5,7,8],[0,1,2,3,4,5,6,7]):
    #         assert(A2[local,local2] == global_A[glob, glob2])
    # # print(reduced)
    # assert np.max(d-reduced) <= 1e-6
    # print(A1.view())
    # print(A2.view())
