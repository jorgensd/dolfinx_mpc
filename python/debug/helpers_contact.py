import numpy as np
from mpi4py import MPI
import dolfinx.geometry as geometry
import dolfinx.cpp
import dolfinx_mpc.cpp


def compute_masters_local_block(V, local_slaves, local_masters, n_vec):
    """
    Compute master contributions from the other sub-spaces (0,tdim-1).
    All of these masters will be local for to the proc
    """
    masters_block = {}
    for i, slave in enumerate(local_slaves):
        for j in range(len(local_masters)):
            coeff = -n_vec[local_masters[j][i]]/n_vec[slave]
            if not np.isclose(coeff, 0):
                if slave in masters_block.keys():
                    masters_block[slave]["masters"].append(local_masters[j][i])
                    masters_block[slave]["coeffs"].append(coeff)
                    masters_block[slave]["owners"].append(MPI.COMM_WORLD.rank)
                else:
                    masters_block[slave] = {"masters": [local_masters[j][i]],
                                            "coeffs": [coeff],
                                            "owners": [MPI.COMM_WORLD.rank]}

    return masters_block


def compute_masters_local(V, loc_slaves, loc_coords,
                          n_vec, local_slave_to_cell):
    """
    Compute coefficient for each local slave with local masters
    (other side of the interface)
    """
    tdim = V.mesh.topology.dim
    bs = V.dofmap.index_map.block_size
    local_size = V.dofmap.index_map.size_local*bs
    ghost_owners = V.dofmap.index_map.ghost_owner_rank()
    masters_local = {}
    for i, slave in enumerate(loc_slaves):
        # Compute normal vector for local slave
        b_off = slave % bs
        block_dofs = np.array([slave-b_off+i for i in range(bs)])
        n = list(n_vec[block_dofs])

        # Extract cell and compute basis values
        if i in local_slave_to_cell.keys():
            cell = local_slave_to_cell[i]
        else:
            continue
        coord = loc_coords[i]
        basis_values = dolfinx_mpc.cpp.mpc.\
            get_basis_functions(V._cpp_object,
                                coord, cell)
        cell_dofs = V.dofmap.cell_dofs(cell)
        masters_ = []
        owners_ = []
        coeffs_ = []
        for k in range(tdim):
            for local_idx, dof in enumerate(cell_dofs):
                coeff = basis_values[local_idx, k]
                coeff *= (n[k] / n[tdim-1])
                if not np.isclose(coeff, 0):
                    if dof < local_size:
                        owners_.append(MPI.COMM_WORLD.rank)
                    else:
                        # If master i ghost, find its owner
                        owners_.append(
                            ghost_owners[dof//bs -
                                         local_size//bs])
                    masters_.append(dof)
                    coeffs_.append(coeff)
        if len(masters_) > 0:
            masters_local[slave] = {"masters": masters_,
                                    "coeffs": coeffs_,
                                    "owners": owners_}
    return masters_local


def locate_dofs_colliding_with_cells(mesh, dofs, coords, cells, tree):
    """
    Function returning dictionary of which dofs that are colliding with
    a set of cells
    """
    tdim = mesh.topology.dim
    cell_map = mesh.topology.index_map(tdim)
    [cmin, cmax] = cell_map.local_range
    l2g_cells = np.array(cell_map.global_indices(False), dtype=np.int64)
    dof_to_cell = {}
    for i, (dof, coord) in enumerate(zip(dofs, coords)):
        possible_collisions = np.array(
            geometry.compute_collisions_point(tree, coord.T))
        if len(possible_collisions) > 0:
            # Check if possible cells are in local range and if
            # they are in the supplied list of local cells
            contains_cells = np.isin(possible_collisions, cells)
            is_owned = np.logical_and(cmin <= l2g_cells[possible_collisions],
                                      l2g_cells[possible_collisions] < cmax)
            candidates = possible_collisions[np.logical_and(
                is_owned, contains_cells)]
            # Check if actual cell in mesh is within a distance of 1e-14
            cell = dolfinx.cpp.geometry.select_colliding_cells(
                mesh, list(candidates), coord.T, 1)
            if len(cell) > 0:
                dof_to_cell[i] = cell[0]
    return dof_to_cell


def compute_masters_from_global(V, slaves_glob, coords_glob, normals_glob,
                                owners, unique_master_cells, tree):
    """
    Given the set of cells that can have a master dof, locate the actual
    masters that are close to slave dofs from another processor, and
    compute the corresponding coefficients and owners
    """
    tdim = V.mesh.topology.dim
    ghost_owners = V.dofmap.index_map.ghost_owner_rank()
    global_indices = V.dofmap.index_map.global_indices(False)
    bs = V.dofmap.index_map.block_size
    local_size = V.dofmap.index_map.size_local*bs
    # Find local cells that can possibly contain masters
    slaves_array = np.array(slaves_glob).reshape((len(slaves_glob), 1))
    cells_with_masters_for_global_slave = locate_dofs_colliding_with_cells(
        V.mesh, slaves_array, coords_glob, unique_master_cells, tree)

    masters_for_slaves = {}
    # Loop through slave dofs and compute local master contributions
    for index in cells_with_masters_for_global_slave.keys():
        cell = cells_with_masters_for_global_slave[index]
        coord = coords_glob[index]
        basis_values = dolfinx_mpc.cpp.mpc.\
            get_basis_functions(V._cpp_object,
                                coord, cell)
        cell_dofs = V.dofmap.cell_dofs(cell)
        masters_ = []
        owners_ = []
        coeffs_ = []
        for k in range(tdim):
            for local_idx, dof in enumerate(cell_dofs):
                coeff = basis_values[local_idx, k]
                coeff *= (normals_glob[index][k] /
                          normals_glob[index][tdim-1])
                if not np.isclose(coeff, 0):
                    if dof < local_size:
                        owners_.append(MPI.COMM_WORLD.rank)
                    else:
                        # If master i ghost, find its owner
                        owners_.append(
                            ghost_owners[dof//bs -
                                         local_size//bs])
                    masters_.append(global_indices[dof])
                    coeffs_.append(coeff)
        if owners[index] in masters_for_slaves.keys():
            masters_for_slaves[owners[index]
                               ][slaves_glob[index]] = {"masters": masters_,
                                                        "coeffs": coeffs_,
                                                        "owners": owners_}

        else:
            masters_for_slaves[owners[index]] = {slaves_glob[index]:
                                                 {"masters": masters_,
                                                  "coeffs": coeffs_,
                                                  "owners": owners_}}
    return masters_for_slaves


def recv_masters(loc_slaves_flat, glob_slaves, possible_recv, tag):
    """
    Receive masters from other processors (those listed in possible_recv),
     and sort them by slave.
    If duplicates, add both of them.
    Also returns the processors where data is actually received.
    """
    narrow_master_recv = []
    global_masters = {}
    for proc in possible_recv:
        # Receive info from other proc and only do something if
        # there is data
        data = MPI.COMM_WORLD.recv(source=proc, tag=tag)
        if len(data.keys()) > 0:
            narrow_master_recv.append(proc)
            # print("{0:d} received masters for {2:d} slaves from {1:d}".format(
            #     MPI.COMM_WORLD.rank, proc, len(data.keys())))
            # Loop through the slaves which sending proc has masters for
            for global_slave in data.keys():
                idx = np.where(glob_slaves == global_slave)[0][0]
                if loc_slaves_flat[idx] in global_masters.keys():
                    global_masters[loc_slaves_flat[idx]
                                   ][proc] = data[global_slave]
                else:
                    global_masters[loc_slaves_flat[idx]] = {
                        proc: data[global_slave]}
    return global_masters, narrow_master_recv


def send_masters(masters_for_slaves, owners, tag):
    """
    Sends masters from local processor to the processor
    owning the slave (Those in owners).
    Returns which processors actually received data.
    """
    # Communicators that might later have to receive from master
    narrow_slave_send = []
    # Send master info to slave processor
    for proc in set(owners):
        if proc in masters_for_slaves.keys():
            MPI.COMM_WORLD.send(masters_for_slaves[proc],
                                dest=proc, tag=tag)
            narrow_slave_send.append(proc)
        else:
            MPI.COMM_WORLD.send({}, dest=proc, tag=tag)
    return narrow_slave_send


def select_masters(global_masters):
    """
    Select masters from from receiving processors, using a random
    selection if multiple procs claim the master
    """
    masters = {}
    if len(global_masters.keys()) > 0:
        for local_slave in global_masters.keys():
            # Check if multiple masters claim ownership to slave
            if len(global_masters[local_slave].keys()) == 1:
                key = next(iter(global_masters[local_slave]))
                masters[local_slave] = global_masters[local_slave][key]
            else:
                # Choose randomly which we select
                selected_proc = np.random.choice(np.fromiter(
                    global_masters[local_slave].keys(), dtype=np.int32))
                masters[local_slave] = (global_masters[local_slave]
                                        [selected_proc])
    return masters


def recv_bb_collisions(tag):
    """
    Receive bb collisions form other proc. These processors are potential
    candidates to contain masters
    """
    slaves_glob = []
    coords_glob = None
    normals_glob = None
    owners = np.ones(0, dtype=np.int32)
    for i in range(MPI.COMM_WORLD.size):
        if i != MPI.COMM_WORLD.rank:
            data = MPI.COMM_WORLD.recv(source=i, tag=tag)
            if len(data) > 0:
                slaves_glob.extend(data["slaves"])
                owners = np.concatenate(
                    (owners, np.full(len(data["slaves"]), i,
                                     dtype=np.int32)), axis=0)
                if coords_glob is None:
                    coords_glob = data["coords"]
                else:
                    coords_glob = np.vstack([coords_glob, data["coords"]])
                if normals_glob is None:
                    normals_glob = data["normals"]
                else:
                    normals_glob = np.vstack([normals_glob, data["normals"]])
    return slaves_glob, owners, coords_glob, normals_glob


def send_bb_collisions(V, loc_slaves, loc_coords, n_vec, tag):
    """
    Find all processors where a slave can be located
    according to the bounding box tree. Send information
    about these slaves to those processors.
    Returns the processors that got sent info
    (that is not) zero.
    """
    mesh = V.mesh
    fdim = V.mesh.topology.dim - 1
    loc_to_glob = np.array(V.dofmap.index_map.global_indices(False),
                           dtype=np.int64)
    bs = V.dofmap.index_map.block_size
    facettree = geometry.BoundingBoxTree(mesh, dim=fdim)
    slaves_to_send = {}
    slave_coords_to_send = {}
    normals_to_send = {}

    for dof, point in zip(loc_slaves, loc_coords):
        # Zip coordinates for slave a 3D array
        b_off = dof % bs
        block_dofs = np.array([dof-b_off+i for i in range(bs)])
        n = list(n_vec[block_dofs].T[0])

        # Get processor number for a point
        procs = dolfinx_mpc.cpp.mpc.compute_process_collisions(
            facettree._cpp_object, point.T)
        for proc in procs:
            if proc != MPI.COMM_WORLD.rank:
                if proc not in slaves_to_send.keys():
                    slaves_to_send[proc] = [loc_to_glob[dof[0]]]
                else:
                    slaves_to_send[proc].append(loc_to_glob[dof[0]])
                if proc not in slave_coords_to_send.keys():
                    slave_coords_to_send[proc] = [point[0]]
                else:
                    slave_coords_to_send[proc].append(point[0])
                if proc not in normals_to_send.keys():
                    normals_to_send[proc] = [n]
                else:
                    normals_to_send[proc].append(n)

    # Send information about possible collisions with
    # slave to the other processors
    possible_recv = []
    for proc in range(MPI.COMM_WORLD.size):
        if proc in slaves_to_send.keys():
            MPI.COMM_WORLD.send({"slaves": slaves_to_send[proc],
                                 "coords": slave_coords_to_send[proc],
                                 "normals": normals_to_send[proc]},
                                dest=proc, tag=tag)
            # Cache proc for later receive
            possible_recv.append(proc)
        elif proc != MPI.COMM_WORLD.rank:
            MPI.COMM_WORLD.send({}, dest=proc, tag=tag)
    return possible_recv


def gather_masters_for_local_slaves(local_slaves, loc_to_glob,
                                    block_masters, loc_masters, glob_master):
    # Gather masters for each slave owned by the processor
    # These can be:
    # 1. Masters from slave block
    # 2. Masters from same proc or Masters from other proc
    # Always chose same proc over other proc
    masters_for_all_local = []
    coeffs_for_all_local = []
    owners_for_all_local = []
    offsets_for_all_local = [0]
    for dof in local_slaves:
        # Added blocked dofs
        if dof in block_masters.keys():
            # Map to global
            masters_for_all_local.extend(
                loc_to_glob[block_masters[dof]["masters"]])
            coeffs_for_all_local.extend(block_masters[dof]["coeffs"])
            owners_for_all_local.extend(block_masters[dof]["owners"])

        if (dof in glob_master.keys()
                or dof in loc_masters.keys()):
            if dof in loc_masters.keys():
                # Always chose local masters over other processor
                masters_for_all_local.extend(
                    loc_to_glob[loc_masters[dof]["masters"]])
                coeffs_for_all_local.extend(loc_masters[dof]["coeffs"])
                owners_for_all_local.extend(loc_masters[dof]["owners"])
            elif dof in glob_master.keys():
                masters_for_all_local.extend(
                    glob_master[dof]["masters"])
                coeffs_for_all_local.extend(glob_master[dof]["coeffs"])
                owners_for_all_local.extend(glob_master[dof]["owners"])
        offsets_for_all_local.append(len(masters_for_all_local))
    return (masters_for_all_local, coeffs_for_all_local,
            owners_for_all_local, offsets_for_all_local)
