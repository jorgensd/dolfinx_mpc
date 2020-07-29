import typing

import dolfinx_mpc.cpp
import numpy as np
from petsc4py import PETSc
import dolfinx.fem as fem
import dolfinx.function as function
import dolfinx.geometry as geometry
from mpi4py import MPI


def close_to(point):
    """
    Convenience function for locating a point [x,y,z]
    within an array x [[x0,...,xN],[y0,...,yN], [z0,...,zN]].
    If the point is not 3D it is padded with zeros
    """
    return lambda x: np.isclose(x, point).all(axis=0)


def create_dictionary_constraint(V: function.FunctionSpace, slave_master_dict:
                                 typing.Dict[bytes, typing.Dict[bytes, float]],
                                 subspace_slave=None,
                                 subspace_master=None
                                 ):
    """
    Returns a multi point constraint for a given function space
    and dictionary constraint.
    Input:
        V - The function space
        slave_master_dict - The dictionary.
        subspace_slave - If using mixed or vector space, 
                         and only want to use dofs from 
                         a sub space as slave add index here
        subspace_master - Subspace index for mixed or vector spaces
    Example:
        If the dof D located at [d0,d1] should be constrained to the dofs E and
        F at [e0,e1] and [f0,f1] as
        D = alpha E + beta F
        the dictionary should be:
            {np.array([d0, d1], dtype=np.float64).tobytes():
                 {np.array([e0, e1], dtype=np.float64).tobytes(): alpha,
                  np.array([f0, f1], dtype=np.float64).tobytes(): beta}}
    """
    dfloat = np.float64
    comm = V.mesh.mpi_comm()
    mesh = V.mesh
    bs = V.dofmap.index_map.block_size
    local_size = V.dofmap.index_map.size_local*bs
    loc_to_glob = V.dofmap.index_map.global_indices(False)
    slaves_dict = {}
    tree = geometry.BoundingBoxTree(mesh, dim=mesh.topology.dim)
    masters_to_send = {}
    recv_from_procs = []
    for i, slave_point in enumerate(slave_master_dict.keys()):
        slave_point_nd = np.zeros((3, 1), dtype=dfloat)
        for j, coord in enumerate(np.frombuffer(slave_point,
                                                dtype=dfloat)):
            slave_point_nd[j] = coord

        slaves_dict[i] = {"num_local_masters": 0,
                          "slave": None,
                          "masters": np.full(len(
                              list(slave_master_dict[slave_point].keys())),
                              -1, dtype=np.int64),
                          "coeffs": np.zeros(len(
                              list(slave_master_dict[slave_point].keys())),
                              dtype=PETSc.ScalarType),
                          "owners": np.full(len(
                              list(slave_master_dict[slave_point].keys())),
                              -1, dtype=np.int64),
                          "local_master_index": []}
        if subspace_slave is None:
            slave_dofs = np.array(fem.locate_dofs_geometrical(
                V, close_to(slave_point_nd))[:, 0])
        else:
            Vsub = V.sub(subspace_slave).collapse()
            slave_dofs = np.array(fem.locate_dofs_geometrical(
                (V.sub(subspace_slave), Vsub), close_to(slave_point_nd))[:, 0])
        if len(slave_dofs) == 1:
            slaves_dict[i]["slave"] = slave_dofs[0]
            # Decide if slave is ghost or not
            if slave_dofs[0] < local_size:
                slaves_dict[i]["ghost"] = False
            else:
                slaves_dict[i]["ghost"] = True
        elif len(slave_dofs) > 1:
            raise RuntimeError("Multiple slaves found at same point. " +
                               "You should use sub-space locators.")
        # Wrap as list to ensure order later
        master_points = list(slave_master_dict[slave_point].keys())
        master_points_nd = np.zeros((
            3, len(master_points)), dtype=dfloat)
        for (j, master_point) in enumerate(master_points):
            # Wrap bytes as numpy array
            for k, coord in enumerate(np.frombuffer(master_point,
                                                    dtype=dfloat)):
                master_points_nd[k, j] = coord
            if subspace_master is None:
                master_dofs = fem.locate_dofs_geometrical(
                    V, close_to(master_points_nd[:, j:j+1]))[:, 0]
            else:
                Vsub = V.sub(subspace_master).collapse()
                master_dofs = np.array(fem.locate_dofs_geometrical(
                    (V.sub(subspace_master), Vsub),
                    close_to(master_points_nd[:, j:j+1]))[:, 0])

            # Only add masters owned by this processor
            master_dofs = master_dofs[master_dofs < local_size]
            if len(master_dofs) == 1:
                slaves_dict[i]["masters"][j] = loc_to_glob[master_dofs[0]]
                slaves_dict[i]["coeffs"][j] = \
                    slave_master_dict[slave_point][master_point]
                slaves_dict[i]["owners"][j] = comm.rank
                slaves_dict[i]["num_local_masters"] += 1
                slaves_dict[i]["local_master_index"].append(j)
            elif len(master_dofs) > 1:
                raise RuntimeError("Multiple masters found at same point. "
                                   + "You should use sub-space locators.")
        # If we got a master, but not the slave
        procs_with_slave = np.array(
            dolfinx_mpc.cpp.mpc.compute_process_collisions(
                tree._cpp_object,
                np.frombuffer(slave_point_nd, dtype=dfloat)),
            dtype=np.int32)
        # Sort slaves by receving processors
        for proc in procs_with_slave[procs_with_slave !=
                                     comm.rank]:
            comm.send(slaves_dict[i], dest=proc, tag=i)

        # If processor do not own all masters, recv until all are added
        if (slaves_dict[i]["slave"] is not None
            and not slaves_dict[i]["ghost"]
            and len(slaves_dict[i]["masters"]) !=
                slaves_dict[i]["num_local_masters"]):
            while (slaves_dict[i]["num_local_masters"]
                   < len(slaves_dict[i]["masters"])):
                # status = MPI.Status()
                data = comm.recv(source=MPI.ANY_SOURCE, tag=i)
                # If we got the slave (and it is owned locally)
                if (slaves_dict[i]["slave"] is not None
                    and not slaves_dict[i]["ghost"]
                    and len(slaves_dict[i]["masters"]) !=
                        slaves_dict[i]["num_local_masters"]):
                    # Add each local index to the local slave array
                    for master_index in data["local_master_index"]:
                        slaves_dict[i]["masters"][master_index] = \
                            data["masters"][master_index]
                        slaves_dict[i]["coeffs"][master_index] = \
                            data["coeffs"][master_index]
                        slaves_dict[i]["owners"][master_index] =  \
                            data["owners"][master_index]
                        slaves_dict[i]["num_local_masters"] += 1
    # Flatten slave data for slaves that are local, leave ghost data
    slaves, masters, coeffs, owners, offsets = [], [], [], [], [0]
    ghost_slaves = {}
    for slave_index in slaves_dict.keys():
        if slaves_dict[slave_index]["slave"] is not None:
            if slaves_dict[slave_index]["ghost"]:
                key = slaves_dict[slave_index]["slave"]
                ghost_slaves[loc_to_glob[key]] = slaves_dict[slave_index]
                ghost_slaves[loc_to_glob[key]]["local_slave"] = key
                ghost_slaves[loc_to_glob[key]].pop("slave")
                ghost_slaves[loc_to_glob[key]].pop("ghost")

            else:
                slaves.append(slaves_dict[slave_index]["slave"])
                masters.extend(slaves_dict[slave_index]["masters"])
                owners.extend(slaves_dict[slave_index]["owners"])
                coeffs.extend(slaves_dict[slave_index]["coeffs"])
                offsets.append(len(masters))
    del slaves_dict
    num_owned_slaves = len(slaves)
    shared_indices = dolfinx_mpc.cpp.mpc.compute_shared_indices(
        V._cpp_object)
    # Send masters to processors that have the slave as a ghost
    send_ghost_masters = {}
    for i, slave in enumerate(slaves):
        block = slave // bs
        if block in shared_indices.keys():
            for proc in shared_indices[block]:
                if proc in send_ghost_masters.keys():
                    send_ghost_masters[proc][loc_to_glob[slave]] = {
                        "masters": masters[offsets[i]:offsets[i+1]],
                        "coeffs": coeffs[offsets[i]:offsets[i+1]],
                        "owners": owners[offsets[i]:offsets[i+1]]
                    }
                else:
                    send_ghost_masters[proc] = {loc_to_glob[slave]: {
                        "masters": masters[offsets[i]:offsets[i+1]],
                        "coeffs": coeffs[offsets[i]:offsets[i+1]],
                        "owners": owners[offsets[i]:offsets[i+1]]}}
        del block
    for proc in send_ghost_masters.keys():
        comm.send(send_ghost_masters[proc], dest=proc, tag=3)
    del send_ghost_masters

    # For ghosted slaves, find their respective owner
    local_blocks_size = V.dofmap.index_map.size_local
    ghost_owners = V.dofmap.index_map.ghost_owner_rank()
    ghost_recv = []
    for slave in ghost_slaves.keys():
        block = ghost_slaves[slave]["local_slave"]//bs - local_blocks_size
        owner = ghost_owners[block]
        ghost_recv.append(owner)
        del block, owner
    ghost_recv = set(ghost_recv)

    # Receive masters for ghosted slaves
    for owner in ghost_recv:
        proc_masters = comm.recv(source=owner, tag=3)
        for global_slave in proc_masters.keys():
            local_slave = ghost_slaves[global_slave]["local_slave"]
            slaves.append(local_slave)
            masters.extend(proc_masters[global_slave]["masters"])
            coeffs.extend(proc_masters[global_slave]["coeffs"])
            owners.extend(proc_masters[global_slave]["owners"])
            offsets.append(len(masters))
        del proc_masters
    cc = dolfinx_mpc.cpp.mpc.ContactConstraint(
        V._cpp_object, slaves, num_owned_slaves)
    cc.add_masters(masters, coeffs, owners, offsets)
    return cc
