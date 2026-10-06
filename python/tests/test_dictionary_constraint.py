from mpi4py import MPI

import dolfinx
import numpy as np
import pytest

from dolfinx_mpc.dictcondition import create_dictionary_constraint


@pytest.mark.parametrize("cell_type", (dolfinx.mesh.CellType.hexahedron, dolfinx.mesh.CellType.tetrahedron))
def test_ghost_slaves_get_all_masters(cell_type):
    """Every copy of a slave, owned or ghost, must receive the same, fully resolved masters.

    A master owned by a process on which the slave is only a ghost has to reach the ghost copies
    of that slave on the other processes as well.
    """
    comm = MPI.COMM_WORLD
    N = 6
    mesh = dolfinx.mesh.create_unit_cube(comm, N, N, N, cell_type=cell_type)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))
    xdt = mesh.geometry.x.dtype

    # Periodic relation u(x) = u(x - s), s_i = 1 if x_i = 1, on the faces x_i = 1 (the corner (1, 1, 1) excluded)
    x = V.tabulate_dof_coordinates()
    on_faces = np.isclose(x, 1.0).any(axis=1) & ~np.isclose(x, 1.0).all(axis=1)
    slave_points = np.unique(np.round(np.vstack(comm.allgather(x[on_faces])), 14), axis=0)
    slave_master_dict = {
        np.asarray(p, dtype=xdt).tobytes(): {np.where(np.isclose(p, 1.0), 0.0, p).astype(xdt).tobytes(): 1.0}
        for p in slave_points
    }

    slaves, masters, _, owners, offsets = create_dictionary_constraint(V, slave_master_dict, 0, 0)

    # All masters are resolved
    assert comm.allreduce(int(np.sum(masters < 0)) + int(np.sum(owners < 0)), op=MPI.SUM) == 0

    # Owned and ghost copies of each slave have the same masters
    bs = V.dofmap.index_map_bs
    blocks = V.dofmap.index_map.local_to_global(slaves // bs)
    global_slaves = blocks * bs + slaves % bs
    local = {int(s): tuple(sorted(masters[offsets[i] : offsets[i + 1]])) for i, s in enumerate(global_slaves)}
    reference = {}
    for part in comm.allgather(local):
        for s, m in part.items():
            assert reference.setdefault(s, m) == m
