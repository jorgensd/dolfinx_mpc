from IPython import embed
import dolfinx
import dolfinx.geometry
import dolfinx.fem
import dolfinx_mpc
import numpy as np
from mpi4py import MPI
# mesh = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, 1, 1, 1)
mesh = dolfinx.BoxMesh(MPI.COMM_WORLD, [np.array([0.0, 0.0, 0.0]),
                                        np.array([1.0, 1.0, 1.0])],
                       [1, 1, 1])
V = dolfinx.VectorFunctionSpace(mesh, ("Lagrange", 1))
x_coords = V.tabulate_dof_coordinates()
print(x_coords)


def tab_dof_coords(V):
    index_map = V.dofmap.index_map
    bs = index_map.block_size
    local_size = bs * (index_map.size_local + index_map.num_ghosts)
    X = V.element.dof_reference_coordinates()
    x_dofmap = mesh.geometry.dofmap
    num_dofs_g = len(x_dofmap.links(0))

    x_g = mesh.geometry.x
    x = np.zeros((local_size, 3))

    coordinates = np.zeros((V.element.space_dimension(), mesh.geometry.dim))
    coordinate_dofs = np.zeros((num_dofs_g, mesh.geometry.dim))

    i_map = mesh.topology.index_map(mesh.topology.dim)
    num_cells = i_map.size_local + i_map.num_ghosts
    print(num_cells)
    for c in range(num_cells):
        x_dofs = x_dofmap.links(c)
        for i in range(num_dofs_g):
            coordinate_dofs[i] = x_g[x_dofs[i]][: mesh.geometry.dim]
        dofs = V.dofmap.cell_dofs(c)
        # Manual implementation
        from IPython import embed
        embed()
        #    mesh.geometry.cmap.push_forward(coordinates, X, coordinate_dofs)

        for i in range(len(dofs)):
            x[dofs[i], : mesh.geometry.dim] = coordinates[i]
    return x


coordinate_index = -1
for i, coord in enumerate(x_coords):
    if np.allclose(coord, [0, 1, 0]):
        coordinate_index = i
        break
assert coordinate_index != -1


print(x_coords[coordinate_index])

x2 = tab_dof_coords(V)

embed()
