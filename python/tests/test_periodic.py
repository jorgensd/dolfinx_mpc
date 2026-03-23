from mpi4py import MPI

import basix.ufl
import dolfinx
import numpy as np
import pytest

import dolfinx_mpc


@pytest.mark.parametrize("geometrical", (True, False))
@pytest.mark.parametrize("cell_type", (dolfinx.cpp.mesh.CellType.triangle, dolfinx.cpp.mesh.CellType.quadrilateral))
@pytest.mark.parametrize("ghost_mode", (dolfinx.cpp.mesh.GhostMode.none, dolfinx.cpp.mesh.GhostMode.shared_facet))
def test_periodic_mixed_space(cell_type, ghost_mode, geometrical):
    N = 4
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type=cell_type, ghost_mode=ghost_mode)

    tol = 500 * np.finfo(mesh.geometry.x.dtype).eps

    def left(x):
        return np.isclose(x[0], 1.0, atol=tol, rtol=tol)

    def periodic_map(x):
        out = np.zeros((3, x.shape[1]), dtype=dolfinx.default_scalar_type)
        out[0] = 1 - x[0]
        out[1:] = x[1:]
        return out

    # Create the function space
    cellname = mesh.basix_cell()
    Ve = basix.ufl.element(
        basix.ElementFamily.P, cellname, 2, shape=(mesh.geometry.dim,), dtype=dolfinx.default_real_type
    )
    Qe = basix.ufl.element(basix.ElementFamily.P, cellname, 1, dtype=dolfinx.default_real_type)

    W = dolfinx.fem.functionspace(mesh, basix.ufl.mixed_element([Ve, Qe]))

    if geometrical:
        condition = dolfinx_mpc.MultiPointConstraint(W)
        condition.create_periodic_constraint_geometrical(W.sub(0), left, periodic_map, [], tol=tol)
        condition.finalize()

        condition_sub = dolfinx_mpc.MultiPointConstraint(W)
        condition_sub.create_periodic_constraint_geometrical(W.sub(0).sub(0), left, periodic_map, [], tol=tol)
        condition_sub.create_periodic_constraint_geometrical(W.sub(0).sub(1), left, periodic_map, [], tol=tol)
        condition_sub.finalize()

    else:
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
        num_facets_local = (
            mesh.topology.index_map(mesh.topology.dim - 1).size_local
            + mesh.topology.index_map(mesh.topology.dim - 1).num_ghosts
        )
        marker = np.zeros(num_facets_local, dtype=np.int32)
        marker[dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1, left)] = 1
        ft = dolfinx.mesh.meshtags(mesh, mesh.topology.dim - 1, np.arange(num_facets_local, dtype=np.int32), marker)

        condition = dolfinx_mpc.MultiPointConstraint(W)
        condition.create_periodic_constraint_topological(W.sub(0), ft, 1, periodic_map, [], tol=tol)
        condition.finalize()

        condition_sub = dolfinx_mpc.MultiPointConstraint(W)
        condition_sub.create_periodic_constraint_topological(W.sub(0).sub(0), ft, 1, periodic_map, [], tol=tol)
        condition_sub.create_periodic_constraint_topological(W.sub(0).sub(1), ft, 1, periodic_map, [], tol=tol)
        condition_sub.finalize()

    np.testing.assert_allclose(condition.slaves, condition_sub.slaves)
    np.testing.assert_allclose(condition.masters.array, condition_sub.masters.array)
    np.testing.assert_allclose(condition.masters.offsets, condition_sub.masters.offsets)
    np.testing.assert_allclose(condition.coefficients()[0], condition_sub.coefficients()[0])
    np.testing.assert_allclose(condition.coefficients()[1], condition_sub.coefficients()[1])
