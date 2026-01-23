from mpi4py import MPI

import numpy as np
import pytest
import ufl
from basix.ufl import element
from dolfinx import default_real_type, fem, mesh

import dolfinx_mpc


@pytest.mark.parametrize(
    "cell_type",
    [mesh.CellType.quadrilateral, mesh.CellType.hexahedron, mesh.CellType.tetrahedron, mesh.CellType.triangle],
)
@pytest.mark.parametrize("deg", [1, 2, 4])
@pytest.mark.parametrize("N", [3, 5, 8])
def test_multiple_mpc_spaces_sparsity(cell_type, deg, N):
    """
    Test of sparsity pattern of square matrices with mpcs on two distinct spaces
    """
    cd = mesh.cell_dim(cell_type)
    Ns = np.full(cd, N, dtype=int)
    if cd == 2:
        mesh_constructor = mesh.create_unit_square
    elif cd == 3:
        mesh_constructor = mesh.create_unit_cube
    domain = mesh_constructor(MPI.COMM_WORLD, *Ns, cell_type=cell_type)

    V = fem.functionspace(domain, element("Lagrange", domain.basix_cell(), deg, dtype=default_real_type))
    Q = V.clone()
    atol = 10 * np.finfo(default_real_type).eps

    def periodic_boundary(x):
        return np.logical_or(np.isclose(x[0], 1, atol=atol), np.isclose(x[2], 1, atol=atol))

    def periodic_map(x):
        out = x.copy()
        out[0][np.isclose(x[0], 1).nonzero()] -= 1
        out[domain.geometry.dim - 1][np.isclose(x[2], 1).nonzero()] -= 1
        return out

    mpc_u = dolfinx_mpc.MultiPointConstraint(V)
    mpc_u.create_periodic_constraint_geometrical(V, periodic_boundary, periodic_map, [])
    mpc_u.finalize()

    mpc_p = dolfinx_mpc.MultiPointConstraint(Q)
    mpc_p.create_periodic_constraint_geometrical(Q, periodic_boundary, periodic_map, [])
    mpc_p.finalize()

    # Stokes weak form
    u, p = ufl.TrialFunction(V), ufl.TrialFunction(Q)
    v, q = ufl.TestFunction(V), ufl.TestFunction(Q)

    # Weak statementmpc_u,
    a01 = ufl.inner(p, v) * ufl.dx
    form_01 = fem.form(a01)
    mpcs_01 = [mpc_u, mpc_p]

    p0 = dolfinx_mpc.create_sparsity_pattern(form_01, mpcs_01)
    p0.finalize()

    p1 = dolfinx_mpc.create_sparsity_pattern(form_01, [mpc_u, mpc_u])
    p1.finalize()

    assert p0.num_nonzeros == p1.num_nonzeros

    a10 = ufl.inner(u, q) * ufl.dx
    form_10 = fem.form(a10)

    mpcs_10 = [mpc_p, mpc_u]
    p0 = dolfinx_mpc.create_sparsity_pattern(form_10, mpcs_10)
    p0.finalize()

    p1 = dolfinx_mpc.create_sparsity_pattern(form_10, [mpc_u, mpc_u])
    p1.finalize()

    assert p0.num_nonzeros == p1.num_nonzeros
