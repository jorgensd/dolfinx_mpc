import pytest

import dolfinx
import dolfinx_mpc
import dolfinx_mpc.utils
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc


@pytest.mark.parametrize("cell_type",
                         (dolfinx.cpp.mesh.CellType.triangle,
                          dolfinx.cpp.mesh.CellType.quadrilateral))
@pytest.mark.parametrize("ghost_mode",
                         (dolfinx.cpp.mesh.GhostMode.none,
                          dolfinx.cpp.mesh.GhostMode.shared_facet))
def test_mixed_element(cell_type, ghost_mode):
    N = 4
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, N, N,
                                  cell_type=cell_type, ghost_mode=ghost_mode)

    # Rotate the mesh to induce more interesting slip BCs
    th = np.pi / 4.0
    rot = np.array([[np.cos(th), -np.sin(th)],
                    [np.sin(th), np.cos(th)]])
    gdim = mesh.geometry.dim
    mesh.geometry.x[:, :gdim] = (rot @ mesh.geometry.x[:, :gdim].T).T

    # Create the function space
    Ve = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2)
    Qe = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    V = dolfinx.FunctionSpace(mesh, Ve)
    Q = dolfinx.FunctionSpace(mesh, Qe)
    W = dolfinx.FunctionSpace(mesh, Ve * Qe)

    # Inlet velocity Dirichlet BC
    bc_facets = dolfinx.mesh.locate_entities_boundary(
        mesh, mesh.topology.dim - 1, lambda x: np.isclose(x[0], 0.0))
    other_facets = dolfinx.mesh.locate_entities_boundary(
        mesh, mesh.topology.dim - 1, lambda x: np.isclose(x[0], 1.0))

    mt = dolfinx.MeshTags(mesh, mesh.topology.dim - 1,
                          other_facets, np.full_like(other_facets, 1))

    inlet_velocity = dolfinx.Function(V)
    inlet_velocity.interpolate(
        lambda x: np.zeros((mesh.geometry.dim, x[0].shape[0]), dtype=np.double))
    inlet_velocity.x.scatter_forward()

    # -- Nested assembly
    dofs = dolfinx.fem.locate_dofs_topological(V, 1, bc_facets)
    bc1 = dolfinx.DirichletBC(inlet_velocity, dofs)

    # Collect Dirichlet boundary conditions
    bcs = [bc1]

    mpc_v = dolfinx_mpc.MultiPointConstraint(V)
    n_approx = dolfinx_mpc.utils.create_normal_approximation(
        V, mt.indices[mt.values == 1])
    mpc_v.create_slip_constraint((mt, 1), n_approx, bcs=bcs)
    mpc_v.finalize()

    mpc_q = dolfinx_mpc.MultiPointConstraint(Q)
    mpc_q.finalize()

    f = dolfinx.Constant(mesh, ((0, 0)))
    (u, p) = ufl.TrialFunction(V), ufl.TrialFunction(Q)
    (v, q) = ufl.TestFunction(V), ufl.TestFunction(Q)
    a00 = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    a01 = - ufl.inner(p, ufl.div(v)) * ufl.dx
    a10 = - ufl.inner(ufl.div(u), q) * ufl.dx
    a11 = None

    L0 = ufl.inner(f, v) * ufl.dx
    L1 = dolfinx.Constant(mesh, 0.0) * q * ufl.dx

    n = ufl.FacetNormal(mesh)
    g_tau = ufl.as_vector((0.0, 0.0))
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=mt, subdomain_id=1)

    a00 -= ufl.inner(ufl.outer(n, n) * ufl.dot(ufl.grad(u), n), v) * ds
    a01 -= ufl.inner(ufl.outer(n, n) * ufl.dot(
        - p * ufl.Identity(u.ufl_shape[0]), n), v) * ds
    L0 += ufl.inner(g_tau, v) * ds

    a_nest = ((a00, a01),
              (a10, a11))
    L_nest = (L0, L1)

    # Assemble MPC nest matrix
    A_nest = dolfinx_mpc.create_matrix_nest(a_nest, [mpc_v, mpc_q])
    dolfinx_mpc.assemble_matrix_nest(A_nest, a_nest, [mpc_v, mpc_q], bcs)
    A_nest.assemble()

    # Assemble original nest matrix
    A_org_nest = dolfinx.fem.assemble_matrix_nest(a_nest, bcs)
    A_org_nest.assemble()

    # MPC nested rhs
    b_nest = dolfinx_mpc.create_vector_nest(L_nest, [mpc_v, mpc_q])
    dolfinx_mpc.assemble_vector_nest(b_nest, L_nest, [mpc_v, mpc_q])

    dolfinx.fem.assemble.apply_lifting_nest(b_nest, a_nest, bcs)
    for b_sub in b_nest.getNestSubVecs():
        b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD,
                          mode=PETSc.ScatterMode.REVERSE)

    bcs0 = dolfinx.cpp.fem.bcs_rows(
        dolfinx.fem.assemble._create_cpp_form(L_nest), bcs)
    dolfinx.fem.assemble.set_bc_nest(b_nest, bcs0)

    # Original dolfinx rhs
    b_org_nest = dolfinx.fem.assemble_vector_nest(L_nest)
    dolfinx.fem.assemble.apply_lifting_nest(b_org_nest, a_nest, bcs)
    for b_sub in b_org_nest.getNestSubVecs():
        b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD,
                          mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.assemble.set_bc_nest(b_org_nest, bcs0)

    # -- Monolithic assembly
    dofs = dolfinx.fem.locate_dofs_topological((W.sub(0), V), 1, bc_facets)
    bc1 = dolfinx.DirichletBC(inlet_velocity, dofs, W.sub(0))

    bcs = [bc1]

    V, V_to_W = W.sub(0).collapse(True)
    mpc_vq = dolfinx_mpc.MultiPointConstraint(W)
    n_approx = dolfinx_mpc.utils.create_normal_approximation(
        V, mt.indices[mt.values == 1])
    mpc_vq.create_slip_constraint((mt, 1), n_approx, sub_space=W.sub(0),
                                  sub_map=V_to_W, bcs=bcs)
    mpc_vq.finalize()

    f = dolfinx.Constant(mesh, ((0, 0)))
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    a = (
        ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        - ufl.inner(ufl.div(u), q) * ufl.dx
    )

    L = ufl.inner(f, v) * ufl.dx + dolfinx.Constant(mesh, 0.0) * q * ufl.dx

    # No prescribed shear stress
    n = ufl.FacetNormal(mesh)
    g_tau = ufl.as_vector((0.0, 0.0))
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=mt, subdomain_id=1)

    # Terms due to slip condition
    # Explained in for instance: https://arxiv.org/pdf/2001.10639.pdf
    a -= ufl.inner(ufl.outer(n, n) * ufl.dot(ufl.grad(u), n), v) * ds
    a -= ufl.inner(ufl.outer(n, n) * ufl.dot(
        - p * ufl.Identity(u.ufl_shape[0]), n), v) * ds
    L += ufl.inner(g_tau, v) * ds

    # Assemble LHS matrix and RHS vector
    A = dolfinx_mpc.assemble_matrix_cpp(a, mpc_vq, bcs)
    A.assemble()
    A_org = dolfinx.fem.assemble_matrix(a, bcs)
    A_org.assemble()

    b = dolfinx_mpc.assemble_vector(L, mpc_vq)
    b_org = dolfinx.fem.assemble_vector(L)

    # Set Dirichlet boundary condition values in the RHS
    dolfinx.fem.assemble.apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b, bcs)

    dolfinx.fem.assemble.apply_lifting(b_org, [a], [bcs])
    b_org.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b_org, bcs)

    # -- Verification
    def nest_matrix_norm(A):
        assert A.getType() == "nest"
        nrows, ncols = A.getNestSize()
        sub_A = [A.getNestSubMatrix(row, col)
                 for row in range(nrows) for col in range(ncols)]
        return sum(map(lambda A_: A_.norm()**2 if A_ else 0.0, sub_A))**0.5

    # -- Ensure monolithic and nest matrices are the same
    assert np.isclose(nest_matrix_norm(A_nest), A.norm())
