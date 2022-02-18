# Copyright (C) 2022 Nathan Sime
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
import pytest
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import dolfinx
import dolfinx_mpc
import ufl


class NonlinearMPCProblem(dolfinx.fem.petsc.NonlinearProblem):

    def __init__(self, F, u, mpc, bcs=[], J=None, form_compiler_params={},
                 jit_params={}):
        self.mpc = mpc
        super().__init__(F, u, bcs=bcs, J=J,
                         form_compiler_params=form_compiler_params,
                         jit_params=jit_params)

    def F(self, x: PETSc.Vec, F: PETSc.Vec):
        with F.localForm() as F_local:
            F_local.set(0.0)
        dolfinx_mpc.assemble_vector(self._L, self.mpc, b=F)

        # Apply boundary condition
        dolfinx_mpc.apply_lifting(F, [self._a], bcs=[self.bcs],
                                  constraint=self.mpc, x0=[x], scale=-1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.set_bc(F, self.bcs, x, -1.0)

    def J(self, x: PETSc.Vec, A: PETSc.Mat):
        A.zeroEntries()
        dolfinx_mpc.assemble_matrix(self._a, self.mpc, bcs=self.bcs, A=A)
        A.assemble()


class NewtonSolverMPC(dolfinx.cpp.nls.petsc.NewtonSolver):
    def __init__(self, comm: MPI.Intracomm, problem: NonlinearMPCProblem,
                 mpc: dolfinx_mpc.MultiPointConstraint):
        """A Newton solver for non-linear MPC problems."""
        super().__init__(comm)
        self.mpc = mpc
        self.u_mpc = dolfinx.fem.Function(mpc.function_space)

        # Create matrix and vector to be used for assembly of the non-linear
        # MPC problem
        self._A = dolfinx_mpc.cpp.mpc.create_matrix(
            problem.a, mpc._cpp_object)
        self._b = dolfinx.cpp.la.petsc.create_vector(
            mpc.function_space.dofmap.index_map,
            mpc.function_space.dofmap.index_map_bs)

        self.setF(problem.F, self._b)
        self.setJ(problem.J, self._A)
        self.set_form(problem.form)
        self.set_update(self.update)

    def update(self, solver: dolfinx.nls.petsc.NewtonSolver,
               dx: PETSc.Vec, x: PETSc.Vec):
        # We need to use a vector created on the MPC's space to update ghosts
        self.u_mpc.vector.array = x.array_r
        self.u_mpc.vector.axpy(-1.0, dx)
        self.u_mpc.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                      mode=PETSc.ScatterMode.FORWARD)
        self.mpc.homogenize(self.u_mpc.vector)
        self.mpc.backsubstitution(self.u_mpc.vector)
        x.array = self.u_mpc.vector.array_r
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                      mode=PETSc.ScatterMode.FORWARD)

    def solve(self, u: dolfinx.fem.Function):
        """Solve non-linear problem into function u. Returns the number
        of iterations and if the solver converged."""
        n, converged = super().solve(u.vector)
        u.x.scatter_forward()
        return n, converged

    @property
    def A(self) -> PETSc.Mat:
        """Jacobian matrix"""
        return self._A

    @property
    def b(self) -> PETSc.Vec:
        """Residual vector"""
        return self._b


@pytest.mark.skipif(np.issubdtype(PETSc.ScalarType, np.complexfloating),
                    reason="This test does not work in complex mode.")
@pytest.mark.parametrize("poly_order", [1, 2, 3])
def test_nonlinear_possion(poly_order):
    # Solve a standard Poisson problem with known solution which has
    # rotational symmetry of pi/2 at (x, y) = (0.5, 0.5). Therefore we may
    # impose MPCs on those DoFs which lie on the symmetry plane(s) and test
    # our numerical approximation. We do not impose any constraints at the
    # rotationally degenerate point (x, y) = (0.5, 0.5).

    N_vals = np.array([4, 8, 16], dtype=np.int32)
    l2_error = np.zeros_like(N_vals, dtype=np.double)
    for run_no, N in enumerate(N_vals):
        mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)

        V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", poly_order))

        def boundary(x):
            return np.ones_like(x[0], dtype=np.int8)

        u_bc = dolfinx.fem.Function(V)
        with u_bc.vector.localForm() as u_local:
            u_local.set(0.0)

        facets = dolfinx.mesh.locate_entities_boundary(mesh, 1, boundary)
        topological_dofs = dolfinx.fem.locate_dofs_topological(V, 1, facets)
        zero = np.array(0, dtype=PETSc.ScalarType)
        bc = dolfinx.fem.dirichletbc(zero, topological_dofs, V)
        bcs = [bc]

        # Define variational problem
        u = dolfinx.fem.Function(V)
        v = ufl.TestFunction(V)
        x = ufl.SpatialCoordinate(mesh)

        u_soln = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
        f = -ufl.div((1 + u_soln**2) * ufl.grad(u_soln))

        F = ufl.inner((1 + u**2) * ufl.grad(u), ufl.grad(v)) * ufl.dx \
            - ufl.inner(f, v) * ufl.dx
        J = ufl.derivative(F, u)

        # -- Impose the pi/2 rotational symmetry of the solution as a constraint,
        # -- except at the centre DoF
        def periodic_boundary(x):
            eps = 1e-10
            return np.isclose(x[0], 0.5) & (
                (x[1] < 0.5 - eps) | (x[1] > 0.5 + eps))

        def periodic_relation(x):
            out_x = np.zeros(x.shape)
            out_x[0] = x[1]
            out_x[1] = x[0]
            out_x[2] = x[2]
            return out_x

        mpc = dolfinx_mpc.MultiPointConstraint(V)
        mpc.create_periodic_constraint_geometrical(
            V, periodic_boundary, periodic_relation, bcs)
        mpc.finalize()

        # Sanity check that the MPC class has some constraints to impose
        num_slaves_global = mesh.comm.allreduce(len(mpc.slaves), op=MPI.SUM)
        num_masters_global = mesh.comm.allreduce(
            len(mpc.masters.array), op=MPI.SUM)

        assert num_slaves_global > 0
        assert num_masters_global == num_slaves_global

        problem = NonlinearMPCProblem(F, u, mpc, bcs=bcs, J=J)
        solver = NewtonSolverMPC(mesh.comm, problem, mpc)

        # Ensure the solver works with nonzero initial guess
        u.interpolate(lambda x: x[0]**2 * x[1]**2)
        solver.solve(u)

        l2_error_local = dolfinx.fem.assemble_scalar(
            dolfinx.fem.form((u - u_soln) ** 2 * ufl.dx))
        l2_error_global = mesh.comm.allreduce(l2_error_local, op=MPI.SUM)

        l2_error[run_no] = l2_error_global**0.5

    rates = np.log(l2_error[:-1] / l2_error[1:]) / np.log(2.0)

    assert np.all(rates > poly_order + 0.9)


@pytest.mark.parametrize("element", [ufl.FiniteElement, ufl.VectorElement])
@pytest.mark.parametrize("poly_order", [1, 2, 3])
def test_homogenize(element, poly_order):
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)

    V = dolfinx.fem.FunctionSpace(
        mesh, element("CG", mesh.ufl_cell(), poly_order))

    def periodic_boundary(x):
        return np.isclose(x[0], 0.0)

    def periodic_relation(x):
        out_x = np.zeros(x.shape)
        out_x[0] = 1.0 - x[0]
        out_x[1] = x[1]
        out_x[2] = x[2]
        return out_x

    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.create_periodic_constraint_geometrical(
        V, periodic_boundary, periodic_relation, [])
    mpc.finalize()

    # Sanity check that the MPC class has some constraints to impose
    num_slaves_global = mesh.comm.allreduce(len(mpc.slaves), op=MPI.SUM)
    assert num_slaves_global > 0

    u = dolfinx.fem.Function(V)
    u.vector.set(1.0)

    assert np.isclose(u.vector.min()[1], u.vector.max()[1])
    assert np.isclose(u.vector.array_r[0], 1.0)

    mpc.homogenize(u.vector)

    with u.vector.localForm() as u_:
        for i in range(V.dofmap.index_map.size_local * V.dofmap.index_map_bs):
            if i in mpc.slaves:
                assert np.isclose(u_.array_r[i], 0.0)
            else:
                assert np.isclose(u_.array_r[i], 1.0)
