# Copyright (C) 2022-2025 Nathan Sime, Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
from __future__ import annotations

from mpi4py import MPI

import basix
import dolfinx
import numpy as np
import pytest
import ufl

import dolfinx_mpc


@pytest.mark.skipif(
    np.issubdtype(dolfinx.default_scalar_type, np.complexfloating),
    reason="This test does not work in complex mode.",
)
@pytest.mark.parametrize("poly_order", [1, 2, 3])
def test_nonlinear_poisson(poly_order):
    # Solve a standard Poisson problem with known solution which has
    # rotational symmetry of pi/2 at (x, y) = (0.5, 0.5). Therefore we may
    # impose MPCs on those DoFs which lie on the symmetry plane(s) and test
    # our numerical approximation. We do not impose any constraints at the
    # rotationally degenerate point (x, y) = (0.5, 0.5).

    N_vals = np.array([4, 8, 16], dtype=np.int32)
    l2_error = np.zeros_like(N_vals, dtype=np.double)
    for run_no, N in enumerate(N_vals):
        mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
        V = dolfinx.fem.functionspace(mesh, ("Lagrange", poly_order))
        u_bc = dolfinx.fem.Function(V)
        u_bc.x.array[:] = 0.0

        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
        facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
        topological_dofs = dolfinx.fem.locate_dofs_topological(V, 1, facets)
        zero = np.array(0, dtype=dolfinx.default_scalar_type)
        bc = dolfinx.fem.dirichletbc(zero, topological_dofs, V)
        bcs = [bc]

        # -- Impose the pi/2 rotational symmetry of the solution as a constraint,
        # -- except at the centre DoF
        def periodic_boundary(x):
            eps = 1000 * np.finfo(x.dtype).resolution
            return np.isclose(x[0], 0.5, atol=eps) & ((x[1] < 0.5 - eps) | (x[1] > 0.5 + eps))

        def periodic_relation(x):
            out_x = np.zeros_like(x)
            out_x[0] = x[1]
            out_x[1] = x[0]
            out_x[2] = x[2]
            return out_x

        mpc = dolfinx_mpc.MultiPointConstraint(V)
        mpc.create_periodic_constraint_geometrical(V, periodic_boundary, periodic_relation, bcs)
        mpc.finalize()

        # Define variational problem
        V_mpc = mpc.function_space

        # NOTE: The unknown function should always live in the MPC space,
        # but to be compatible with the dirichlet bc the arguments (test and trial function)
        # has to live in the original space V
        u = dolfinx.fem.Function(V_mpc)
        v = ufl.TestFunction(V)
        x = ufl.SpatialCoordinate(mesh)

        u_soln = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
        f = -ufl.div((1 + u_soln**2) * ufl.grad(u_soln))

        F = ufl.inner((1 + u**2) * ufl.grad(u), ufl.grad(v)) * ufl.dx - ufl.inner(f, v) * ufl.dx
        J = ufl.derivative(F, u, ufl.TrialFunction(V))

        # Sanity check that the MPC class has some constraints to impose
        num_slaves_global = mesh.comm.allreduce(len(mpc.slaves), op=MPI.SUM)
        num_masters_global = mesh.comm.allreduce(len(mpc.masters.array), op=MPI.SUM)

        assert num_slaves_global > 0
        assert num_masters_global == num_slaves_global
        tol = 1e1 * np.finfo(u.x.array.dtype).resolution
        petsc_options = {
            "snes_type": "newtonls",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "snes_atol": tol,
            "snes_rtol": tol,
            "snes_linesearch_type": "none",
            "ksp_error_if_not_converged": True,
            "snes_error_if_not_converged": True,
            "snes_monitor": None,
        }
        problem = dolfinx_mpc.NonlinearProblem(F, u, mpc=mpc, bcs=bcs, J=J, petsc_options=petsc_options)
        # solver = NewtonSolverMPC(mesh.comm, problem, mpc)

        # Ensure the solver works with nonzero initial guess
        u.interpolate(lambda x: x[0] ** 2 * x[1] ** 2)
        _, converged, num_iterations = problem.solve()

        print(f"Converged: {converged}, iterations: {num_iterations}")
        l2_error_local = dolfinx.fem.assemble_scalar(dolfinx.fem.form((u - u_soln) ** 2 * ufl.dx))
        l2_error_global = mesh.comm.allreduce(l2_error_local, op=MPI.SUM)

        l2_error[run_no] = l2_error_global**0.5

    rates = np.log(l2_error[:-1] / l2_error[1:]) / np.log(2.0)

    assert np.all(rates > poly_order + 0.9)


@pytest.mark.parametrize("tensor_order", [0, 1, 2])
@pytest.mark.parametrize("poly_order", [1, 2, 3])
def test_homogenize(tensor_order, poly_order):
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    if tensor_order == 0:
        shape = ()
    elif tensor_order == 1:
        shape = (mesh.geometry.dim,)
    elif tensor_order == 2:
        shape = (mesh.geometry.dim, mesh.geometry.dim)
    else:
        pytest.xfail("Unknown tensor order")

    cellname = mesh.ufl_cell().cellname()
    el = basix.ufl.element(basix.ElementFamily.P, cellname, poly_order, shape=shape, dtype=mesh.geometry.x.dtype)

    V = dolfinx.fem.functionspace(mesh, el)

    def periodic_boundary(x):
        return np.isclose(x[0], 0.0)

    def periodic_relation(x):
        out_x = np.zeros(x.shape)
        out_x[0] = 1.0 - x[0]
        out_x[1] = x[1]
        out_x[2] = x[2]
        return out_x

    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.create_periodic_constraint_geometrical(V, periodic_boundary, periodic_relation, [])
    mpc.finalize()

    # Sanity check that the MPC class has some constraints to impose
    num_slaves_global = mesh.comm.allreduce(len(mpc.slaves), op=MPI.SUM)
    assert num_slaves_global > 0

    u = dolfinx.fem.Function(V)
    u.x.petsc_vec.set(1.0)
    assert np.isclose(u.x.petsc_vec.min()[1], u.x.petsc_vec.max()[1])
    assert np.isclose(u.x.petsc_vec.array_r[0], 1.0)

    mpc.homogenize(u)

    with u.x.petsc_vec.localForm() as u_:
        for i in range(V.dofmap.index_map.size_local * V.dofmap.index_map_bs):
            if i in mpc.slaves:
                assert np.isclose(u_.array_r[i], 0.0)
            else:
                assert np.isclose(u_.array_r[i], 1.0)
    u.x.petsc_vec.destroy()
