# Copyright (C) 2026 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
"""Express a scalar integral condition as a multi point constraint."""

from __future__ import annotations

import typing

from mpi4py import MPI

import dolfinx.fem as _fem
import numpy as np
import numpy.typing as npt
import ufl
from dolfinx import default_scalar_type, la


def create_integral_constraint(
    V: _fem.FunctionSpace,
    weight_form: ufl.Form,
    value: np.floating | np.complexfloating | float | complex,
    bcs: typing.Optional[typing.Sequence[_fem.DirichletBC]] = None,
    rtol: np.floating = 1e-14,
) -> tuple[
    npt.NDArray[np.int32],
    npt.NDArray[np.int64],
    npt.NDArray,
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    _fem.Function,
]:
    r"""Build the constraint data enforcing a scalar integral condition.

    A linear functional :math:`L` applied to :math:`u_h=\sum_i u_i\phi_i` is a
    single linear equation between all the coefficients,

    .. math::
        L(u_h) = \sum_i w_i u_i = \gamma, \qquad w_i = L(\phi_i),

    where :math:`w` is the assembled vector of ``weight_form``. Solving it for
    one degree of freedom :math:`s` gives the affine relation

    .. math::
        u_s = \sum_{i \neq s} \left(-\frac{w_i}{w_s}\right) u_i
              + \frac{\gamma}{w_s},

    that is a constraint with a single slave, every other degree of freedom in
    the support of the functional as a master, and the inhomogeneity
    :math:`\gamma/w_s`. Typical uses are fixing the mean value of a solution,
    :math:`\int_\Omega u~\mathrm{d}x = \gamma`, which makes a pure Neumann
    problem non-singular, or prescribing a boundary average or a flow rate,
    :math:`\int_\Gamma u\cdot n~\mathrm{d}s = \gamma`, without prescribing the
    profile that carries it. It is an alternative to a Lagrange multiplier in a
    real space, eliminating an unknown rather than adding one.

    Args:
        V: The function space the constraint acts on.
        weight_form: A linear form in ``ufl.TestFunction(V)`` defining the
            functional, for instance ``v * ufl.dx`` or
            ``ufl.dot(v, n) * ds(tag)``.
        value: The prescribed value :math:`\gamma` of the functional.
        bcs: Dirichlet conditions on ``V``. A constrained degree of freedom is
            never chosen as the slave, but may still appear as a master, in
            which case passing the same conditions to
            :class:`MultiPointConstraint` folds its contribution into the
            constraint offset.
        rtol: Discard a master whose coefficient is below this fraction of the
            largest one. The weights of degrees of freedom outside the support
            of the functional are returned by quadrature as roundoff rather than
            as exact zeros, so the threshold is relative rather than a test
            against zero.

    Returns:
        The ``slaves``, ``masters``, ``coeffs``, ``owners`` and ``offsets``
        arrays accepted by :meth:`MultiPointConstraint.add_constraint`, and a
        :class:`dolfinx.fem.Function` holding the inhomogeneity, to be passed to
        :class:`MultiPointConstraint` as ``rhs_coeffs``.

    Note:
        Collective. The weights are communicated only to the processes that hold
        the slave, its owner and any process ghosting it, since those are the
        only ones that declare the constraint.

    Note:
        The functional couples every degree of freedom in its support, so the
        master list is as long as that support and the reduced operator
        :math:`K^HAK` gains a dense block over it. That is cheap for a facet
        functional and quadratic in the number of degrees of freedom for a cell
        functional.
    """
    bcs = [] if bcs is None else list(bcs)
    comm = V.mesh.comm
    imap = V.dofmap.index_map
    bs = V.dofmap.index_map_bs
    num_owned = imap.size_local * bs
    dtype = np.dtype(default_scalar_type)
    mpi_scalar = MPI._typedict[dtype.char]

    arguments = ufl.algorithms.extract_arguments(weight_form)
    if len(arguments) != 1 or arguments[0].number() != 0:
        raise ValueError(f"weight_form must be linear in a single test function, got {len(arguments)} argument(s)")
    if arguments[0].ufl_function_space() != V:
        raise ValueError("The test function of weight_form must be in the function space of the constraint")

    # Assemble the functional and accumulate ghost contributions onto the owner
    w = _fem.assemble_vector(_fem.form(weight_form, dtype=dtype))
    w.scatter_reverse(la.InsertMode.add)
    w_owned = w.array[:num_owned]

    # A Dirichlet degree of freedom may be a master, but must not be the slave.
    # Zero those weights in a scratch copy: `set` writes alpha * (value - x0), so
    # alpha=0 masks them whatever the condition prescribes.
    masked = w.array.copy()
    for bc in bcs:
        bc.set(masked, alpha=0.0)

    # The largest weight makes the best slave, since it bounds every coefficient
    # by one in magnitude. That matters, because cond(K^H A K) grows with the
    # square of the coefficient norm and the weights can span many orders of
    # magnitude. Reducing over the masked weights picks it without communicating
    # either the weights or the Dirichlet markers.
    local_best = int(np.argmax(np.abs(masked[:num_owned]))) if num_owned > 0 else 0
    local_magnitude = float(abs(masked[local_best])) if num_owned > 0 else -1.0
    best_magnitude, slave_global = comm.allreduce(
        (local_magnitude, int(imap.local_range[0] * bs) + local_best if num_owned > 0 else -1),
        op=MPI.MAXLOC,
    )
    # `best_magnitude` is |w_s|, never the signed weight, so it is negative only
    # for the sentinel a process with no owned dofs contributes. Comparing it
    # against zero would be too weak: a weight outside the support of the
    # functional comes back from quadrature as roundoff rather than as an exact
    # zero, so a slave whose weight is negligible *relative to the functional*
    # would otherwise be accepted and give coefficients of order 1/rtol.
    scale = comm.allreduce(float(np.abs(w_owned).max(initial=0.0)), op=MPI.MAX)
    if best_magnitude <= rtol * scale:
        raise RuntimeError(
            "No admissible slave: the functional vanishes, to within rtol, on "
            "every degree of freedom that is not constrained by a Dirichlet condition"
        )

    # Only the processes holding the slave declare the constraint, so only they
    # need the weights: gather to one of them and forward to the few others.
    # Every process still contributes its own weights, and only the weights, as
    # the owned degrees of freedom of a process form a contiguous global range
    # and the processes concatenate in rank order, so a gathered entry's
    # position is already its global index.
    slave_block = int(imap.global_to_local(np.array([slave_global // bs], dtype=np.int64))[0])
    holds_slave = slave_block != -1
    holders = [r for r, holds in enumerate(comm.allgather(holds_slave)) if holds]
    root = holders[0]

    counts = np.array(comm.allgather(num_owned), dtype=np.int32)
    displ = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.int32)
    total = int(counts.sum())
    all_weights = np.empty(total, dtype=dtype) if holds_slave else None
    # mpi4py reads a three-entry tuple as (buffer, counts, datatype), so the
    # datatype has to be spelled out whenever displacements are given
    recvbuf = (all_weights, counts, displ, mpi_scalar) if comm.rank == root else None
    comm.Gatherv(np.ascontiguousarray(w_owned, dtype=dtype), recvbuf, root=root)
    if comm.rank == root:
        for other in holders[1:]:
            comm.Send(all_weights, dest=other, tag=0)
    elif holds_slave:
        comm.Recv(all_weights, source=root, tag=0)

    rhs_coeffs = _fem.Function(V, dtype=dtype)
    rhs_coeffs.x.array[:] = 0
    slaves = np.zeros(0, dtype=np.int32)
    masters = np.zeros(0, dtype=np.int64)
    coeffs = np.zeros(0, dtype=dtype)
    owners = np.zeros(0, dtype=np.int32)
    offsets = np.zeros(1, dtype=np.int32)
    if holds_slave:
        assert all_weights is not None
        w_slave = all_weights[slave_global]
        # The slave is not a master of itself
        keep = np.delete(np.arange(total, dtype=np.int64), slave_global)
        all_coeffs = -np.delete(all_weights, slave_global) / w_slave
        # Drop negligible coefficients; they change nothing in the constraint but
        # each costs a ghost, a row of the sparsity pattern and an entry in every
        # element matrix modification
        significant = np.abs(all_coeffs) >= rtol * np.abs(all_coeffs).max(initial=0.0)
        all_owners = np.repeat(np.arange(comm.size, dtype=np.int32), counts)

        slaves = np.array([slave_block * bs + slave_global % bs], dtype=np.int32)
        masters = np.ascontiguousarray(keep[significant], dtype=np.int64)
        coeffs = np.ascontiguousarray(all_coeffs[significant], dtype=dtype)
        owners = np.ascontiguousarray(np.delete(all_owners, slave_global)[significant], dtype=np.int32)
        offsets = np.array([0, masters.size], dtype=np.int32)
        rhs_coeffs.x.array[slaves[0]] = dtype.type(value) / w_slave
    rhs_coeffs.x.scatter_forward()
    return slaves, masters, coeffs, owners, offsets, rhs_coeffs
