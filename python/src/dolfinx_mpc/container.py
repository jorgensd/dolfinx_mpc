from __future__ import annotations

import warnings
from typing import Callable, Optional, Union

import dolfinx
import numpy
import numpy.typing as npt

import dolfinx_mpc.cpp.mpc

_mpc_data_classes = Union[
    dolfinx_mpc.cpp.mpc.mpc_data_double,
    dolfinx_mpc.cpp.mpc.mpc_data_float,
    dolfinx_mpc.cpp.mpc.mpc_data_complex_double,
    dolfinx_mpc.cpp.mpc.mpc_data_complex_float,
]
_float_array_types = Union[
    npt.NDArray[numpy.float32],
    npt.NDArray[numpy.float64],
    npt.NDArray[numpy.complex64],
    npt.NDArray[numpy.complex128],
]

_mpc_classes = Union[
    dolfinx_mpc.cpp.mpc.MultiPointConstraint_double,
    dolfinx_mpc.cpp.mpc.MultiPointConstraint_float,
    dolfinx_mpc.cpp.mpc.MultiPointConstraint_complex_double,
    dolfinx_mpc.cpp.mpc.MultiPointConstraint_complex_float,
]
_float_classes = Union[numpy.float32, numpy.float64, numpy.complex128, numpy.complex64]

_type_names = {
    numpy.float32: "float",
    numpy.float64: "double",
    numpy.complex64: "complex_float",
    numpy.complex128: "complex_double",
}


class _Unset:
    """The default of a deprecated argument, to tell whether it was passed."""

    def __repr__(self) -> str:
        return "<unset>"


_UNSET = _Unset()


def _deprecated(old: str, new: str, stacklevel: int = 3):
    """Warn that the argument `old` is deprecated in favour of `new`, at the caller of the public
    function `stacklevel - 2` frames up."""
    warnings.warn(f"`{old}` is deprecated, use {new} instead.", DeprecationWarning, stacklevel=stacklevel)


def _default_tolerance(dtype: npt.DTypeLike) -> float:
    """The default distance and coefficient tolerance: 500 machine epsilon of the real type of `dtype`.

    Mirrors `dolfinx_mpc::default_tolerance` in C++.
    """
    return float(500 * numpy.finfo(dtype).eps)


def _tolerance(value: Optional[float], dtype: npt.DTypeLike) -> float:
    """`value` as a Python float, by default :func:`_default_tolerance` of `dtype`."""
    return _default_tolerance(dtype) if value is None else float(value)


def _cpp_function(name: str, dtype: npt.DTypeLike) -> Callable:
    """The C++ function `name` for constraints of scalar type `dtype`, bound as `name_<type>`."""
    return getattr(dolfinx_mpc.cpp.mpc, f"{name}_{_type_names[numpy.dtype(dtype).type]}")


def _scalar_type(real_type: npt.DTypeLike, dtype: npt.DTypeLike | None = None) -> type:
    """The scalar type of a constraint on a mesh with coordinates of `real_type`.

    Args:
        real_type: The type of the mesh coordinates
        dtype: The scalar type asked for. Defaults to the default scalar type of DOLFINx, real or
            complex, at the precision of the mesh.

    Raises:
        ValueError: If `dtype` is not one of float32, float64, complex64 and complex128, or its
            precision differs from the mesh's.
    """
    real = numpy.dtype(real_type)
    if dtype is None:
        is_complex = numpy.issubdtype(dolfinx.default_scalar_type, numpy.complexfloating)
        dtype = numpy.promote_types(real, numpy.complex64) if is_complex else real
    scalar = numpy.dtype(dtype)
    if scalar.type not in (numpy.float32, numpy.float64, numpy.complex64, numpy.complex128):
        raise ValueError(f"Unsupported scalar type {scalar} for a constraint")
    if numpy.finfo(scalar).dtype != real:
        raise ValueError(
            f"A constraint of scalar type {scalar} needs a mesh of {numpy.finfo(scalar).dtype}, not {real}"
        )
    return scalar.type


class MPCData:
    r"""The rows of a constraint on one process, before it is finalized.

    Row `i` is the equation :math:`u_s = \sum_j c_j u_{m_j}` of the slave :math:`s` =
    `slaves[i]`, with :math:`m_j` = `masters[j]` and :math:`c_j` = `coeffs[j]` for
    `offsets[i] <= j < offsets[i + 1]`.

    Attributes:
        slaves: The slave of each row: a dof of the space of the slaves, local to the process
            (owned or ghost) and unrolled
        masters: The masters of all rows, row after row: global, unrolled dofs of the space of
            each master
        coeffs: The coefficient of each master
        owners: The process owning each master
        offsets: The masters of row `i` are `masters[offsets[i]:offsets[i + 1]]`, so `offsets`
            has one entry more than `slaves`
        master_blocks: The block of each master, in the numbering of whoever made the rows, or
            `None` if every master is in the space of the slaves. Given on every process or on
            none. A :class:`MultiPointConstraint` stores codes here until it is finalized (see
            :meth:`MultiPointConstraint.__init__`).

    The arrays are views of the C++ object, valid while this object is.
    """

    _cpp_object: _mpc_data_classes

    def __init__(
        self,
        slaves: npt.NDArray[numpy.int32],
        masters: npt.NDArray[numpy.int64],
        coeffs: _float_array_types,
        owners: npt.NDArray[numpy.int32],
        offsets: npt.NDArray[numpy.int32],
        master_blocks: Optional[npt.NDArray[numpy.int32]] = None,
    ):
        args = (
            numpy.asarray(slaves, dtype=numpy.int32),
            numpy.asarray(masters, dtype=numpy.int64),
            coeffs,
            numpy.asarray(owners, dtype=numpy.int32),
            numpy.asarray(offsets, dtype=numpy.int32),
            None if master_blocks is None else numpy.asarray(master_blocks, dtype=numpy.int32),
        )
        if coeffs.dtype.type == numpy.float32:
            self._cpp_object = dolfinx_mpc.cpp.mpc.mpc_data_float(*args)
        elif coeffs.dtype.type == numpy.float64:
            self._cpp_object = dolfinx_mpc.cpp.mpc.mpc_data_double(*args)
        elif coeffs.dtype.type == numpy.complex64:
            self._cpp_object = dolfinx_mpc.cpp.mpc.mpc_data_complex_float(*args)
        elif coeffs.dtype.type == numpy.complex128:
            self._cpp_object = dolfinx_mpc.cpp.mpc.mpc_data_complex_double(*args)
        else:
            raise ValueError(f"Unsupported dtype {coeffs.dtype.type} for coefficients")

    @classmethod
    def empty(cls, dtype: npt.DTypeLike, master_blocks: bool = False) -> MPCData:
        """No rows, with coefficients of `dtype`, and with (empty) blocks if `master_blocks`."""
        return cls(
            numpy.zeros(0, dtype=numpy.int32),
            numpy.zeros(0, dtype=numpy.int64),
            numpy.zeros(0, dtype=dtype),
            numpy.zeros(0, dtype=numpy.int32),
            numpy.zeros(1, dtype=numpy.int32),
            numpy.zeros(0, dtype=numpy.int32) if master_blocks else None,
        )

    @classmethod
    def from_cpp(cls, cpp_object: _mpc_data_classes) -> MPCData:
        """Wrap rows made in C++, such as the result of a constraint generator."""
        data = cls.__new__(cls)
        data._cpp_object = cpp_object
        return data

    def append(
        self,
        slaves: npt.NDArray[numpy.int32],
        masters: npt.NDArray[numpy.int64],
        coeffs: _float_array_types,
        owners: npt.NDArray[numpy.int32],
        offsets: npt.NDArray[numpy.int32],
        master_blocks: Optional[npt.NDArray[numpy.int32]] = None,
    ) -> MPCData:
        """These rows followed by the rows given, as new data with the coefficient type of these.

        The blocks are kept if either has them; missing ones must then be given by the caller.
        """
        if (self.master_blocks is None) != (master_blocks is None):
            raise ValueError("Either both or neither of the rows must have master blocks")
        dtype = self.coeffs.dtype
        return MPCData(
            numpy.concatenate([self.slaves, slaves]),
            numpy.concatenate([self.masters, masters]),
            numpy.concatenate([self.coeffs, numpy.asarray(coeffs, dtype=dtype)]),
            numpy.concatenate([self.owners, owners]),
            numpy.concatenate([self.offsets, numpy.asarray(offsets[1:]) + self.offsets[-1]]),
            None if master_blocks is None else numpy.concatenate([self.master_blocks, master_blocks]),
        )

    @property
    def slaves(self):
        return self._cpp_object.slaves

    @property
    def masters(self):
        return self._cpp_object.masters

    @property
    def coeffs(self):
        return self._cpp_object.coeffs

    @property
    def owners(self):
        return self._cpp_object.owners

    @property
    def offsets(self):
        return self._cpp_object.offsets

    @property
    def master_blocks(self):
        return self._cpp_object.master_blocks
