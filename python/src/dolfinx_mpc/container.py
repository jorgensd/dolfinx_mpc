from typing import Union
import dolfinx_mpc.cpp.mpc
import numpy.typing as npt
import numpy

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


class MPCData:
    _cpp_object: _mpc_data_classes

    def __init__(
        self,
        slaves: npt.NDArray[numpy.int32],
        masters: npt.NDArray[numpy.int64],
        coeffs: _float_array_types,
        owners: npt.NDArray[numpy.int32],
        offsets: npt.NDArray[numpy.int32],
    ):
        if coeffs.dtype.type == numpy.float32:
            self._cpp_object = dolfinx_mpc.cpp.mpc.mpc_data_float(slaves, masters, coeffs, owners, offsets)
        elif coeffs.dtype.type == numpy.float64:
            self._cpp_object = dolfinx_mpc.cpp.mpc.mpc_data_double(slaves, masters, coeffs, owners, offsets)
        elif coeffs.dtype.type == numpy.complex64:
            self._cpp_object = dolfinx_mpc.cpp.mpc.mpc_data_complex_float(slaves, masters, coeffs, owners, offsets)
        elif coeffs.dtype.type == numpy.complex128:
            self._cpp_object = dolfinx_mpc.cpp.mpc.mpc_data_complex_double(slaves, masters, coeffs, owners, offsets)
        else:
            raise ValueError("Unsupported dtype {coeffs.dtype.type} for coefficients")

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
