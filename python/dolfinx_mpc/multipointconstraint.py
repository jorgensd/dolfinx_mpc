from dolfinx import function
from dolfinx_mpc import cpp
import typing


class MultiPointConstraint(cpp.mpc.MultiPointConstraint):
    def __init__(
            self,
            V: typing.Union[function.FunctionSpace],
            master_slave_map: typing.Dict[int, typing.Dict[int, float]]):

        """Representation of MultiPointConstraint which is imposed on
        a linear system.

        """
        slaves = []
        masters = []
        coefficients = []
        offsets = []

        for slave in master_slave_map.keys():
            offsets.append(len(masters))
            slaves.append(slave)
            for master in master_slave_map[slave]:
                masters.append(master)
                coefficients.append(master_slave_map[slave][master])
        # Add additional element to indicate end of list
        # (to determine number of master dofs later on)
        offsets.append(len(masters))
        # Extract cpp function space
        try:
            _V = V._cpp_object
        except AttributeError:
            _V = V
        super().__init__(_V, slaves, masters, coefficients, offsets)
