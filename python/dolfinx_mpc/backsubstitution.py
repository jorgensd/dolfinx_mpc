import numba
from petsc4py import PETSc


def backsubstitution(constraint, vector):

    # Unravel data from constraint
    coefficients = constraint.coefficients()
    masters_local = constraint.masters_local().array
    offsets = constraint.masters_local().offsets
    slaves_local = constraint.slaves()
    constraint_wrapper = (slaves_local, masters_local,
                          coefficients, offsets)
    backsubstitution_numba(vector, constraint_wrapper)
    vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                       mode=PETSc.ScatterMode.FORWARD)
    return vector


@numba.njit(cache=True)
def backsubstitution_numba(b, mpc):
    """
    Insert mpc values into vector bc
    """
    (slaves,
     masters, coefficients, offsets) = mpc
    for i, slave in enumerate(slaves):
        masters_i = masters[offsets[i]:offsets[i+1]]
        coeffs_i = coefficients[offsets[i]:offsets[i+1]]
        for master, coeff in zip(masters_i, coeffs_i):
            b[slave] += coeff*b[master]
