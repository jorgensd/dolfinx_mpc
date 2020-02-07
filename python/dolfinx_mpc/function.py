import numba


@numba.njit
def backsubstitution(b, mpc):
    """
    Insert mpc values into vector bc
    """
    (slaves, masters, coefficients, offsets) = mpc
    for i, slave in enumerate(slaves):
        masters_i = masters[offsets[i]:offsets[i+1]]
        coeffs_i = coefficients[offsets[i]:offsets[i+1]]
        for j, (master, coeff) in enumerate(zip(masters_i, coeffs_i)):
            if slave == master:
                print("No slaves (since slave is same as master dof)")
                continue

            b[slave] += coeff*b[master]
