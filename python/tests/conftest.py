import pytest


@pytest.fixture
def get_assemblers(request):
    """
    Get eiher numba assembler or C++ assembler depending on the request
    """
    if request.param == "numba":
        try:
            import numba  # noqa: F401
        except ModuleNotFoundError:
            pytest.skip("Numba not installed")
        from dolfinx_mpc.numba import assemble_matrix, assemble_vector

        return (assemble_matrix, assemble_vector)
    elif request.param == "C++":
        from dolfinx_mpc import assemble_matrix, assemble_vector

        return (assemble_matrix, assemble_vector)
    else:
        raise RuntimeError(f"Undefined assembler type: {request.param}.\n" + "Options are 'numba' or 'C++'")
