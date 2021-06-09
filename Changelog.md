# Changelog

## Dev
- Internal updates to match DOLFINx main API
- Slip condition constructor moved to C++ (Speedup for large problems)
- Update GMSH code in demos to be compatible with [GMSH 4.8.4](https://gitlab.onelab.info/gmsh/gmsh/-/tags/gmsh_4_8_4).
- Renaming and cleanup in `assemble_matrix` in C++/ 
- Use scipy sparse matrices for verification
- Introduce new class `dolfinx_mpc.LinearProblem` mimicking the DOLFINx class (Usage illustrated in `demo_periodic.py`)
- Additional `kwarg` `b: PETSc.Vec` for `assemble_vector` to be able to re-use Vector.
- Additional `kwargs`: `form_compiler_parameters` and `jit_parameters` to `assemble_matrix`, `assemble_vector`, to allow usage of fast math etc.  



## 0.1.0 (11.05.2021)
- First tagged release of dolfinx_mpc, compatible with [DOLFINx 0.1.0](https://github.com/FEniCS/dolfinx/releases/tag/0.1.0).