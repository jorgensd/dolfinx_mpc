periodic:
	mpirun -n 32 python3 ref_periodic.py -n 2 --tet --boomeramg
	mpirun -n 32 python3 bench_periodic.py -n 2 --tet --boomeramg
	python3 visualize_iterations.py --periodic

elasticity:
	mpirun -n 32 python3 ref_elasticity.py -n 5 --gamg --xdmf
	mpirun -n 32 python3 bench_elasticity_edge.py -n 5 --gamg --xdmf
	python3 visualize_iterations.py --elasticity 
