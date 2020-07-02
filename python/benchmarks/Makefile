periodic:
	mpirun -n 32 python3 ref_periodic.py --nref=6 --tet --boomeramg
	mpirun -n 32 python3 bench_periodic.py --nref=6 --tet --boomeramg
	python3 visualize_iterations.py --periodic

elasticity:
	mpirun -n 32 python3 ref_elasticity.py --nref=6 --gamg --xdmf
	mpirun -n 32 python3 bench_elasticity_edge.py --nref=6 --gamg --xdmf
	python3 visualize_iterations.py --elasticity 
