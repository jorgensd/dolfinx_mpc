periodic:
	mpirun -n 23 python3 ref_periodic.py --nref 6 --tet --boomeramg --degree 2
	mpirun -n 23 python3 bench_periodic.py --nref 6 --tet --boomeramg --degree 2
	python3 visualize_iterations.py --periodic

elasticity:
	mpirun -n 23 python3 ref_elasticity.py --nref 6 --gamg --xdmf
	mpirun -n 23 python3 bench_elasticity_edge.py --nref 6 --gamg --xdmf --info
	python3 visualize_iterations.py --elasticity 
