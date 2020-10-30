contact3D-serial:
	python3 demo_contact_3D.py --theta 0 --timing
	python3 demo_contact_3D.py --theta 1.05 --timing
	python3 demo_contact_3D.py --gmsh --theta 0 --timing
	python3 demo_contact_3D.py --gmsh --theta 1.05 --timing
	python3 demo_contact_3D.py --gmsh --no-slip --theta 0 --timing
	python3 demo_contact_3D.py --gmsh --no-slip --theta 1.05 --timing
	python3 demo_contact_3D.py --gmsh --no-slip --hex --theta 0 --timing
	python3 demo_contact_3D.py --gmsh --no-slip --hex --theta 1.05 --timing


contact3D-parallel:
	mpirun -n 4 python3 demo_contact_3D.py --theta 0 --timing
	mpirun -n 4 python3 demo_contact_3D.py --theta 1.05 --timing
	mpirun -n 4 python3 demo_contact_3D.py --gmsh --theta 0 --timing
	mpirun -n 4 python3 demo_contact_3D.py --gmsh --theta 1.05 --timing
	mpirun -n 4 python3 demo_contact_3D.py --gmsh --no-slip --theta 0 --timing
	mpirun -n 4 python3 demo_contact_3D.py --gmsh --no-slip --theta 1.05 --timing
	mpirun -n 4 python3 demo_contact_3D.py --gmsh --no-slip --hex --theta 0 --timing
	mpirun -n 4 python3 demo_contact_3D.py --gmsh --no-slip --hex --theta 1.05 --timing

contact2D-serial:
	python3 demo_contact_2D.py --theta 0 --timing
	python3 demo_contact_2D.py --theta 1.05 --timing
	python3 demo_contact_2D.py --gmsh --theta 0 --timing
	python3 demo_contact_2D.py --gmsh --theta 1.05 --timing
	python3 demo_contact_2D.py --quad --gmsh --theta 0 --timing
	python3 demo_contact_2D.py --quad --gmsh --theta 1.05 --timing

contact2D-parallel:
	mpirun -n 4 python3 demo_contact_2D.py --theta 0 --timing
	mpirun -n 4 python3 demo_contact_2D.py --theta 1.05 --timing
	mpirun -n 4 python3 demo_contact_2D.py --gmsh --theta 0 --timing
	mpirun -n 4 python3 demo_contact_2D.py --gmsh --theta 1.05 --timing
	mpirun -n 4 python3 demo_contact_2D.py --quad --gmsh --theta 0 --timing
	mpirun -n 4 python3 demo_contact_2D.py --quad --gmsh --theta 1.05 --timing