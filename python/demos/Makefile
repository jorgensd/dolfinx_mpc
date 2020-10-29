demos3D-serial:
	mpirun -n 1 python3 demo_contact_3D.py --theta 0
	mpirun -n 1 python3 demo_contact_3D.py --theta 1.05
	mpirun -n 1 python3 demo_contact_3D.py --gmsh --theta 0
	mpirun -n 1 python3 demo_contact_3D.py --gmsh --theta 1.05
	mpirun -n 1 python3 demo_contact_3D.py --gmsh --no-slip --theta 0
	mpirun -n 1 python3 demo_contact_3D.py --gmsh --no-slip --theta 1.05
	mpirun -n 1 python3 demo_contact_3D.py --gmsh --no-slip --hex --theta 0
	mpirun -n 1 python3 demo_contact_3D.py --gmsh --no-slip --hex --theta 1.05


demos3D-parallel:
	mpirun -n 4 python3 demo_contact_3D.py --theta 0
	mpirun -n 4 python3 demo_contact_3D.py --theta 1.05
	mpirun -n 4 python3 demo_contact_3D.py --gmsh --theta 0
	mpirun -n 4 python3 demo_contact_3D.py --gmsh --theta 1.05
	mpirun -n 4 python3 demo_contact_3D.py --gmsh --no-slip --theta 0
	mpirun -n 4 python3 demo_contact_3D.py --gmsh --no-slip --theta 1.05
	mpirun -n 4 python3 demo_contact_3D.py --gmsh --no-slip --hex --theta 0
	mpirun -n 4 python3 demo_contact_3D.py --gmsh --no-slip --hex --theta 1.05
