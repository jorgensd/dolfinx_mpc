import pygmsh
import numpy as np

geom = pygmsh.built_in.Geometry()
rect = geom.add_rectangle(0.0, 2.0, 0.0, 1.0, 0.0, 0.1)
geom.rotate(rect, [0,0,0], np.pi/4, [0,0,1])

pygmsh.generate_mesh(geom, geo_filename="test_mesh.geo")
