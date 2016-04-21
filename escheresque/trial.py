import escheresque.stl
import numpy as np
import escheresque.computational_geometry as cg

triangles = np.arange(3)[::-1].reshape(1,3)
inner = np.eye(3)
outer = np.eye(3) + 1
mesh = cg.Mesh(outer, triangles).extrude(inner)
assert mesh.is_orientated()


filename = r'..\data\part{0}.stl'
mesh = escheresque.stl.load_stl(filename.format(0))

mesh.plot()

print(mesh.vertices.shape)
print(mesh.faces.shape)
assert mesh.is_orientated()

mesh = mesh.decimate(500)

print(mesh.vertices.shape)
print(mesh.faces.shape)

escheresque.stl.save_STL(filename.format('dec'), mesh)
mesh.plot()