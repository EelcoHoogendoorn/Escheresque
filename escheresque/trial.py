import openmesh
import escheresque.stl
import numpy as np
from mayavi import mlab
import escheresque.computational_geometry as cg

triangles = np.arange(3)[::-1].reshape(1,3)
inner = np.eye(3)
outer = np.eye(3) + 1
vertices, triangles = cg.extrude(outer, inner, triangles)
assert cg.is_orientated(triangles)
print(vertices)
print(triangles)
filename = r'C:\Users\Eelco\Dropbox\Git\Escheresque\data\part{0}.stl'
vertices, triangles = escheresque.stl.load_stl(filename.format(0))
print(vertices.shape)
print(triangles.shape)
assert cg.is_orientated(triangles)

def plot(vertices, triangles):
    q = vertices.mean(axis=0)
    x, y, z = (vertices+q/4).T
    mlab.triangular_mesh(x, y, z, triangles)
    # mlab.triangular_mesh(x, y, z, t, color=(0, 0, 0), representation='wireframe')
    mlab.show()

mesh = openmesh.TriMesh()
vhandles = [mesh.add_vertex(openmesh.TriMesh.Point(*vertex)) for vertex in vertices.astype(np.float64)]

for triangle in triangles:
    mesh.add_face([vhandles[v] for v in triangle])

print(mesh.n_vertices())
print(mesh.n_faces())

decimater = openmesh.TriMeshDecimater(mesh)
modquadrichandle = openmesh.TriMeshModQuadricHandle()
decimater.add(modquadrichandle)
decimater.initialize()

print(decimater.decimate(1000))
print(decimater.decimate_to_faces(10000))

# mesh = decimater.mesh()       # no such method available!
fh = list(mesh.faces())[0]
print(dir(fh))

mesh.garbage_collection()
print(mesh.n_vertices())
print(mesh.n_faces())

print(dir(mesh))

fh = list(mesh.faces())[0]
print([fh.idx() for fh in mesh.faces()][-20:])
print(dir(fh))
quit()

vertices = np.array([list(mesh.point(vh)) for vh in mesh.vertices()])
faces = np.array([mesh.face(fh) for fh in mesh.faces()])
print(vertices)
print(faces)
plot(vertices, faces)