import numpy as np
import scipy.spatial
import scipy.optimize

import escheresque.stl
import escheresque.computational_geometry as cg



filename = r'..\data\part{0}.stl'
mesh = escheresque.stl.load_stl(filename.format(0))

# normalize orientation
u, s, v = np.linalg.svd(mesh.vertices, full_matrices=0)
mesh.vertices = mesh.vertices.dot(v[:, ::-1])


print(mesh.vertices.shape)
print(mesh.faces.shape)
assert mesh.is_orientated()


print('volume', mesh.volume()*40**3 * 12)


seed = np.zeros_like(mesh.vertices[:, 0])
seed[np.argmin(mesh.vertices[:,2])] = 1
# seed[np.argmin(mesh.vertices[:,0])] = 1
distance = mesh.geodesic(seed)
mesh.plot(np.cos(distance*30))

mesh = mesh.decimate(30000)


print(mesh.vertices.shape)
print(mesh.faces.shape)


def find_displacement(mesh, seperation):
    """find displacement which results in a minimum seperation distance"""
    tree = scipy.spatial.cKDTree(mesh.vertices)

    def objective(displacement):
        print(displacement)
        return tree.query(mesh.vertices + [0,0,displacement])[0].min() - seperation

    step = mesh.vertices.max() - mesh.vertices.min()
    step = scipy.optimize.root(objective, x0=step)
    return step.x

# repeat the mesh for cost efficiency
n_copies = 4
step = find_displacement(mesh, seperation=0.01)
meshes = [cg.Mesh(mesh.vertices+[0,0,step*i], mesh.faces) for i in range(n_copies)]
mesh = reduce(lambda x,y:x.merge(y), meshes)




escheresque.stl.save_STL(filename.format('dec'), mesh)
mesh.plot()