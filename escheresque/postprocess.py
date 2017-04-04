import numpy as np
import scipy.spatial
import scipy.optimize

import escheresque.stl
import escheresque.computational_geometry as cg



filename = r'..\data\part{0}.stl'

mesh = escheresque.stl.load_stl(filename.format(0))


# normalize orientation
u, s, v = np.linalg.svd(mesh.vertices, full_matrices=0)
mesh.vertices = mesh.vertices.dot(v[:, ::1])
mesh.vertices *= 100
print(mesh.volume())

mesh = mesh.decimate(30000)

escheresque.stl.save_STL(filename.format('dec'), mesh)
quit()

def vispy_plot():

    from vispy import app, scene, io

    # Prepare canvas
    canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)
    canvas.measure_fps()

    # Set up a viewbox to display the image with interactive pan/zoom
    view = canvas.central_widget.add_view()

    # from escheresque.group.octahedral import Pyritohedral
    # group = Pyritohedral()

    for i in range(12):
        mesh = escheresque.stl.load_stl(filename.format(i))

        meshvis = scene.visuals.Mesh(
            mesh.vertices * 100,
            mesh.faces[:, ::-1],
            shading='flat',
            parent=view.scene)

    # Create three cameras (Fly, Turntable and Arcball)
    fov = 60.
    cam1 = scene.cameras.FlyCamera(parent=view.scene, fov=fov, name='Fly')
    view.camera = cam1

    app.run()

# vispy_plot()


print(mesh.vertices.shape)
print(mesh.faces.shape)
assert mesh.is_orientated()


print('volume', mesh.volume()*40**3 * 12)


seed = np.zeros_like(mesh.vertices[:, 0])
#seed[np.argmax(mesh.vertices[:,0])] = 1
seed[np.argmin(mesh.vertices[:,2])] = 1
distance = mesh.geodesic(seed)
# mesh.plot(np.cos(distance*30))

mesh = mesh.decimate(30000)


print(mesh.vertices.shape)
print(mesh.faces.shape)

import util
dir = util.normalize([[-0.3, 0, 1]])[0]

def find_displacement(mesh, seperation):
    """find displacement which results in a minimum seperation distance"""
    tree = scipy.spatial.cKDTree(mesh.vertices)

    def objective(displacement):
        print(displacement)
        return tree.query(mesh.vertices + dir*displacement)[0].min() - seperation

    step = mesh.vertices.max() - mesh.vertices.min()
    step = scipy.optimize.root(objective, x0=step)
    return step.x

# repeat the mesh for cost efficiency
n_copies = 4
step = find_displacement(mesh, seperation=0.015)
scale = 32
meshes = [cg.Mesh(mesh.vertices+dir*step*i, mesh.faces) for i in range(n_copies)]
mesh = reduce(lambda x,y:x.merge(y), meshes)
mesh.vertices *= scale




escheresque.stl.save_STL(filename.format('dec'), mesh)
mesh.plot()