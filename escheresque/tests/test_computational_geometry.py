import numpy as np
import scipy.spatial

import escheresque.computational_geometry as cg
from escheresque import util


def test_orientation():
    triangles = np.arange(3)[::-1].reshape(1,3)
    inner = np.eye(3)
    outer = np.eye(3) + 1
    mesh = cg.Mesh(outer, triangles).extrude(inner)
    assert mesh.is_orientated()


def test_grid():
    """test that we get a simple 4-point stencil on a regular grid"""
    axis = np.arange(10)
    p = np.array(np.meshgrid(axis, axis)).T.reshape(-1, 2)
    t = scipy.spatial.Delaunay(p)
    mesh = cg.Mesh(np.concatenate([p, np.zeros_like(p[:,0:1])],axis=1), t.simplices)
    L = mesh.laplacian_vertex()

    field = (p**2).sum(axis=1)
    grad = mesh.compute_gradient(field)
    div = mesh.compute_divergence(grad)
    assert L.data.max() == 4

    seed = np.zeros_like(mesh.vertices[:, 0])
    seed[0] = 1
    distance = mesh.geodesic(seed)
    mesh.plot(color=distance)
    print(distance.max())


def test_sphere():
    """some tests on a sphere"""
    vertices = util.normalize(np.random.normal(0, 1, (3000, 3)))
    mesh = cg.Mesh(vertices, scipy.spatial.ConvexHull(vertices).simplices)

    seed = np.zeros_like(mesh.vertices[:, 0])
    seed[0] = 1
    distance = mesh.geodesic(seed)
    mesh.plot(color=distance)
    print(distance.max())


def test_triangulation():

    #random points on the sphere
    points = util.normalize(np.random.randn(10000,3))

    #build curve. add sharp convex corners, as well as additional cuts
    N = 267#9
    radius = np.cos( np.linspace(0,np.pi*2*12,N, False)) +1.1
    curve = np.array([(np.cos(a)*r,np.sin(a)*r,1) for a,r in zip( np.linspace(0,np.pi*2,N, endpoint=False), radius)])
    curve = np.append(curve, [[1,0,-4],[-1,0,-4]], axis=0)      #add bottom slit
    curve = util.normalize(curve)
    curve = cg.Curve(curve)
#    print curve

    #do triangulation
    mesh, curve = cg.triangulate(points, curve)
    #add partitioning of points here too?
    partitions = mesh.partition(curve)
