"""
stl handling module
"""
import numpy as np

from escheresque import util
import numpy_indexed as npi
from escheresque.computational_geometry import Mesh


def save_STL(filename, mesh):
    """save a mesh to plain stl. vertex ordering is assumed to be correct"""
    header      = np.zeros(80, '<c')
    triangles   = np.array(len(mesh.faces), '<u4')
    dtype       = [('normal', '<f4', 3,),('vertex', '<f4', (3,3)), ('abc', '<u2', 1,)]
    data        = np.empty(triangles, dtype)

    data['abc']    = 0     #standard stl cruft
    data['vertex'] = mesh.vertices[mesh.faces]
    data['normal'] = util.normalize(mesh.face_normals())

    with open(filename, 'wb') as fh:
        header.   tofile(fh)
        triangles.tofile(fh)
        data.     tofile(fh)


def load_stl(filename):
    dtype       = [('normal', '<f4', 3,),('vertex', '<f4', (3,3)), ('abc', '<u2', 1,)]

    with open(filename, 'rb') as fh:
        header    = np.fromfile(fh, '<c', 80)
        triangles = np.fromfile(fh, '<u4', 1)[0]
        data      = np.fromfile(fh, dtype, triangles)

    vertices, triangles = npi.unique(data['vertex'].reshape(-1, 3), return_inverse=True)
    return Mesh(vertices, triangles.reshape(-1, 3))



def save_STL_complete(complex, radius, filename):
    """
    split this as mesh_from_datamodel

    save a mesh to binary STL format
    the number of triangles grows quickly
    shapeway and solidworks tap out at a mere 1M and 20k triangles respectively...
    or 100k for sw surface
    """
    data        = np.empty((complex.group.index, complex.group.order, complex.topology.P2, 3, 3), np.float)

    #essence here is in exact transformations given by the basis trnasforms. this gives a guaranteed leak-free mesh
    PP = complex.geometry.decomposed
    FV = complex.topology.FV
    for i,B in enumerate(complex.group.basis):
        for t, b in enumerate(B.reshape(-1,3,3)):
            b = util.normalize(b.T).T                      #now every row is a normalized vertex
            P = np.dot(b, PP.T).T * radius[:,i][:, None]   #go from decomposed coords to local coordinate system
            fv = FV[:,::np.sign(np.linalg.det(b))]
            data[i,t] = util.gather(fv, P)

    save_STL(filename, data.reshape(-1,3,3))


def save_OFF(filename, vertices, triangles):
    header = r'OFF BINARY\n'

    n_triangles = len(triangles)
    n_vertices = len(vertices)

    vertices = vertices.astype('>f32')
    faces = np.zeros((n_triangles, 5), dtype='>i32')
    faces[:, 0] = 3
    faces[:, 1:4] = triangles
    counts = np.array([n_vertices, n_triangles, 0], dtype='>i32')

    with open(filename, 'wb') as fh:
        fh.write(header)
        counts.tofile(fh)
        vertices.tofile(fh)
        faces.tofile(fh)




