"""
stl handling module

and stl load module?
"""


import numpy as np
from itertools import izip

from . import util


def save_STL(filename, P):
    """save a triangles x vertex3 x dim3 array to plain stl. vertex ordering is assumed to be correct"""
    header      = np.zeros(80, '<c')
    triangles   = np.array(len(P), '<u4')
    dtype       = [('normal', '<f4', 3,),('vertex', '<f4', (3,3)), ('abc', '<u2', 1,)]      #use struct array for memory layout
    data        = np.empty(triangles, dtype)

    data['abc']    = 0     #standard stl cruft
    data['vertex'] = P
    data['normal'] = util.normalize(np.cross(P[:,1,:]-P[:,0,:],P[:,2,:]-P[:,0,:]))

    with open(filename, 'wb') as fh:
        header.   tofile(fh)
        triangles.tofile(fh)
        data.     tofile(fh)

    print 'saved {t} triangles to plain STL'.format(t=triangles)


def save_STL_complete(complex, radius, filename):
    """
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




