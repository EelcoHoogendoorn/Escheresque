
"""
various utility functions

"""

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, dia_matrix



def gather(idx, vals):
    """return vals[idx]. return.shape = idx.shape + vals.shape[1:]"""
    return vals[idx]
def scatter(idx, vals, target):
    """target[idx] += vals. """
    np.add.at(target.ravel(), idx.ravel(), vals.ravel())


def adjoint(A):
    """compute inverse without division by det; ...xv3xc3 input, or array of matrices assumed"""
    AI = np.empty_like(A)
    for i in xrange(3):
        AI[...,i,:] = np.cross(A[...,i-2,:], A[...,i-1,:])
    return AI

def null(A):
    """
    vectorized nullspace algrithm, for 3x3 rank-2 matrix
    simply cross each pair of vectors, and take the average
    """
    return adjoint(A).sum(axis=-2)


def inverse_transpose(A):
    """
    efficiently compute the inverse-transpose for stack of 3x3 matrices
    """
    I = adjoint(A)
    det = dot(I, A).mean(axis=-1)
    return I / det[...,None,None]

def inverse(A):
    """inverse of a stack of 3x3 matrices"""
##    return np.transpose( inverse_transpose(A), (0,2,1))
    return np.swapaxes( inverse_transpose(A), -1,-2)


def normals(triangles):
    """triangles is T x v3 x c3 array. output is T x c3 normal array"""
    edges = np.roll(triangles, -1, axis=1)-np.roll(triangles, +1, axis=1)
    return normalize(null(edges))

def dot(A, B):
    """dot arrays of vecs; contract over last indices"""
    return np.einsum('...i,...i->...', A, B)

def normalize(listofvec):
    """normalize an array of vecs"""
    norm = np.sqrt(dot(listofvec, listofvec))
    norm[norm==0] = 1
    return listofvec / np.expand_dims( norm, -1)



def coo_shift(M, r, c):
    """shift a coo matrix by a certain number of rows and colums"""
    return coo_matrix((M.data, (M.row+r, M.col+c)))
def coo_append(*M):
    """append coo matrices. no overlap assumed"""
    return coo_matrix((
        np.hstack([ m.data for m in M]),
        (
            np.hstack([m.row for m in M]),
            np.hstack([m.col for m in M]),
        )))
def dia_simple(diag):
    """construct a simple sparse diagonal matrix"""
    return dia_matrix((diag, 0), shape=(len(diag),)*2)




def rotation_axis_angle(axis, angle):
    """
    rotation matrix about a given axis and angle
    """

    angle = angle * np.pi / 180
    sa = np.sin(angle)
    ca = np.cos(angle)

    q = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    q=np.roll(q, axis, 0)
    q=np.roll(q, axis, 1)
    return q
