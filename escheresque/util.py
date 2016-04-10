
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


def rotation_matrix(angles):
    """
    empirical knowledge:angles are in-order: XYZ
    """
##    axisorder = [0,1,2]
##    axissign = [+1,+1,+1]
##    mulorder = [0,1,2]

    angles = angles * np.pi / 180
##    Sx,Sy,Sz = np.sin(angles)
##    Cx,Cy,Cz = np.cos(angles)
    S = np.sin(angles)
    C = np.cos(angles)
    CS = np.array((C,S)).T
##    print CS
##    quit()

    def primitive(axis, ca, sa):
##        q = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
        q = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
        q=np.roll(q, axis, 0)
        q=np.roll(q, axis, 1)
        return q

##    Q  = np.array([primitive(o, *cs) for o,cs in zip(axisorder, CS)])
##    yaw   = primitive(, 0).T
##    pitch = primitive(Cy, Sy, 1)
##    roll  = primitive(Cz, Sz, 2).T
##
##    pitch = primitive(Cx, Sx, 0).T
##    roll  = primitive(Cy, Sy, 1)
##    yaw   = primitive(Cz, Sz, 2).T

    def multiply(a,b,c):
        return np.dot(np.dot(a, b), c)
##        return np.dot(a, np.dot(b, c))


##    return multiply(*Q[mulorder])

##    X = [primitive(0, *cs) for cs in CS]
##    Y = [primitive(1, *cs) for cs in CS]
##    Z = [primitive(2, *cs) for cs in CS]

    X,Y,Z = [primitive(i, *cs) for i,cs in enumerate(CS)]
    return multiply(Z, X, Y)



    """http://en.wikipedia.org/wiki/Euler_angles    """
    #proper euler
    XZX = multiply(X[0], Z[1], X[2])
    XYX = multiply(X[0], Y[1], X[2])#
    YXY = multiply(Y[0], X[1], Y[2])
    YZY = multiply(Y[0], Z[1], Y[2])
    ZYZ = multiply(Z[0], Y[1], Z[2])#
    ZXZ = multiply(Z[0], X[1], Z[2])
    #tait-byran
    XZY = multiply(X[0], Z[1], Y[2])
    XYZ = multiply(X[0], Y[1], Z[2])#
    YXZ = multiply(Y[0], X[1], Z[2])
    YZX = multiply(Y[0], Z[1], X[2])
    ZYX = multiply(Z[0], Y[1], X[2])#
    ZXY = multiply(Z[0], X[1], Y[2])
    return ZYX


    #xyz order
    M = np.empty((3,3))
    if False:
        M[0][0]=Cy*Cz;
        M[0][1]=-Cy*Sz;
        M[0][2]=Sy;
        M[1][0]=Cz*Sx*Sy+Cx*Sz;
        M[1][1]=Cx*Cz-Sx*Sy*Sz;
        M[1][2]=-Cy*Sx;
        M[2][0]=-Cx*Cz*Sy+Sx*Sz;
        M[2][1]=Cz*Sx+Cx*Sy*Sz;
        M[2][2]=Cx*Cy;
    if True:#
        M[0][0]=Cy*Cz;
        M[0][1]=Sx*Sy-Cx*Cy*Sz;
        M[0][2]=Cx*Sy+Cy*Sx*Sz;
        M[1][0]=Sz;
        M[1][1]=Cx*Cz;
        M[1][2]=-Cz*Sx;
        M[2][0]=-Cz*Sy;
        M[2][1]=Cy*Sx+Cx*Sy*Sz;
        M[2][2]=Cx*Cy-Sx*Sy*Sz;

    if False:
        M[0][0]=Cy*Cz-Sx*Sy*Sz;
        M[0][1]=-Cx*Sz;
        M[0][2]=Cz*Sy+Cy*Sx*Sz;
        M[1][0]=Cz*Sx*Sy+Cy*Sz;
        M[1][1]=Cx*Cz;
        M[1][2]=-Cy*Cz*Sx+Sy*Sz;
        M[2][0]=-Cx*Sy;
        M[2][1]=Sx;
        M[2][2]=Cx*Cy;
    if False:
        M[0][0]=Cy*Cz;
        M[0][1]=Cz*Sx*Sy-Cx*Sz;
        M[0][2]=Cx*Cz*Sy+Sx*Sz;
        M[1][0]=Cy*Sz;
        M[1][1]=Cx*Cz+Sx*Sy*Sz;
        M[1][2]=-Cz*Sx+Cx*Sy*Sz;
        M[2][0]=-Sy;
        M[2][1]=Cy*Sx;
        M[2][2]=Cx*Cy;
    if False:#
        M[0][0]=Cy*Cz+Sx*Sy*Sz;
        M[0][1]=Cz*Sx*Sy-Cy*Sz;
        M[0][2]=Cx*Sy;
        M[1][0]=Cx*Sz;
        M[1][1]=Cx*Cz;
        M[1][2]=-Sx;
        M[2][0]=-Cz*Sy+Cy*Sx*Sz;
        M[2][1]=Cy*Cz*Sx+Sy*Sz;
        M[2][2]=Cx*Cy;
    if False:
        M[0][0]=Cy*Cz;
        M[0][1]=-Sz;
        M[0][2]=Cz*Sy;
        M[1][0]=Sx*Sy+Cx*Cy*Sz;
        M[1][1]=Cx*Cz;
        M[1][2]=-Cy*Sx+Cx*Sy*Sz;
        M[2][0]=-Cx*Sy+Cy*Sx*Sz;
        M[2][1]=Cz*Sx;
        M[2][2]=Cx*Cy+Sx*Sy*Sz;



    return M
