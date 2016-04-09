

"""
subdivision curve module
"""

import numpy as np

from escheresque.util import normalize


class Curve(object):
    """
    subdivision curve
    precomputes weights over a basis

    do subdivision in tcr space
    """
    #4 x n; tangent, cotangent radius and tension
##    control = np.array([[0,1], [0,0], [1,1], [0,0]], dtype=np.float).T
##    d = 0.05
##    control = np.array([[0,0.25, 0.5, 0.75,1], [0,d,0,-d*2,0], [1,1,1,1,1], [0,0,0,0,0]], dtype=np.float).T
    @property
    def tangent(self):
        return self.control[:,0]
    @property
    def cotangent(self):
        return self.control[:,1]
    @property
    def radius(self):
        return self.control[:,2]
    @property
    def coords(self):
        return self.control[:,0:3]
    @property
    def tension(self):
        return self.control[:,3]

    def __init__(self, edge):
##        q=4
##        self.control = np.array([np.linspace(0, 1, q), [0]*q, [1]*q, [0]*q], dtype=np.float).T
        self.control = np.array([np.linspace(0, 1, 3), [0,0.2,0], [1]*3, [0]*3], dtype=np.float).T
##        self.control = np.array([np.linspace(0, 1, 3), [0,0.2,0], [1]*3, [0]*3], dtype=np.float).T

        self.edge = edge        #references a topological edge
        self.update()

    def update(self):
        """
        update weights, given the control polygon
        """

    def apply(self, tcr):
        """
        apply the edge basis to the abstract subdivided curve
        output is mirror x rotations x subpoints x 3
        """
        t, c, r = tcr.T     #length equal to subpoints
        t = t[None, None, :, None]
        c = c[None, None, :, None]
        r = r[None, None, :, None]

        L, R, C = self.edge.basis()     #edges x 3
##        print L.shape, 'what?'
        L = L[:,:,None,:]
        R = R[:,:,None,:]
        C = C[:,:,None,:]

        P = L*t + R*(1-t) + C*c
        return normalize(P) * r

    def match(self, edge, point, pos):
        """match the underlying spline coords to the given world coord position"""

        L, R, C = self.edge.basis()     #edges x 3
        L = L[edge]
        R = R[edge]
        C = C[edge]

        #project normalized point back into plane
        # (p*scale - M ) . M = 0
        # scale = M.M / p.M
        M = (L+R)/2
        q = pos * np.dot(M,M) / np.dot(pos, M)

        T = np.array((L,R,C)).T
        f = np.linalg.solve(T, q)

        t = f[0]
        c = f[2]
        r = np.linalg.norm(pos)

        self.coords[point] = t,c,r


    def controlpoints(self):
        """return control points. just transform the coords"""
        return self.apply(self.coords)
    def controlmesh(self, level=3):
        """return control mesh"""
        return self.apply(self.Core(3, True))
    def curve(self):
        """compute the curve itself"""
        return self.apply(self.Core(6, False))


    def Refine(self, P, w):
        t = w.copy()
        t[t>1] = 1      #tension of one is fully linear interpolation

        #cubic approximating scheme: [4/8,4/8], [1/8,6/8,1/8]
        #odd vertex is blended towards a linear scheme, [0,1,0]
        #with a linear weighting between these two based on t
        a = 1-t[1:-1, np.newaxis]
        Pv = (a*P[0:-2,:] + (8-2*a) * P[1:-1,:] + a*P[2:,:]) / 8
        Pe = (P[0:-1,:] + P[1:,:]) / 2

        Pn = np.zeros((len(Pv)+len(Pe)+2, 3))
        Pn[0     ,:] = P[0,:]
        Pn[1::2  ,:] = Pe
        Pn[2:-2:2,:] = Pv
        Pn[-1    ,:] = P[-1,:]

        #propagation of weights
        wv = w
        wn = np.zeros((len(wv)*2-1))
        wn[0::2] = wv - 1
        wn[wn<0] = 0

        return Pn, wn


    def Core(self, N, control=False):
        #perform refinement
        P = self.coords
        w = self.tension
        if control: w = np.ones_like(w)
        w = w * N

        for i in xrange(N):
            P, w = self.Refine(P, w)
        return P


    def add_cp(self, l, r):
        new = (self.control[l]+self.control[r]) / 2
        self.control = np.insert(self.control, max(l,r), new, axis=0)
    def remove_cp(self, index):
        self.control = np.delete(self.control, index, axis=0)


