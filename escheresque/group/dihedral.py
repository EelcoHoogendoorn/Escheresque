

"""
dihedral symmetry group



this is a funny one, since it is parametrizable

how does it fit in with regular domain generation?

smaller N is treated with subtiles of a bigger N, to avoid degenerate geometry?

for now, just include a few examples; nothing exhaustive
"""


import numpy as np
import itertools

from escheresque.group.group import *


class Dihedral(Group):

    """
    This class of symmetry groups is exceptional,
    in the sense that they have a free parameter,
    denoting the number of rotations around the two vertices
    """
    index = '1'
    order = '4n'
    mirrors = '2'

    def __init__(self, N=2):
        self.N = N
        assert(N>1)
        self.order = eval(self.order.replace('n','*'+str(N)))
        self.index = eval(self.index.replace('n','*'+str(N)))
        self.mirrors = eval(self.mirrors.replace('n', str(N)))
        super(Dihedral, self).__init__()

    def geometry(self):
        def wrap(i): return (i%self.N)
        R =np.arange(self.N)

        #primal and dual should be the other way around; but averaging and domain generation will not work otherwise
        self.primal = np.array([[0,np.cos(a), np.sin(a)] for a in np.linspace(0, np.pi*2, self.N, False)])
        self.dual   = np.array([[1,0,0],[-1,0,0]])
        ring = np.array([(r, wrap(r+1)) for r in R])
        self.mid = np.array([[0,np.cos(a), np.sin(a)] for a in np.linspace(0, np.pi*2, self.N, False)+np.pi/self.N])


        #indices of primal verts
        self.edges = ring
        self.faces = ring.T



    def fundamental_domain(self):
        return self.full()





class ChiralDihedral(Dihedral):
    """
    Chiral Dihedral group
    """
    index = '2'
    order = '2n'
    mirrors = '1'

    def fundamental_domain(self):
        return self.chiral()

class Origin(Dihedral):
    """
    A single reflection through the origin
    This only works for even-valued N, and will give rise to an error otherwise
    As a matter of fact, an origin reflection should be possible on uneven dihedral groups too,
    but only if we allow fundamental points of a different kind to be mapped to one another.
    It is merely a matter of 'coincidence' that domains with unmatching fundamental points
    map to the same region under this transform.
    The same considerations are why there is no origin reflection on the tetrahedral symmetry group
    """
    index = '2n'
    order = '2'
    mirrors = '2'

    def fundamental_domain(self):
        assert self.N % 2 == 0
        return self.origin()

class Null(Dihedral):
    """
    Null symmetry
    Finding a good tessellation on this 'symmetry group',
    is probably the biggest challenge of all.
    """
    index = '4n'
    order = '1'
    mirrors = '1'

    def fundamental_domain(self):
        return self.null()
