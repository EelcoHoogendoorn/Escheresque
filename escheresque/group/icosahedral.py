"""
icosahedron. how to generate faces? find triplet of edges that share all vertices?

could implement a chiral dihedral(5) subgroup here
otherwise, i think thats it, no?
nope; chiral 222 is possible too
chiral dihedral 3 too
in other words; a chiral dihedral for every fundamental point

then of course also trivial subgroups of those

"""

import numpy as np
import itertools

from escheresque.group.group import *


class Icosahedral(Group):
    """
    Full Icosahedral symmetry
    This symmetry group has the highest order of all
    """
    index = 1
    order = 120
    mirrors = 2

    def geometry(self):

        phi = (1+np.sqrt(5))/2
        Q = np.array( list( itertools.product([0], [-1,+1], [-phi, +phi])))
        self.primal = np.vstack([np.roll(Q, r, axis=1) for r in range(3)])

        self.edges = np.array([
            e for e in itertools.combinations(range(12), 2)
                if np.linalg.norm( np.diff( self.primal[list(e)], axis=0)[0]) < 3 ])

        def verts(e3):
            return set(v for e in e3 for v in self.edges[e])

        self.faces = np.array([tuple(verts(e3)) for e3 in itertools.combinations(range(30), 3) if len(verts(e3))==3])

    def fundamental_domain(self):
        return self.full()



class ChiralIcosahedral(Icosahedral):
    """
    Chiral Icosahedral symmetry
    """
    index = 2
    order = 60
    mirrors = 1

    def fundamental_domain(self):
        return self.chiral()




class Origin(Icosahedral):
    """
    A single reflection through the origin
    """
    index = 60
    order = 2
    mirrors = 2

    def fundamental_domain(self):
        return self.origin()

class Null(Icosahedral):
    """
    Null symmetry
    Finding a good tessellation on this 'symmetry group',
    is probably the biggest challenge of all.
    """
    index = 120
    order = 1
    mirrors = 1

    def fundamental_domain(self):
        return self.null()
