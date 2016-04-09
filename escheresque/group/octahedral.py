"""
octahedral groups
"""

import numpy as np
import itertools

from escheresque.group.group import *


class Octahedral(Group):
    """
    Full Octahedral symmetry
    """
    index = 1
    order = 48
    mirrors = 2

    def geometry(self):
        self.primal = np.vstack([np.roll([(1,0,0),(-1,0,0)],i,axis=1) for i in range(3)])
        self.edges = np.array([
            e for e in itertools.combinations(range(6), 2)
                if np.count_nonzero(self.primal[list(e)].sum(0))==2 ])
        self.faces = np.array([
            f for f in itertools.combinations(range(6), 3)
                if np.count_nonzero(self.primal[list(f)].sum(0))==3 ])

    def fundamental_domain(self):
        return self.full()

##    symmetries = [Symmetry('4', (1,0,0)), Symmetry('2', (0,1,0)), Symmetry('3', (0,0,1)),
##                  Symmetry('m', (0,1,1)), Symmetry('m', (1,0,1)), Symmetry('m', (1,1,0))]


class ChiralOctahedral(Octahedral):
    """
    Chiral Octohedral symmetry
    """
    index = 2
    order = 24
    mirrors = 1

    def fundamental_domain(self):
        return self.chiral()

##    symmetries = [Symmetry('4', (1,0,0)), Symmetry('2', (0,1,0)), Symmetry('3', (0,0,1))]


class Pyritohedral(Octahedral):
    """
    This is the symmetry group used by Eschers 'Angels and Demons'

    It contains a nontrivial subset of both rotation and mirror symmetries,
    of its parent symmetry group. It is unique in this regard.
    """

    index = 2
    order = 24
    mirrors = 2

    def fundamental_domain(self):
        """We need to find consistent orientation on the cube face. the roll statement encodes that logic"""
        b = self.basis_from_domains(self.domains)
        p,m = b[:,:,0], b[:,:,1]
        return (p * np.roll(m, 1, axis=1)).sum(axis=1) == 0

##    symmetries = [Symmetry('2', (1,0,0)),                       Symmetry('3', (0,0,1)),
##                                                                Symmetry('m', (1,1,0))]

##Pyritohedral()
##quit()
class ChiralTetrahedral(Octahedral):
    """
    Chiral Tetrahedral symmetry, on a Octahedral basis
    """
    index = 4
    order = 12
    mirrors = 1

    def fundamental_domain(self):
        """We need to mark a consistent orientation on the cube face. the roll statement encodes that logic"""
        b = self.basis_from_domains(self.domains)
        p,m = b[:,:,0], b[:,:,1]
        return self.chiral() + ((p * np.roll(m, 1, axis=1)).sum(axis=1) == 0)

##    symmetries = [Symmetry('m', (1,1,0)), Symmetry('3', (0,0,1)), Symmetry('2', (1,0,0))]


"""
some simple symmetries
"""

class Origin(Octahedral):
    """
    A single reflection through the origin
    """
    index = 24
    order = 2
    mirrors = 2

    def fundamental_domain(self):
        return self.origin()

class Null(Octahedral):
    """
    Null symmetry
    Finding a good tessellation on this 'symmetry group',
    is probably the biggest challenge of all.
    """
    index = 48
    order = 1
    mirrors = 1

    def fundamental_domain(self):
        return self.null()


