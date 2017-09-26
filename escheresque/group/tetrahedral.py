"""
tetradal symmetry group

we have no origin reflection subgroup; this would only be possible
by allowing fundamental points of a different kind to be mapped together,
which is not a desirable generalization
"""


import numpy as np
import itertools

from escheresque.group.group import *


class Tetrahedral(Group):
    """
    Full Tetrahedral symmetry
    """
    index = 1
    order = 24
    mirrors = 2

    def geometry(self):
        self.primal = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])
        self.edges = np.array(list(itertools.combinations(range(4), 2)))
        self.faces = np.array(list(itertools.combinations(range(4), 3)))

    def fundamental_domain(self):
        return self.full()

##    symmetries = [Symmetry('3', (0,0,1)), Symmetry('3', (1,0,0)), Symmetry('2', (0,1,0)),
##                  Symmetry('m', (0,1,1)), Symmetry('m', (1,0,1)), Symmetry('m', (1,1,0))]


class ChiralTetrahedral(Tetrahedral):
    """
    Chiral Tetrahedral symmetry
    This is the symmetry group used by Escher's 'Sphere with Fish'
    This group is an interesting one, for having a very large fundamental domain.
    This makes that the curvature of the sphere plays a large role in how the tiles fit together
    """

    index = 2
    order = 12
    mirrors = 1

    def fundamental_domain(self):
        return self.chiral()

##    symmetries = [Symmetry('3', (0,0,1)), Symmetry('3', (1,0,0)), Symmetry('2', (0,1,0))]



class Null(Tetrahedral):
    """
    Null symmetry
    Finding a good tessellation on this 'symmetry group',
    is probably the biggest challenge of all.
    """
    index = 24
    order = 1
    mirrors = 1

    def fundamental_domain(self):
        return self.null()


if False:
    group = ChiralTetrahedral()
##    group._vertices()
    print(group.vertices)
    quit()
