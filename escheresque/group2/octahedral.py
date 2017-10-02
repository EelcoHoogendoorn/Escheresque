
from cached_property import cached_property

import numpy as np
from pycomplex import synthetic

from escheresque.group2.group import Group


class Octahedral(Group):

    @cached_property
    def complex(self):
        return synthetic.n_cube_dual(n_dim=3)

    @property
    def description(self):
        return 4, 2, 3, -1


class ChiralOctahedral(Octahedral):

    @property
    def description(self):
        return 4, 2, 3


class Pyritohedral(Octahedral):
    """
    This is the symmetry group used by Eschers 'Angels and Demons'

    It contains a nontrivial subset of both rotation and mirror symmetries
    of its parent symmetry group. It is unique in this regard amongst all subgroups.
    """

    @property
    def description(self):
        return 2, 1, 3, -1


class ChiralTetrahedral(Octahedral):
    """
    """

    @property
    def description(self):
        return 2, 1, 3


class Dihedral2(Octahedral):
    """subgroup of order 16, index 3"""

    @property
    def description(self):
        return 2, 2, 1, -1


class ChiralDihedral2(Octahedral):
    """subgroup of order 8, index 6"""

    @property
    def description(self):
        return 2, 2, 1


class Origin(Octahedral):
    @property
    def description(self):
        return 1, 1, 1, -1


class Null(Octahedral):
    @property
    def description(self):
        return 1, 1, 1
