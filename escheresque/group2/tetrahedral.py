
from cached_property import cached_property

from pycomplex import synthetic
from escheresque.group2.group import Group


class Tetrahedral(Group):

    @cached_property
    def complex(self):
        return synthetic.n_simplex(n_dim=3).as_spherical().boundary

    @property
    def description(self):
        return 3, 2, 3, -1


class ChiralTetrahedral(Group):
    @property
    def description(self):
        return 3, 2, 3
