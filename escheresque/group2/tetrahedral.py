
from cached_property import cached_property

from escheresque.group2.group import Group, SubGroup


class TetrahedralFull(Group):

    @cached_property
    def complex(self):
        from pycomplex import synthetic
        return synthetic.n_simplex(n_dim=3).as_spherical().boundary


class TetrahedralSubGroup(SubGroup):

    @cached_property
    def group(self):
        return TetrahedralFull()


class Tetrahedral(TetrahedralSubGroup):
    @property
    def description(self):
        return 3, 2, 3, -1


class ChiralTetrahedral(TetrahedralSubGroup):
    @property
    def description(self):
        return 3, 2, 3
