
from cached_property import cached_property

from escheresque.group2.group import PolyhedralGroup, SubGroup


class OctahedralFull(PolyhedralGroup):

    @cached_property
    def complex(self):
        from pycomplex import synthetic
        return synthetic.n_cube_dual(n_dim=3)


class OctahedralSubGroup(SubGroup):

    @cached_property
    def group(self):
        return OctahedralFull()


class Octahedral(OctahedralSubGroup):

    @property
    def description(self):
        return 4, 2, 3, -1


class ChiralOctahedral(OctahedralSubGroup):

    @property
    def description(self):
        return 4, 2, 3


class Pyritohedral(OctahedralSubGroup):
    """
    This is the symmetry group used by Eschers 'Angels and Demons'

    It contains a nontrivial subset of both rotation and mirror symmetries
    of its parent symmetry group. It is unique in this regard amongst all subgroups.
    """

    @property
    def description(self):
        return 2, 1, 3, -1


class ChiralTetrahedral(OctahedralSubGroup):
    """subgroup of order 12, index 4"""
    @property
    def description(self):
        return 2, 1, 3


class Dihedral2(OctahedralSubGroup):
    """subgroup of order 16, index 3"""
    @property
    def description(self):
        return 2, 2, 1, -1


class ChiralDihedral2(OctahedralSubGroup):
    """subgroup of order 8, index 6"""
    @property
    def description(self):
        return 2, 2, 1


class Origin(OctahedralSubGroup):
    @property
    def description(self):
        return 1, 1, 1, -1


class Plane(OctahedralSubGroup):
    @property
    def description(self):
        return -1, 1, 1


class TriPlane(OctahedralSubGroup):
    @property
    def description(self):
        return -1, -1, 1


class Null(OctahedralSubGroup):
    @property
    def description(self):
        return 1, 1, 1

