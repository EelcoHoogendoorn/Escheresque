
from cached_property import cached_property

from escheresque.group2.group import Group, SubGroup


class IcosahedralFull(Group):

    @cached_property
    def complex(self):
        from pycomplex import synthetic
        return synthetic.icosahedron()


class IcosahedralSubGroup(SubGroup):
    @cached_property
    def group(self):
        return IcosahedralFull()




class Icosahedral(IcosahedralSubGroup):
    @property
    def description(self):
        return 5, 2, 3, -1


class ChiralIcosahedral(IcosahedralSubGroup):
    @property
    def description(self):
        return 5, 2, 3


class Dihedral5(IcosahedralSubGroup):
    @property
    def description(self):
        return -5, 1, 1

class ChiralCyclic5(IcosahedralSubGroup):
    @property
    def description(self):
        return -5, 1, 1, -1


class Dihedral3(IcosahedralSubGroup):
    """this order-6 group appears unattainable"""
    @property
    def description(self):
        return 1, 1, 3, -1

class ChiralDihedral3(IcosahedralSubGroup):
    @property
    def description(self):
        return 1, 1, 3


class Dihedral2(IcosahedralSubGroup):
    @property
    def description(self):
        return 1, 2, 1, -1

class ChiralDihedral2(IcosahedralSubGroup):
    @property
    def description(self):
        return 1, 2, 1


class Null(IcosahedralSubGroup):
    @property
    def description(self):
        return 1, 1, 1
