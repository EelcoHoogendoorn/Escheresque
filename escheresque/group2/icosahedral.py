
from cached_property import cached_property

from escheresque.group2.group import PolyhedralGroup, SubGroup


class IcosahedralFull(PolyhedralGroup):
    @cached_property
    def polyhedron(self):
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


class Pyritohedral(IcosahedralSubGroup):
    @property
    def description(self):
        """Can be described as a mirror plane and a 3-fold rotation"""
        return 1, -1, (2, 3), -1    # without the (2) to select an alternative triangle, this represents the full group


class Cyclic5(IcosahedralSubGroup):
    @property
    def description(self):
        return -10, 1, 1

class ChiralCyclic5(IcosahedralSubGroup):
    @property
    def description(self):
        return -10, 1, 1, -1

class Cyclic3(IcosahedralSubGroup):
    @property
    def description(self):
        return 1, 1, -6

class ChiralCyclic3(IcosahedralSubGroup):
    @property
    def description(self):
        return 1, 1, 3

class Cyclic2(IcosahedralSubGroup):
    @property
    def description(self):
        return 1, 2, 1, -1

class ChiralCyclic2(IcosahedralSubGroup):
    @property
    def description(self):
        return 1, 2, 1


class Dihedral5(IcosahedralSubGroup):
    @property
    def description(self):
        return 5, (20, 2), 1, -1

class ChiralDihedral5(IcosahedralSubGroup):
    @property
    def description(self):
        return 5, (20, 2), 1

class Dihedral3(IcosahedralSubGroup):
    @property
    def description(self):
        return 1, (22, 2), 3, -1

class ChiralDihedral3(IcosahedralSubGroup):
    @property
    def description(self):
        return 1, (22, 2), 3

class Dihedral2(IcosahedralSubGroup):
    @property
    def description(self):
        return 1, -1, 1, -1
#
# class ChiralDihedral2(IcosahedralSubGroup):
#     @property
#     def description(self):
#         return 1, -2, 1



class Origin(IcosahedralSubGroup):
    @property
    def description(self):
        return 1, 1, 1, -1


class Plane(IcosahedralSubGroup):
    @property
    def description(self):
        return 1, -1, 1


class Null(IcosahedralSubGroup):
    @property
    def description(self):
        return 1, 1, 1

