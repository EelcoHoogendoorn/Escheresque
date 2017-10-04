
from cached_property import cached_property

from escheresque.group2.group import TriangleGroup, SubGroup


class IcosahedralFull(TriangleGroup):

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


# FIXME: dihedral on icosahedral is a pain without ability to refer to specific elements in presentation
# obviously dihedral exists, but need a two-fold rotation about a specific edge
# class Dihedral5(IcosahedralSubGroup):
#     @property
#     def description(self):
#         return -10, 2, 1
#
# class ChiralDihedral5(IcosahedralSubGroup):
#     @property
#     def description(self):
#         return -10, 2, 1, -1
#
# class Dihedral3(IcosahedralSubGroup):
#     """this order-6 group appears unattainable"""
#     @property
#     def description(self):
#         return 1, 2, -6, -1
#
# class ChiralDihedral3(IcosahedralSubGroup):
#     @property
#     def description(self):
#         return 1, 2, 3



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

