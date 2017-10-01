
from cached_property import cached_property

from pycomplex import synthetic
from escheresque.group2.group import Group


class Icosahedral(Group):

    @cached_property
    def complex(self):
        return synthetic.icosahedron()

    @property
    def description(self):
        return 5, 2, 3, -1


class ChiralIcosahedral(Icosahedral):

    @property
    def description(self):
        return 5, 2, 3


class Null(Icosahedral):
    @property
    def description(self):
        return 1, 1, 1
