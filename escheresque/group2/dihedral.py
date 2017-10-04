
from cached_property import cached_property

from escheresque.group2.group import Group, SubGroup


class DihedralFull(Group):

    @cached_property
    def complex(self):
        """This isnt actually a polyhedral group; need to override properties that otherwise derive from polyhedron;
        fundamental_domains and vertices

        Also, we need an n constructor argument here
        """
        raise NotImplementedError


class DihedralSubGroup(SubGroup):

    @cached_property
    def group(self):
        """need n constructor argument here"""
        return DihedralFull()

