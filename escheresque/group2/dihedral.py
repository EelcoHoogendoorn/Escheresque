
import numpy as np
from cached_property import cached_property

from escheresque.group2.group import TriangleGroup, SubGroup


class DihedralFull(TriangleGroup):

    """This isnt a true polyhedral group; The hosohedron needs some special treatment"""

    def __init__(self, n):
        self.n = n

    @cached_property
    def vertices(self):
        a, s = np.linspace(0, 2 * np.pi, self.n, retstep=True, endpoint=False)
        p = [0, 0, -1], [0, 0, +1]
        m = np.array([np.cos(a+s/2), np.sin(a+s/2), np.zeros_like(a)]).T
        d = np.array([np.cos(a), np.sin(a), np.zeros_like(a)]).T
        return [np.array(v, np.float64) for v in p, m, d]

    @cached_property
    def triangles(self):
        return np.array([(s, i, (i + f) % self.n) for i in range(self.n) for s in (0, 1) for f in (0, 1)])

    @cached_property
    def complex(self):
        """Return the complex specifying the triangular spherical symmetry"""
        offset = np.cumsum([0] + self.n_elements)[:-1]
        from pycomplex.complex.simplicial.spherical import ComplexSpherical2
        return ComplexSpherical2(
            vertices=np.concatenate(self.vertices, axis=0),
            simplices=self.triangles + offset
        )


class DihedralSubGroup(SubGroup):
    def __init__(self, n):
        self.n = n

    @cached_property
    def group(self):
        return DihedralFull(self.n)


class ChiralDihedral(DihedralSubGroup):
    @property
    def description(self):
        return self.n, 2, 1

class Cyclic(DihedralSubGroup):
    @property
    def description(self):
        return self.n, 1, 1
