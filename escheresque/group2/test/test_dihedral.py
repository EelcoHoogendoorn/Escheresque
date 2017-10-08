
import numpy as np
import numpy.testing as npt
import numpy_indexed as npi

from escheresque.group2 import dihedral

from pycomplex.math import linalg


def test_full():
    di = dihedral.DihedralFull(3)
    print(di.complex.topology.elements[2])
    print(di.factors)
    print(di.table)

    print(npi.count(di.element_order))
    print()
    print(di.multiply([1, 2], 2))


def test_subgroups():
    di = dihedral.ChiralDihedral(3)
    print(di.index)
    di = dihedral.Cyclic(3)
    print(di.index)


def test_basic():
    di = dihedral.ChiralDihedral(3)

    print(di.elements_tables)
    print(di.index)
