
import numpy as np
import numpy.testing as npt
import numpy_indexed as npi

from escheresque.group2 import tetrahedral

from pycomplex.math import linalg


def test_table():
    tet = tetrahedral.TetrahedralFull()
    print(tet.table)
    print(tet.inverse_table)
    print(tet.factors)
    print(npi.count(tet.element_order))
    # print(tet.stabilizer_table())

# test_table()
# quit()

def test_sub_representation():
    print(tetrahedral.Tetrahedral().order)
    print(tetrahedral.ChiralTetrahedral().order)
    # print(tetrahedral.Origin().order)
    print(tetrahedral.Plane().order)
    print(tetrahedral.Null().order)

# test_sub_representation()


def test_basic():
    group = tetrahedral.Tetrahedral()

    tables = group.elements_tables

    print(tables[0])
    print(tables[1])
    print(tables[2])

test_basic()
