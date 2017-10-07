
import numpy as np
import numpy.testing as npt
import numpy_indexed as npi

from escheresque.group2 import dihedral

from pycomplex.math import linalg


def test_table():
    di = dihedral.DihedralFull(3)
    print(di.complex.topology.elements[2])
    print(di.factors)
    print(di.table)

    print(npi.count(di.element_order))
    print(di.multiply(1, 2))
test_table()
quit()
