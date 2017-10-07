
import numpy as np
import numpy.testing as npt
import numpy_indexed as npi

from escheresque.group2 import icosahedral

from pycomplex.math import linalg


def test_table():
    ico = icosahedral.IcosahedralFull()
    print(ico.factors)
    print(ico.table)

    print(npi.count(ico.element_order))
# test_table()


def test_sub_representation():
    print(icosahedral.Icosahedral().order)
    print(icosahedral.ChiralIcosahedral().order)
    print(icosahedral.Pyritohedral().order)

    print(icosahedral.ChiralIcosahedral() == icosahedral.Pyritohedral())

    print(icosahedral.Cyclic5().order)
    print(icosahedral.ChiralCyclic5().order)
    print(icosahedral.Cyclic3().order)
    print(icosahedral.ChiralCyclic3().order)
    print(icosahedral.Cyclic2().order)
    print(icosahedral.ChiralCyclic2().order)

    print()
    print(icosahedral.Dihedral5().order)
    print(icosahedral.ChiralDihedral5().order)
    print(icosahedral.Dihedral3().order)
    print(icosahedral.ChiralDihedral3().order)
    print(icosahedral.Dihedral2().order)
    # print(icosahedral.ChiralDihedral2().order)
    print()


    print(icosahedral.Origin().order)
    print(icosahedral.Plane().order)
    print(icosahedral.Null().order)
# test_sub_representation()
# quit()

def test_basic():
    group = icosahedral.Icosahedral()

    npt.assert_allclose(np.linalg.norm(group.group.complex.vertices, axis=-1), 1.0)

    v = group.elements_tables
    print(v[0])
    # for q in np.split(v, np.cumsum(group.complex.topology.n_elements[:-1]), axis=1):
    #     assert npi.all_unique(q)
    #     print(q)



test_basic()