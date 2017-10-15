
import numpy as np
import numpy.testing as npt
import numpy_indexed as npi

import matplotlib.pyplot as plt

from escheresque.group2 import icosahedral


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
    group = icosahedral.ChiralIcosahedral()
    full = group.group

    tables = group.elements_tables

    # print(tables[0])
    # print(tables[1])
    # print(tables[2])

    print(group.product_idx)
    print(full.orientation[group.product_idx.T])

    v = group.vertex_incidence
    print(v[:, :-1])
    e = group.edge_incidence
    print(e[:, :-1])

    print(e[:, 2].reshape(3, group.index, 2))

test_basic()

def test_pick():
    group = icosahedral.Pyritohedral()
    full = group.group

    from pycomplex.math import linalg
    N = 128
    points = np.moveaxis(np.indices((N, N)).astype(np.float), 0, -1) / (N - 1) * 2 - 1
    z = np.sqrt(np.clip(1 - linalg.dot(points, points), 0, 1))
    points = np.concatenate([points, z[..., None]], axis=-1)


    element_idx, sub_idx, quotient_idx, bary = group.pick(points.reshape(-1, 3))


    if False:
        col = bary
    else:
        col = np.array([
            sub_idx.astype(np.float) / sub_idx.max(),
            sub_idx * 0,
            quotient_idx.astype(np.float) / quotient_idx.max()
        ]).T


    plt.figure()
    img = np.flip(np.moveaxis(col.reshape(N, N, 3), 0, 1), axis=0)
    # img = (img * 255).astype(np.uint8)
    plt.imshow(img)

    plt.show()

# test_pick()
