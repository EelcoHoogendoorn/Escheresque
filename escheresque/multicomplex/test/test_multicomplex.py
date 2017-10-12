from escheresque.group2.octahedral import Pyritohedral
from escheresque.group2.dihedral import Cyclic

from escheresque.multicomplex.multicomplex import MultiComplex

import numpy as np
import matplotlib.pyplot as plt


def test_generate():
    group = Pyritohedral()

    complex = MultiComplex.generate(group, 4)

    complex[-1].triangle.plot()
    plt.show()


def test_pick():
    group = Cyclic(2)
    complex = MultiComplex.generate(group, 6)

    from pycomplex.math import linalg
    N = 1024
    points = np.moveaxis(np.indices((N, N)).astype(np.float), 0, -1) / (N - 1) * 2 - 1
    z = np.sqrt(np.clip(1 - linalg.dot(points, points), 0, 1))
    points = np.concatenate([points, z[..., None]], axis=-1)

    element_idx, sub_idx, quotient_idx, triangle_idx, bary = complex[-1].pick(points.reshape(-1, 3))

    print(bary.min(), bary.max())

    if True:
        col = bary
    else:
        col = np.array([
            sub_idx.astype(np.float) / sub_idx.max(),
            sub_idx * 0,
            quotient_idx.astype(np.float) / quotient_idx.max()
        ]).T

    plt.figure()
    img = np.flip(np.moveaxis(col.reshape(N, N, 3), 0, 1), axis=0)
    plt.imshow(img)

    plt.show()


def test_boundary_info():
    group = Pyritohedral()

    complex = MultiComplex.generate(group, 2)

    complex[-1].boundary_info


test_boundary_info()