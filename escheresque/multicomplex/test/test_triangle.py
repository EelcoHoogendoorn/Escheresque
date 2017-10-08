
import numpy as np
import matplotlib.pyplot as plt

from escheresque.group2.octahedral import Pyritohedral
from escheresque.multicomplex.triangle import Schwartz



def test_generate():
    group = Pyritohedral()
    triangle = Schwartz.from_group(group.group)
    triangle.plot()
    plt.show()

# test_generate()


def test_normals():
    group = Pyritohedral()
    triangle = Schwartz.from_group(group.group)
    for i in range(3):
        triangle = triangle.subdivide()
    radius = np.random.normal(size=(triangle.topology.n_elements[0], 2))
    vertex_normals = triangle.vertex_normals(radius)
    print (vertex_normals.shape)
    triangle.plot()
    plt.show()

test_normals()