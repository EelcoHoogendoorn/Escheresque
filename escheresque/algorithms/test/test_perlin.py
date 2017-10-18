""""Generate Perlin noise on a sphere"""

import numpy as np
import matplotlib.pyplot as plt
from pycomplex.math import linalg
from escheresque.algorithms.perlin import perlin_noise


if __name__ == '__main__':
    from escheresque.multicomplex.multicomplex import MultiComplex
    from escheresque.group2.octahedral import ChiralOctahedral, Pyritohedral, ChiralTetrahedral, ChiralDihedral2
    # from escheresque.group2.icosahedral import Pyritohedral
    group = Pyritohedral()
    complex = MultiComplex.generate(group, 3)[-1]

    p0 = perlin_noise(
        complex,
        [
            (.0, .1),
            # (.2, .2),
            # (.4, .4),
        ]
    )
    # FIXME: this still works really poorly; what does it take
    p0 = complex.stitcher_d2(complex.triangle.hodge_DP[0][:, None] * p0) / complex.hodge_DP[0]

    print(complex.hodge_DP[0])
    print(complex.stitcher_d2(np.ones(complex.shape_p0)))
    print (p0.min(), p0.max())

    complex.plot_p0_form(p0)


    # water_level = 0.42
    # field = np.clip(field, water_level, 1)
    # # add some bump mapping
    # sphere = sphere.copy(vertices=sphere.vertices * (1 + field[:, None] / 5))
    #
    # R = [
    #     [0, 0, -1],
    #     [0, 1, 0],
    #     [1, 0, 0]
    # ]
    # R = linalg.power(R, 1./30)
    #
    # path = r'../output/planet_perlin_2'
    #
    # for i in save_animation(path, frames=30*4, overwrite=True):
    #     sphere = sphere.copy(vertices=np.dot(sphere.vertices, R))
    #     sphere.as_euclidian().plot_primal_0_form(field, cmap='terrain', plot_contour=False, shading='gouraud')
    #     plt.axis('off')