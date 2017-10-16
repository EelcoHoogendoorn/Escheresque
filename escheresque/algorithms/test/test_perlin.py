""""Generate Perlin noise on a sphere"""

import matplotlib.pyplot as plt
from pycomplex.math import linalg
from escheresque.algorithms.perlin import perlin_noise


if __name__ == '__main__':
    from escheresque.multicomplex.multicomplex import MultiComplex
    from escheresque.group2.octahedral import Pyritohedral, ChiralTetrahedral, ChiralDihedral2
    group = Pyritohedral()
    complex = MultiComplex.generate(group, 4)[-1]

    field = perlin_noise(
        complex,
        [
            (.0, .1),
            # (.2, .2),
            # (.4, .4),
        ]
    )

    print (field.min(), field.max())

    for i in range(group.index):
        # apply index transform
        complex.triangle.as_euclidian().plot_primal_0_form(field[:, i], cmap='terrain', plot_contour=False, shading='gouraud')
        # for s in range(group.order):

    plt.axis('off')
    plt.show()


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