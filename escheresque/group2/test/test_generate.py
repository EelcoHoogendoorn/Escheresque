import numpy as np

from escheresque.group2 import generate
from escheresque.group2 import octahedral


def test_generate():
    oct_group = octahedral.Octahedral()

    generators = [
        generate.identity(),
        generate.mirror(),
        generate.rotation(oct_group.vertices[0][0], 2),
        generate.rotation(oct_group.vertices[1][0], 1),
        generate.rotation(oct_group.vertices[2][0], 3),
    ]
    representation = generate.generate(oct_group.full_representation, generators)
    print(representation.shape)


test_generate()
quit()
