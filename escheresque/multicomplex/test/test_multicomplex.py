from escheresque.group2.octahedral import Pyritohedral
from escheresque.multicomplex.multicomplex import MultiComplex

import matplotlib.pyplot as plt


def test_generate():
    group = Pyritohedral()

    complex = MultiComplex.generate(group, 4)

    complex[-1].triangle.plot()
    plt.show()

test_generate()