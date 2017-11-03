""""Generate harmonic functions on a sphere"""

# FIXME: at subdivision 6, we may get negative eigs, and see some discontinuities; numerical stability issue?
# NOTE: better to work towards these high subdiv with MG anyway

from escheresque.algorithms.harmonics import Harmonics


if __name__ == '__main__':
    from escheresque.multicomplex.multicomplex import MultiComplex
    from escheresque.group2.octahedral import ChiralOctahedral, Pyritohedral, ChiralTetrahedral, ChiralDihedral2
    # from escheresque.group2.icosahedral import Pyritohedral, ChiralIcosahedral
    group = ChiralTetrahedral()
    complex = MultiComplex.generate(group, 4)[-1]

    harmonics = Harmonics(complex)

    l, v = harmonics.smallest_alt

    # print(harmonics.largest_eigenvalue)

    # l, v = harmonics.smallest_eigenvalues
    print(l)

    complex.plot_p0_form(v[3])

