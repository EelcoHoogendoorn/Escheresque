
import numpy as np
import numpy.testing as npt
import numpy_indexed as npi

from escheresque.group2 import icosahedral

from pycomplex.math import linalg


def test_sub_representation():
    print(icosahedral.Icosahedral().order)
    print(icosahedral.ChiralIcosahedral().order)
    print(icosahedral.Dihedral5().order)
    print(icosahedral.ChiralCyclic5().order)
    print(icosahedral.Dihedral3().order)
    print(icosahedral.ChiralDihedral3().order)
    print(icosahedral.Dihedral2().order)
    print(icosahedral.ChiralDihedral2().order)
    print(icosahedral.Null().order)
test_sub_representation()
quit()

def test_basic():
    group = icosahedral.Icosahedral()

    npt.assert_allclose(np.linalg.norm(group.complex.vertices, axis=-1), 1.0)

    print(group.fundamental_domains.shape)
    print([v.shape for v in group.vertices])
    basis = group.basis_from_domains(group.fundamental_domains)
    print(basis.shape)

    npt.assert_allclose(np.linalg.norm(basis, axis=-1), 1.0)
    transforms = group.transforms_from_basis(basis)
    orientations = group.orientation_from_basis(basis)
    # print(orientations)
    representation, relative = group.relative_transforms(transforms)
    # print(group.from_orbits(np.arange(120)).shape)
    # print(group.from_orbits(np.zeros(120)).shape)
    # print(group.match_domains(group.fundamental_domains))

    print()
    # E, T = group._edges(group.fundamental_domains)
    # for e in E:
    #     print (e)
    # for e in T:
    #     print (e)
    #
    # E, T = group._vertices(group.fundamental_domains)
    # for v in E:
    #     print (v)
    # for v in T:
    #     print (v)
    # print (group.fundamental_vertices(representation))
    # print (group.fundamental_edges(representation).shape)
    v = group.elements_tables(representation)[0]
    for q in np.split(v, np.cumsum(group.complex.topology.n_elements[:-1]), axis=1):
        assert npi.all_unique(q)
        print(q)



test_basic()