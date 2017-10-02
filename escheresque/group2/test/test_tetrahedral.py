
import numpy as np
import numpy.testing as npt
import numpy_indexed as npi

from escheresque.group2 import tetrahedral

from pycomplex.math import linalg


def test_basic():
    group = tetrahedral.Tetrahedral()

    npt.assert_allclose(np.linalg.norm(group.complex.vertices, axis=-1), 1.0)

    print(group.fundamental_domains.shape)
    print([tables.shape for tables in group.vertices])

    # domains = group.fundamental_domains
    orbits = np.arange(24)
    domains = group.domains_from_orbits(orbits)
    basis = group.basis_from_domains(domains)
    npt.assert_allclose(np.linalg.norm(basis, axis=-1), 1.0)
    transforms = group.transforms_from_basis(basis)
    # transforms = group.representation_from_basis(basis)
    # orientations = group.orientation_from_basis(basis)
    # print(orientations)
    # representation, relative = group.relative_transforms(transforms)

    # orbits = np.zeros(24, dtype=np.uint8) # null group
    orbits = np.arange(24) % 2  # chiral group; index 2
    domains = group.domains_from_orbits(orbits)
    basis = group.basis_from_domains(domains)
    npt.assert_allclose(np.linalg.norm(basis, axis=-1), 1.0)
    transforms = group.transforms_from_basis(basis)
    # orientations = group.orientation_from_basis(basis)
    # print(orientations)
    # representation, relative = group.relative_transforms(transforms)


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
    tables = group.elements_tables(transforms)
    for q in np.split(tables[0], np.cumsum(group.complex.topology.n_elements[:-1]), axis=1):
        assert npi.all_unique(q)
        print(q)

    print(tables[0])
    print(tables[1])
    print(tables[2])
    print(orbits[tables[2]])


test_basic()