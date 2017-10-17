
import numpy as np
import numpy.testing as npt
import numpy_indexed as npi

from escheresque.group2 import octahedral


def test_table():
    ico = octahedral.OctahedralFull()
    print(ico.factors)


def test_sub_representation():
    print(octahedral.Octahedral().order)
    print(octahedral.ChiralOctahedral().order)
    print(octahedral.Pyritohedral().order)
    print(octahedral.ChiralTetrahedral().order)
    print(octahedral.Dihedral2().order)
    print(octahedral.ChiralDihedral2().order)
    print(octahedral.Origin().order)
    print(octahedral.TriPlane().order)
    print(octahedral.Plane().order)
    print(octahedral.Null().order)
test_sub_representation()

def test_table():
    group = octahedral.Pyritohedral()
    tables = group.elements_tables

    table = tables[2]
    r, c = np.indices(table.shape)
    import scipy.sparse

    q = table.flatten()
    d = np.ones_like(q)
    M = scipy.sparse.coo_matrix((d, (q, c.flatten())))
    import matplotlib.pyplot as plt
    n_components, labels = scipy.sparse.cs_graph_components(M)
    print(n_components)
    plt.scatter(q, c)
    plt.show()


def test_orbits():
    group = octahedral.ChiralTetrahedral()
    orbits = group.orbits
    # for o in orbits:
    #     print(o)
    print(group.cosets)



def test_basic():
    # tet_group = tetrahedral.Tetrahedral()
    # orbits = np.arange(24) % 2
    # domains = tet_group.from_orbits(orbits)
    # basis = tet_group.basis_from_domains(domains)
    # transforms = tet_group.transforms_from_basis(basis)


    group = octahedral.Octahedral()

    npt.assert_allclose(np.linalg.norm(group.complex.vertices, axis=-1), 1.0)

    print(group.fundamental_domains.shape)
    print([tables.shape for tables in group.vertices])

    # domains = group.fundamental_domains
    orbits = np.arange(48)
    domains = group.domains_from_orbits(orbits)
    basis = group.basis_from_domains(domains)
    npt.assert_allclose(np.linalg.norm(basis, axis=-1), 1.0)
    # transforms = group.transforms_from_basis(basis)
    # transforms = group.representation_from_basis(basis)
    # orientations = group.orientation_from_basis(basis)
    # print(orientations)
    # representation, relative = group.relative_transforms(transforms)

    # orbits = np.zeros(24, dtype=np.uint8) # null group
    orbits = np.arange(48) % 2  # chiral group; index 2, order 24
    orbits = octahedral.Pyritohedral().orbits()
    # print(orbits.shape)
    # quit()
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
    print(npi.count(orbits[tables[2]]))


def test_shape():
    group = octahedral.Pyritohedral()
    group.group.complex.plot(plot_dual=False)
    import matplotlib.pyplot as plt
    plt.show()