
from abc import abstractproperty
from cached_property import cached_property

import numpy as np
import numpy_indexed as npi
import scipy.sparse
import scipy.spatial


class TriangleGroup(object):
    """Full symmetry group over a triangulated sphere

    Primary output of this class is a full representation of all elements in the complete group

    Notes
    -----
    polyhedral and dihedral may be seperate subclasses
    """

    @abstractproperty
    def complex(self):
        """The spherical complex, the fundamental division of which represents the full symmetry group"""
        raise NotImplementedError

    @cached_property
    def order(self):
        """Number of transformations composing the representation; rotations x mirrors"""
        return len(self.representation)

    @cached_property
    def fundamental_domains(self):
        """Construct fundamental domains of the full symmetry group

        Returns
        -------
        ndarray, [n_triangles, 3, 2, 3], int
            last axis are PMD indices

        """
        return self.complex.topology.fundamental_domains()

    @cached_property
    def fundamental_subdivision(self):
        """

        Returns
        -------
        SphericalComplex

        Notes
        -----
        triangles are not identical to fundamental domains we seek; contains an orientation-preserving flip!

        """
        return self.complex.subdivide_fundamental()

    @cached_property
    def vertices(self):
        """Primal, edge, and dual vertex positions

        Returns
        -------
        3-tuple of ndarray, [n_n-elements, 3], float
        """
        return self.complex.primal_position

    def basis_from_domains(self, domains):
        """Construct corners

        Parameters
        ----------
        domains : ndarray, [..., 3], int
            domains given as pmd corners

        Returns
        -------
        basis : ndarray, [..., 3, 3], float
            last axis are spatial coordinates,
            second to last are pmd corners
        """
        return np.concatenate([self.vertices[i][domains[..., None, i]] for i in range(3)], axis=-2)

    def transforms_from_basis(self, basis):
        """Generate all linear transforms, from one basis to another

        Parameters
        ----------
        basis : ndarray, [n_domains, 3, 3], float

        Returns
        -------
        transforms : ndarray, [n_domains, 3, 3], float
            transform mapping element
        """
        basis = basis.reshape(-1, 3, 3)
        return np.linalg.solve(basis[0], basis)

    @cached_property
    def representation(self):
        """Get a representation of the full group

        Returns
        -------
        representation : ndarray, [n_domains, 3, 3], float
            representation of the full group
        """
        basis = self.basis_from_domains(self.fundamental_domains)
        return self.transforms_from_basis(basis)


class SubGroup(object):
    """Subgroup of a complete triangle group"""

    @abstractproperty
    def group(self):
        """full triangle group"""
        raise NotImplementedError

    @abstractproperty
    def description(self):
        """Orbifold-ish description of the generators of the subgroup

        Can see this as a presentation; set of operators on the n-elements of the triangles.
        integer is the power applied to that given generator to get the identity
        -1 is a mirror plane along the same axis; extra 4th minus is a mirror in the origin

        what about tet group in octahedral? we have a 3-fold rotation about half the vertices of the cube.
        do we give up anything by not specifying which elements of the polyhedron form the generators?
        not much, in all likelihood..
        need to pick right edge to specify dihedral group on icosahedral
        also, a plane group on a tet basis requires a mirror on an axis not in primal_position
        """
        raise NotImplementedError

    @cached_property
    def is_chiral(self):
        return self.description[-1] < 0

    @cached_property
    def index(self):
        """The index of the subgroup; or the number of independent tiles"""
        return self.group.order / self.order
    @cached_property
    def order(self):
        """Number of transformations composing the representation; rotations x mirrors"""
        return len(self.representation)
    @cached_property
    def mirrors(self):
        return 2 if self.is_chiral else 1
    @cached_property
    def rotations(self):
        return self.order / self.mirrors

    @cached_property
    def representation(self):
        """Get a representation of the sub-group, by parsing the Orbifold description

        Returns
        -------
        representation : ndarray, [order, 3, 3], float
            representation of the sub-group in terms of orthonormal matrices
        """
        r = self.description
        from escheresque.group2 import generate
        generators = [generate.identity()] + [generate.rotation(self.group.vertices[i][0], r[i]) for i in range(3)]
        if len(r) == 4:
            if r[-1] < 0:
                generators += [generate.mirror()]
            else:
                raise Exception('can only mirror through the origin')
        return generate.generate(self.group.representation, generators)

    def table(self, representation, points):
        """Lookup table of indices, specifying how points transform under the given representation of the group

        Parameters
        ----------
        representation : ndarray, [n_transforms, 3, 3], float
        points : ndarray, [n_points, 3], float

        Returns
        -------
        ndarray, [n_transforms, n_points], uint8
            indices into points
            table describing how points map to eachother

        """

        def apply_representation(representation, points):
            return np.einsum('onc,vc->ovn', representation, points)

        basis = apply_representation(representation, points)
        tree = scipy.spatial.cKDTree(points)
        idx = tree.query(basis)[1]
        return idx.astype(np.uint8)

    @cached_property
    def elements_tables(self):
        """label elements in the full symmetry group as being in a given orbit,
        as defined by the representation of the group

        Returns
        -------
        3-tuple of ndarray, [order, n_elements], int
            tuple of tables describing how n-elements in the full group map to eachother
            n-th element in the tuple pertains to n-elements
        """
        return tuple([self.table(self.representation, p) for p in self.fundamental_subdivision.primal_position])

    def get_orbits(self, table):
        """

        Parameters
        ----------
        table : ndarray, [order, n_elements], int

        Returns
        -------
        index : int
        labels : ndarray, [n_elements], int
            all elements with the same label from a coset
            the number of independent cosets is the index of the subgroup

        Notes
        -----
        cosets may be more accurate terminology
        """
        r, c = np.indices(table.shape)
        q = table.flatten()
        d = np.ones_like(q)
        M = scipy.sparse.coo_matrix((d, (q, c.flatten())))
        return scipy.sparse.cs_graph_components(M)

    @cached_property
    def orbits(self):
        """Compute the orbits of all elements in the subgroup"""
        return [self.get_orbits(t) for t in self.elements_tables]

    def quotient(self):
        """Compute a representation of the quotient group"""

    def mirror_group(self):
        """Representation of the mirror group

        This is a group of either order 1 or 2
        """

    def get_root(self, orbit):
        """Pick a set of domains as root tiles

        Returns
        -------
        ndarray, [n_tiles], int
        """
        n_tiles, labels = orbit
        # orbits = npi.group_by(labels).split_array_as_array(np.arange(len(labels)))[:, 0]
        # orbits = npi.group_by(labels).first(np.arange(len(labels)))[1]
        cosets = npi.group_by(labels).split(np.arange(len(labels)))
        return cosets

    @cached_property
    def roots(self):
        return [self.get_root(o) for o in self.orbits]

    @cached_property
    def structured_transforms(self):
        """Impose structure on the set of transforms

        viewed differently; construct the extension / product of the subgroup and the quotient group

        Returns
        -------
        ndarray, [index, rotations, mirrors], int

        """

