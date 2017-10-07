
from abc import abstractproperty
from cached_property import cached_property

import numpy as np
import numpy_indexed as npi
import scipy.sparse
import scipy.spatial


class Group(object):
    """abc for group related operations"""

    @abstractproperty
    def representation(self):
        """The representation of the full group

        Returns
        -------
        representation : ndarray, [order, 3, 3], float
            representation of the full group, in terms of orthonormal matrices
        """
        raise NotImplementedError

    @cached_property
    def order(self):
        """Number of elements in the group"""
        return len(self.representation)
    @cached_property
    def factors(self):
        """Number of potential subgroup orders and their index

        Returns
        -------
        ndarray, [n_factors], int
            sorted list of integer factors
        """
        n = self.order
        return np.unique([e for i in range(1, int(n**0.5) + 1) if n % i == 0 for e in (i, n//i)])

    def compute_table(self, l, r):
        """Compute a transformation table

        Parameters
        ----------
        l : ndarray, [n, 3, 3]
            a group action representation
        r : ndarray, [m]
            elements transformed by the group action

        Returns
        -------
        table : ndarray, [n, m], uint8
            how elements of r transform under the group action l
            table[i, j] = k means i * j = k
        """
        n, m = len(l), len(r)
        tree = scipy.spatial.cKDTree(r.reshape(m, -1))
        p = np.einsum('nij,mj...->nmi...', l, r)
        dist, idx = tree.query(p.reshape(n * m, -1))
        assert np.allclose(dist, 0), "Not a valid group"
        return idx.reshape(n, m).astype(np.uint8)

    @cached_property
    def table(self):
        """Get Cayley table of the group

        Returns
        -------
        table : ndarray, [order, order], uint8
            how elements of the group combine under multiplication
            table[i, j] = k means i * j = k
        """
        r = self.representation
        table = self.compute_table(r, r)
        assert np.alltrue(np.sort(table, axis=0) == np.arange(self.order)[:, None])
        assert np.alltrue(np.sort(table, axis=1) == np.arange(self.order)[None, :])
        return table

    @cached_property
    def inverse_table(self):
        """Get inverse Cayley table of the group

        Returns
        -------
        table : ndarray, [order, order], uint8
            how elements of the group combine under multiplication
            table[i, j] = k means i / j = k
        """
        r = self.representation
        table = self.compute_table(r, np.swapaxes(r, -1, -2))
        assert np.alltrue(np.sort(table, axis=0) == np.arange(self.order)[:, None])
        assert np.alltrue(np.sort(table, axis=1) == np.arange(self.order)[None, :])
        return table

    @cached_property
    def element_order(self):
        """Compute the length of the orbit of each element in the group

        Returns
        -------
        ndarray, [order], int
            the power to which each group element needs to be raised individually
            to map back to the identity element
        """
        r = self.representation
        s = r
        o = np.ones(self.order, np.uint8)
        for i in range(1, 20):
            idx = np.linalg.norm(s - np.eye(3), axis=(1,2)) < 1e-3
            o[np.logical_and(idx, o == 1)] = i
            s = np.einsum('...ij,...jk->...ik', s, r)
        return o

    def __eq__(self, other):
        """Return true if two groups are identical in terms of representation"""
        return np.array_equiv(self.representation, other.representation)

    def multiply(self, l, r):
        """Combine elements of the group; compute l * r

        Parameters
        ----------
        l : ndarray, [...], int
            group elements
        r : ndarray, [...], int
            group elements

        Returns
        -------
        e : ndarray, [...], int
            group elements
        """
        l, r = np.asarray(l), np.asarray(r)
        broadcast = np.broadcast(l, r)
        ones = np.ones(broadcast.shape, dtype=np.uint8)
        return self.table[
            np.asarray(l*ones).flatten(),
            np.asarray(r*ones).flatten()].reshape(broadcast.shape)

    def divide(self, l, r):
        """Combine elements of the group; compute l / r

        Parameters
        ----------
        l : ndarray, [...], int
            group elements
        r : ndarray, [...], int
            group elements

        Returns
        -------
        e : ndarray, [...], int
            group elements
        """
        l, r = np.asarray(l), np.asarray(r)
        broadcast = np.broadcast(l, r)
        ones = np.ones(broadcast.shape, dtype=np.uint8)
        return self.inverse_table[
            np.asarray(l*ones).flatten(),
            np.asarray(r*ones).flatten()].reshape(broadcast.shape)


class TriangleGroup(Group):
    """Full symmetry group over a triangulated sphere"""

    @abstractproperty
    def vertices(self):
        """3-tuple of ndarray"""
        raise NotImplementedError

    @abstractproperty
    def complex(self):
        raise NotImplementedError

    @abstractproperty
    def triangles(self):
        """Mobius triangles in terms of pmd indices"""
        raise NotImplementedError

    @cached_property
    def n_elements(self):
        return [len(v) for v in self.vertices]

    @cached_property
    def basis(self):
        return self.complex.vertices[self.complex.topology.elements[2]]

    @cached_property
    def representation(self):
        """The representation of the full group

        Returns
        -------
        representation : ndarray, [order, 3, 3], float
            representation of the full group, in terms of orthonormal matrices
        """
        basis = self.basis
        representation = np.linalg.solve(basis[0], basis)
        assert np.allclose(np.einsum('...ij,...kj->...ik', representation, representation), np.eye(3))
        return representation


class PolyhedralGroup(TriangleGroup):
    """TriangleGroup built from a regular polyhedron."""

    @abstractproperty
    def polyhedron(self):
        """The spherical complex, the fundamental division of which represents the full symmetry group"""
        raise NotImplementedError

    @cached_property
    def vertices(self):
        """Primal, edge, and dual vertex positions

        Returns
        -------
        3-tuple of ndarray, [n_n-elements, 3], float
        """
        return self.polyhedron.primal_position

    @cached_property
    def complex(self):
        """

        Returns
        -------
        SphericalComplex

        """
        return self.polyhedron.subdivide_fundamental(oriented=False)

    @cached_property
    def triangles(self):
        """Fundamental domain triangles

        Returns
        -------
        ndarray, [order, 3], int
            last axis are PMD indices

        """
        return self.polyhedron.topology.fundamental_domains().reshape(-1, 3)


class SubGroup(Group):
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
        same for pyritohedral sym hidden in ico
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
        generators = [generate.identity()] + [generate.rotation(self.group.vertices[i], r[i]) for i in range(3)]
        if len(r) == 4:
            if r[-1] < 0:
                generators += [generate.mirror()]
            else:
                raise Exception('can only mirror through the origin')
        return generate.generate(self.group.representation, generators)

    @cached_property
    def elements_tables(self):
        """label elements in the full symmetry group as being in a given orbit,
        as defined by the representation of the group

        Returns
        -------
        3-tuple of ndarray, [order, complex.n_elements], int
            tuple of tables describing how n-elements in the full group map to eachother
            n-th element in the tuple pertains to n-elements
        """
        return tuple([self.compute_table(self.representation, p) for p in self.group.complex.primal_position])

    @cached_property
    def table(self):
        return self.elements_tables[2]

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

    def pick(self, points):
        """Pick the subgroup

        Parameters
        ----------
        points : ndarray, [n_points, 3], float

        Returns
        -------
        element : ndarray, [n_points], int
            picked subgroup element
        index : ndarray, [n_points], int
            picked quotient group element
        bary : ndarray, [n_points, 3], float
            pmd barycentric coordinates
        """
        domains, bary = self.group.complex.pick_primal(points)
        raise NotImplementedError('decompose domains into sub and quotient')
