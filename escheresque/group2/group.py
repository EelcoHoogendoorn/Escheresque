
from abc import abstractproperty
from cached_property import cached_property

import numpy as np
import numpy_indexed as npi
import scipy.sparse
import scipy.spatial


class Group(object):
    """
    main symmetry group object, encoding all static information

    Needs a representation, or set of linear transforms, shape [rotations, mirrors, 3, 3]

    needs a symmetrized topology; group tris, edges and verts that form an orbit under the representation
    """

    @abstractproperty
    def complex(self):
        """The spherical complex, the fundamental division of which represents the full symmetry group"""
        raise NotImplementedError

    @abstractproperty
    def description(self):
        """Orbifold-ish description of the generators of the subgroup"""
        raise NotImplementedError

    @cached_property
    def is_chiral(self):
        return self.description[-1] < 0

    @property
    def n_domains(self):
        """Order of the full group"""
        return len(self.fundamental_domains.reshape(-1, 3))

    @cached_property
    def index(self):
        """The index of the subgroup; or the number of independent tiles"""
        return len(self.full_representation / self.representation)
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
        """Primal, edge, and dual vertex positions"""
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
    def full_representation(self):
        """Get a representation of the full group

        Returns
        -------
        representation : ndarray, [n_domains, 3, 3], float
            representation of the full group
        """
        basis = self.basis_from_domains(self.fundamental_domains)
        return self.transforms_from_basis(basis)

    @cached_property
    def representation(self):
        """Get a representation of the sub-group, by parsing the Orbifold description

        Returns
        -------
        representation : ndarray, [order, 3, 3], float
            representation of the sub-group
        """
        r = self.description
        from escheresque.group2 import generate
        generators = [generate.identity()] + [generate.rotation(self.vertices[i][0], r[i]) for i in range(3)]
        if r[-1] < 0:
            generators += [generate.mirror()]
        return generate.generate(self.full_representation, generators)

    # def orientation_from_basis(self, basis):
    #     """generate all linear transforms
    #
    #     Parameters
    #     ----------
    #     basis : ndarray, [..., 3, 3], float
    #
    #     Returns
    #     -------
    #     orientation : ndarray, [..., 3, 3], float
    #         -1 or 1
    #     """
    #     return np.sign(np.linalg.det(basis))

    def pick(self, position):
        """Pick the full symmetry group

        Parameters
        ----------
        points : ndarray, [n_points, 3], float

        Returns
        -------
        domain_idx : ndarray, [n_points], int
            index of the domain being picked
        bary : ndarray, [n_points, n_dim], float
            barycentric coordinates

        """
        # FIXME: add remapping to subgroup
        return self.fundamental_subdivision.pick_primal(position)

    # def relative_transforms(self, transforms):
    #     """Construct relative transforms
    #
    #     Parameters
    #     ----------
    #     transforms : ndarray, [order, order, 3, 3], float
    #
    #     Returns
    #     -------
    #     representation : ndarray, [order, 3, 3], float
    #         set of canonical transformation matrices
    #     relative : ndarray, [order, order], int
    #         table describing how transformations combine
    #
    #     """
    #     transforms = transforms.reshape(transforms.shape[:2] + (-1,))
    #     tree = scipy.spatial.cKDTree(representation)
    #     relative = np.array([tree.query(r)[1] for r in transforms], dtype=np.uint8)
    #     return representation.reshape(-1,3,3), relative

    # def domains_from_orbits(self, orbits):
    #     """Specify the symmetry group in terms of its orbits
    #
    #     Parameters
    #     ----------
    #     orbits : ndarray, [n_domains], int
    #         associates an orbit-label with each domain
    #         identical labels together form an orbit
    #         a set of one label each forms a fundamental domain
    #
    #     Returns
    #     -------
    #     domains: ndarray, [index, order, 3], int
    #         domains, grouped by orbits
    #     """
    #     return npi.group_by(orbits).split_array_as_array(self.fundamental_domains.reshape(-1, 3))

    # def match_domains(self, domains):
    #     """Given a domains array grouped by orbit,
    #     reorder the order axis so they all match the linear representation
    #
    #     Parameters
    #     ----------
    #     domains : ndarray, [index, order, 3], int
    #
    #     Returns
    #     -------
    #     domains : ndarray, [index, rotations, mirrors, 3], int
    #
    #     """
    #
    #     canonical, table = self.relative_transforms(self.transforms_from_basis(self.basis_from_domains(domains)))
    #
    #
    #     orientation = self.orientation_from_basis(self.basis_from_domains(domains))
    #
    #
    #     domains = npi.group_by(orientation).split_array_as_array(self.fundamental_domains.reshape(-1, 3))


    # def _edges(self, domains):
    #     """
    #     Parameters
    #     ----------
    #     domains : ndarray, [index, rotations, mirrors, 3], int
    #         fundamental domains
    #
    #     Returns
    #     -------
    #     edges : ndarray
    #     transforms : ndarray
    #
    #     Notes
    #     -----
    #     compute edges from group description
    #     for each item in group index, compute neighbor for each edge
    #
    #     edges array = order x 3 (PMD)
    #     """
    #
    #     def neighbor(index, edge):
    #         """find neighboring domain indces, and their assocaited transforms"""
    #         edgei = np.ones(3, np.bool)
    #         edgei[edge] = 0
    #         edge = domains[index, 0, 0, edgei]
    #
    #         d = domains.reshape(-1, 3)[:, edgei]
    #         for i, neighbor in enumerate(d):
    #             i, t = np.unravel_index(i, (self.index, self.order))
    #             if np.all(neighbor == edge):
    #                 yield i, t
    #
    #     P = [np.array([list(iter(neighbor(i, p))) for i in range(self.index)], np.int32) for p in range(3)]
    #     edges = tuple([p[:, :, 0] for p in P])
    #     edge_transforms = tuple([p[:, :, 1] for p in P])
    #     return edges ,edge_transforms
    #
    # def _vertices(self, domains):
    #     """
    #
    #     Returns
    #     -------
    #
    #
    #     Notes
    #     -----
    #     analogous to edges
    #     for each index, loop over verts
    #     find all indices this vert is shared with
    #     duplicate entries are no problem
    #
    #     """
    #
    #     def verts(index, point):
    #         point = domains[index, 0, 0, point]
    #
    #         d = domains.reshape(-1, 3)[:, point]
    #         for j, neighbor in enumerate(d):
    #             i, t = np.unravel_index(j, (self.index, self.order))
    #             if neighbor == point:
    #                 yield i, t
    #
    #     P = [np.array([list(iter(verts(i, p))) for i in range(self.index)], np.int32) for p in range(3)]
    #     vertices = tuple([p[:, :, 0] for p in P])
    #     vertices_transforms = tuple([p[:, :, 1] for p in P])
    #     return vertices, vertices_transforms


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
        n_tiles : int
        labels : ndarray, [n_elements], int
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

    def get_root(self, orbit):
        """Pick a set of domains as root tiles

        Returns
        -------
        ndarray, [n_tiles], int
        """
        n_tiles, labels = orbit
        # orbits = npi.group_by(labels).split_array_as_array(np.arange(len(labels)))[:, 0]
        # orbits = npi.group_by(labels).first(np.arange(len(labels)))[1]
        orbits = npi.group_by(labels).split(np.arange(len(labels)))
        return orbits

    @cached_property
    def roots(self):
        return [self.get_root(o) for o in self.orbits]

    @cached_property
    def structured_transforms(self):
        """Impose structure on the set of transforms

        Returns
        -------
        ndarray, [tiles, rotations, mirrors], int

        """
