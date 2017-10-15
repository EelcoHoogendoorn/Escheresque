
from cached_property import cached_property

import numpy as np
import numpy_indexed as npi

from pycomplex.math import linalg

from escheresque.multicomplex.triangle import Schwartz


class MultiComplex(object):
    """Stitch Schwartz triangle into a complete covering of the sphere

    Perhaps SphericalSymmetricComplex is a better name?
    """

    def __init__(self, group, triangle):
        """

        Parameters
        ----------
        group : SubGroup
        triangle : Schwartz triangle
        """
        self.group = group
        self.triangle = triangle
        self.parent = None
        self.child = None

    def pick(self, points):
        """Pick the multicomplex

        Parameters
        ----------
        points : ndarray, [n_points, 3], float

        Returns
        -------
        element : ndarray, [n_points], int
            picked group element
        sub : ndarray, [n_points], int
            picked sub group element
        quotient : ndarray, [n_points], int
            picked quotient group element
        triangle : ndarray, [n_points], int
            picked triangle index
        bary : ndarray, [n_points, 3], float
            barycentric coordinates into triangle
        """
        element, sub, quotient, bary = self.group.pick(points)
        # transform all points back onto the schwartz triangle
        points = np.einsum('nji,nj->ni', self.group.group.representation[element], points)
        # pick the schwartz triangle
        triangle, bary = self.triangle.pick_primal(points)
        return element, sub, quotient, triangle, bary

    @property
    def index(self):
        return self.group.index

    @property
    def topology(self):
        return self.triangle.topology

    @cached_property
    def shape_p0(self):
        """Shape of a p0 form"""
        return self.topology.n_elements[0], self.index

    def vertex_normals(self, radius):
        """Compute vertex normals

        Parameters
        ----------
        radius : ndarray, [n_vertices, index], float

        Returns
        -------
        normals : ndarray, [n_vertices, index, 3], float
            normals for all triangles in the fundamental domain

        """
        vertex_normals = self.triangle.vertex_normals(radius)

        # map these normals over the quotient group
        transforms = self.group.quotient
        rotated_normals = np.einsum('txy,ivy->itvx', transforms, vertex_normals)

        # now do summation over group-boundaries
        # find relative transforms between items in the quotient group
        # ideally, precompute this into sparse matrix operation

        # return normalized result
        return linalg.normalized(rotated_normals)

    @staticmethod
    def generate(group, levels):
        """Generate a multicomplex hierarchy, given a subgroup

        Parameters
        ----------
        group : SubGroup
        levels : int

        Returns
        -------
        list of MultiComplex
        """

        triangle = Schwartz.from_group(group.group)

        hierarchy = [MultiComplex(group, triangle)]

        for i in range(levels):
            parent = hierarchy[-1]
            child = MultiComplex(group, parent.triangle.subdivide())
            parent.child = child
            child.parent = parent
            hierarchy.append(child)
        return hierarchy

    # def grad(self):
    #     """gradient operator, from p0 to p1 forms"""
    #     T01 = self.triangle.topology.matrices[0]
    #     def inner(x):
    #         return T01 * x
    #     return inner
    #
    # def div(self):
    #     """divergence operator, from d1 to d2 forms"""
    #     T01 = self.triangle.topology.matrices[0]
    #     def inner(x):
    #         return T01.T * x
    #     return inner
    #
    # @cached_property
    # def laplace(self):
    #     """Laplace from """
    #     T01 = self.triangle.topology.matrices[0]
    #     return T01.T *
    # @cached_property
    # def hodge_DP(self):
    #     """Compute boundified hodges;
    #     make elements act as their symmetry-completed counterparts
    #
    #     For d1p1, this means adding opposing d1 edge
    #     for p0d2, this means summing over all
    #     """
    def stitch_d2(self, d2):
        """sum over all neighbors and divide by multiplicity"""

    @cached_property
    def stitcher_d2(self):
        """Compute sparse matrix that applies stitching to d2 forms
        acts on flattened d2-form

        """
        info = self.boundary_info
        r = self.ravel(info[:, 1], info[:, 0])
        c = self.ravel(info[:, 2], info[:, 0])
        import scipy.sparse
        def sparse(r, c):
            n = np.prod(self.shape_p0)
            return scipy.sparse.coo_matrix((np.ones_like(r), (r, c)), shape=(n, n))
        return sparse(r, c)

    def ravel(self, q, s):
        """Convert quotient/simplex indices into linear index of flattened form"""
        return self.index * s + q
    def unravel(self, idx):
        return np.unravel_index(idx, self.shape_p0)

    @cached_property
    def boundary_info(self):
        """Return terms describing how the triangle boundary stitches together

        Returns
        -------
        vertices : ndarray, [n_terms], int
            the vertex index this boundary term applies to
            single number for edge vertices; multiple entries for corner vertices
        quotient : ndarray, [n_terms], int
            relative element in quotient group to reach opposing element
            how current index relates to other side of the term
        neighbor : ndarray, [n_terms], int
            relative element in quotient group to reach opposing element
            how current index relates to other side of the term
        sub : ndarray, [n_terms], int
            relative subgroup transform.
            only needed by normal transformation so far to get transformations
        """
        #
        vi = self.group.vertex_incidence            # [n_vertex_entries, 4]
        ei = self.group.edge_incidence              # [n_edge_entries, 4]

        # these are the vertex indices for all edges and corners of the triangle
        bv = self.triangle.boundary_vertices        # [3, 1]
        be = self.triangle.boundary_edge_vertices   # [3, n_boundary_edges]

        def broadcast(a, b):
            shape = len(b), len(a), 3
            a = np.broadcast_to(a[None], shape)
            b = np.broadcast_to(b[:, None], shape[:-1])
            return np.concatenate([b.reshape(-1, 1), a.reshape(-1, 3)], axis=1)

        v = [broadcast(a, b) for a, b in zip(npi.group_by(vi[:, 0]).split(vi[:, 1:]), bv)]
        e = [broadcast(a, b) for a, b in zip(npi.group_by(ei[:, 0]).split(ei[:, 1:]), be)]

        return np.concatenate(v + e, axis=0)
