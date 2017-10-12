
import numpy as np
from cached_property import cached_property
from pycomplex.math import linalg

from escheresque.multicomplex.triangle import Schwartz


class MultiComplex(object):
    """Stitch Schwartz triangle into a complete covering of the sphere"""

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
        return self.topology.P0, self.index

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
    def boundify_d2(self, d2):
        """sum over all neighbors and divide by multiplicity"""

    @cached_property
    def boundifier(self):
        """Compute sparse matrix that applies boundification

        that is, averaging over d2 values
        implement as matrix-multiply?
        boundified = boundifier * unboundified
        or:
        boundified = unboundified - boundifier * unboundified
        last is better since most values should remain untouched
        does not help much with
        """

    @cached_property
    def boundary_info(self):
        """Return terms describing how the triangle boundary stitches together

        Returns
        -------
        vertices : ndarray, [index, n_terms], int
            the vertex index this boundary term applies to
            single number for edge vertices; multiple entries for corner vertices
        quotient : ndarray, [index, n_terms], int
            relative element in quotient group to reach opposing element
            how current index relates to other side of the term
        sub : ndarray, [index, n_terms], int
            relative subgroup transform.
            only needed by normal transformation so far to get transformations
        """
        self.group.elements_tables
        print()
        # note: on group level, take incidence matrices, and apply group structure to them;
        # know which triangle is incident to what vert, and so on
