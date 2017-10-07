
from cached_property import cached_property

import numpy as np

from pycomplex.complex.simplicial.spherical import ComplexSpherical2


class Schwartz(ComplexSpherical2):
    """Schwartz triangle that tiles the group"""

    def __init__(self, group):
        """

        Parameters
        ----------
        group : TriangleGroup
        """
        super(Schwartz, self).__init__(
            vertices=group.basis[0],
            simplices=[[0, 1, 2]]
        )

    def subdivide(self):
        """

        Returns
        -------
        type(self)
        """
        fine = self.subdivide_loop()
        fine.parent = self
        self.child = fine
        return fine

    def vertex_normals(self, radius):
        """"""


class MultiComplex(object):
    """stitch Schwartz triangle into a complete covering of the sphere"""

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
            picked subgroup element
        index : ndarray, [n_points], int
            picked quotient group element
        triangle : ndarray, [n_points], int
            picked triangle index
        bary : ndarray, [n_points, 3], float
            barycentric coordinates into triangle
        """
        element, index, bary = self.group.pick(points)
        # transform all points onto the schwartz triangle
        points = np.einsum('nji,njk->nik', self.group.representation[element], points)
        # pick the schwartz triangle
        triangle, bary = self.triangle.pick_primal(points)
        return element, index, triangle, bary

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

    @cached_property
    def multiplicity(self):
        """How many copies of each vertex in the primal 0-form exist"""
        return self.boundify_p0(np.ones(self.shape))

    @cached_property
    def inverse_multiplicity(self):
        return 1 / self.multiplicity

    def wrap(self, func):
        """wraps internal state into a linear vector interface"""
        def inner(vec):
            return np.ravel(func(vec.reshape(self.shape)))
        return inner

    def vertex_normals(self, radius):
        """compute triangle/vertex normal over indices

        Parameters
        ----------
        radius : ndarray, [n_vertices, index], float

        Returns
        -------
        normals : ndarray, [n_vertices, index, 3], float

        Notes
        -----
        rotate over all transforms -> gives transforms x index x vertex x 3 array
        precompute not only indices, but also transforms belonging to boundary operations
        use this to select normals to average over, in specialized boundary op

        extra cost is not too bad; only once per update, other boundary op is called far more often
        nonetheless, this thing is still slowing the interface down
        numba optimized version of this would not hurt
        also, implement 'lazy' version

        """

        v_normal = vertex_normals_python(self, height)

        transforms = self.group.representation
        rotated_normal = np.einsum('txy,ivy->itvx', transforms, v_normal).astype(np.float32)  #index x transform x vertex

        boundify_normals_numba(self, rotated_normal, v_normal)
##        boundify_normals_dense_numba(self, rotated_normal, v_normal)
##        self.boundify_normal(rotated_normal, v_normal)
        return util.normalize( v_normal)

    @cached_property
    def boundify_average_p0_operator(self):
        """Construct a sparse matrix that performs boundification,
        or matching-by-averaging on the boundary elements of the multicomplex"""


    def boundify_p0(self, vec):
        return boundify_numba(self, vec)
    def deboundify_p0(self, vec):
        return vec * self.inverse_multiplicity

    # mg transfer operators
    def restrict_p0(self, x):
        """
        restrict p0 values. not sure that there is a sensible way of doing this
        """
        parent = self
        child = parent.child
        restriction = parent.topology.restriction

        """
        simplest method;  exactly preserves constant functions
        """
        return self.boundify( restriction * x) / self.bounded_weighting

    def restrict_d2(self, x):
        """coarsen dual 2 form"""
        parent = self
        child = parent.child
        return parent.boundify( parent.geometry.restrict_d2(child.deboundify( x)))

    def interpolate_p0(self, x):
        """interpolate p0-form"""
        return self.topology.interpolation * x

    def interpolate_d2(self, x):
        """interpolate dual 2 form."""
        parent = self
        child = parent.child

        return child.boundify(  parent.geometry.interpolate_d2(parent.deboundify( x)))

    def interpolate_special(self, x):
        """interpolate deboundified midform via d2 pathway."""
        parent = self
        child = parent.child
        #/np.sqrt(self.inverse_multiplicity)
        d2 = (parent.P0s * x)
        d2 = parent.topology.interpolation *( d2)
        return child.sP0 * d2
#         d2 = (parent.d2s * x)
# ##        d2 = parent.deboundify(D2)
#         d2 = parent.geometry.interpolate_d2( d2)
#         return child.sd2 * child.deboundify( child.boundify(d2))

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

        triangle = Schwartz(group.group)

        hierarchy = [MultiComplex(group, triangle)]

        for i in range(levels):
            parent = hierarchy[-1]
            child = MultiComplex(group, parent.triangle.subdivide())
            parent.child = child
            child.parent = parent
            hierarchy.append(child)
        return hierarchy