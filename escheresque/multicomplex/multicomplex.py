
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

    @cached_property
    def stitcher_d2_flat(self):
        """Compute sparse matrix that applies stitching to d2 forms
        acts on flattened d2-form

        Returns
        -------
        sparse matrix
        """
        info = self.boundary_info
        info = info[~np.logical_and(info[:, 1] == info[:, 2], info[:, 3] == 0)]   # remove diagonal
        r = self.ravel(info[:, 1], info[:, 0])
        c = self.ravel(info[:, 2], info[:, 0])
        import scipy.sparse
        def sparse(r, c):
            n = np.prod(self.shape_p0)
            return scipy.sparse.coo_matrix((np.ones_like(r), (r, c)), shape=(n, n))
        return sparse(r, c)

    @cached_property
    def stitch_groups(self):
        info = self.boundary_info
        groups = npi.group_by(info[:, :2])
        return groups

    def alt_stitch_p0(self, p0):
        """stitching using npi

        This can serve as a template for elegant normal stitching
        """
        info = self.boundary_info
        groups = self.stitch_groups
        bt = p0[info[:, 0], info[:, 2]]
        id, mean = groups.mean(bt)
        p0 = p0.copy()
        p0[id[:, 0], id[:, 1]] = mean
        return p0

    def stitch_normals(self, normals):
        """Stitch normals

        Parameters
        ----------
        normals : ndarray, [n_vertices, index, 3], float

        Returns
        -------
        normals : ndarray, [n_vertices, index, 3], float
        """
        info = self.boundary_info
        groups = self.stitch_groups
        r = self.group.group.representation[info[:, 3]]
        bt = normals[info[:, 0], info[:, 2]]
        bt = np.einsum('bij,bj->bi', r, bt)
        id, sum = groups.sum(bt)
        normals = normals.copy()
        normals[id[:, 0], id[:, 1]] = sum
        return normals

    @cached_property
    def stitcher_d2(self):
        """sum over all neighbors

        Returns
        -------
        callable (d2) -> (d2)
        """
        return lambda d2: d2 + (self.stitcher_d2_flat * d2.flatten()).reshape(self.shape_p0)

    @cached_property
    def stitcher_p0(self):
        return lambda p0: self.stitcher_d2(self.triangle.hodge_DP[0][:, None] * p0) / self.hodge_DP[0]


    @cached_property
    def laplacian(self):
        """Laplacian operator from p0 to d2, unstitched

        Returns
        -------
        callable : (p0) -> (d2)
        """
        T01 = self.triangle.topology.matrices[0]
        hodge_D1P1 = self.triangle.hodge_DP[1][:, None]
        return lambda p0: T01 * (hodge_D1P1 * (T01.T * p0))

    @cached_property
    def laplacian_stitched(self):
        """Laplacian operator from p0 to d2

        Returns
        -------
        callable (p0) -> (d2)
        """
        return lambda p0: self.stitcher_d2(self.laplacian(p0))

    @cached_property
    def laplacian_stitched_operator(self):
        """Flat linear operator form of laplacian"""
        from scipy.sparse.linalg import LinearOperator
        n = np.prod(self.shape_p0)
        f = lambda flat_p0: self.laplacian_stitched(flat_p0.reshape(self.shape_p0)).reshape(n)
        return LinearOperator((n, n), f, f)

    @cached_property
    def laplacian_stitched_operator_symm(self):
        """Flat linear operator form of laplacian, symmetrized by right-mult with stitch-transpose"""
        from scipy.sparse.linalg import LinearOperator
        n = np.prod(self.shape_p0)
        f = lambda flat_p0: self.laplacian_stitched((self.stitcher_d2_flat.T * flat_p0).reshape(self.shape_p0)).reshape(n)
        return LinearOperator((n, n), f, f)

    @cached_property
    def hodge_DP(self):
        """Compute stitched hodges;
        make elements act as their symmetry-completed counterparts

        Notes
        -----
        only implemented for p0 so far
        """
        h = np.broadcast_to(self.triangle.hodge_DP[0][:, None], self.shape_p0)
        return [self.stitcher_d2(h), None, None]

    @cached_property
    def hodge_PD(self):
        return [None if h is None else 1. / h for h in self.hodge_DP]

    def plot_p0_form(self, p0):
        """Simple matplotlib debug viz"""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1)
        R = linalg.orthonormalize(np.random.randn(3, 3))
        triangle = self.triangle.as_euclidian()  # .transform(R)
        vmin, vmax = p0.min(), p0.max()
        for s in range(self.group.order):
            for i in range(self.group.index):
                ei = self.group.product_idx[s, i]
                e = self.group.group.representation[ei]
                flip = self.group.group.orientation[ei] < 0
                tile = triangle.transform(e).transform(R)
                tile.plot_primal_0_form(p0[:, i], ax=ax, cmap='terrain',
                                        plot_contour=False, shading='gouraud',
                                        backface_culling=True, flip_normals=flip,
                                        vmin=vmin, vmax=vmax)
        ax.axis('off')
        plt.show()
