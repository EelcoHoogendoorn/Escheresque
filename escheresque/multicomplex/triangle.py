
import numpy as np
import numpy_indexed as npi
from cached_property import cached_property

from pycomplex.complex.simplicial.spherical import ComplexSpherical2, TopologyTriangular
from pycomplex.geometry.euclidian import segment_normals


class Schwartz(ComplexSpherical2):
    """Schwartz triangle that tiles the group"""

    @staticmethod
    def from_group(group):
        """

        Parameters
        ----------
        group : TriangleGroup
        """
        # topology = TopologyTriangular(
        #     elements=[
        #         np.asarray([0, 1, 2]),
        #         np.asarray([[1, 2], [2, 0], [0, 1]]),
        #         np.asarray([[0, 1, 2]])
        #     ],
        #     boundary=[
        #         np.asarray([[1, 2], [2, 0], [0, 1]]),
        #         np.asarray([[0, 1, 2]])
        #     ],
        #     orientation=[
        #         np.asarray([[-1, 1], [-1, 1], [-1, 1]]),
        #         np.asarray([[1, 1, 1]])
        #     ]
        # )
        triangle = Schwartz(
            vertices=group.basis[0],
            simplices=[[0, 1, 2]]
            # topology=topology
        )
        triangle.group = group
        return triangle

    def subdivide(self):
        """

        Returns
        -------
        type(self)
        """
        fine = self.subdivide_loop()
        fine.parent = self
        self.child = fine
        fine.group = self.group
        return fine

    @cached_property
    def boundary_vertices_chain(self):
        """Compute membership of pmd vertices

        Returns
        -------
        ndarray, [n_vertices, 3], int
            chain indicating membership in pmd vertex
        """
        try:
            return self.topology.transfer_matrices[0] * self.parent.boundary_vertices_chain
        except:
            return np.eye(3, dtype=np.int8)

    @cached_property
    def boundary_edges_chain(self):
        """Compute membership of pmd edges

        Returns
        -------
        ndarray, [n_edges, 3], int
            chain indicating membership in pmd edge
        """
        try:
            return self.topology.transfer_matrices[1] * self.parent.boundary_edges_chain
        except:
            return np.eye(3, dtype=np.int8)[self.topology._boundary[1][0]]

    @cached_property
    def boundary_edge_vertices_chain(self):
        """edge vertices minus corners"""
        return np.logical_and(
            (np.abs(self.topology.matrix(0, 1)) * self.boundary_edges_chain) > 0,
            np.logical_not(np.any(self.boundary_vertices_chain, axis=1, keepdims=True))
        ).astype(np.int8)

    @cached_property
    def boundary_vertices(self):
        """

        Returns
        -------
        ndarray : [3, 1], int
        """
        r, c = np.nonzero(self.boundary_vertices_chain)
        return npi.group_by(c).split_array_as_array(r)

    @cached_property
    def boundary_edges(self):
        """

        Returns
        -------
        ndarray, [3, 2**level], int
        """
        r, c = np.nonzero(self.boundary_edges_chain)
        return npi.group_by(c).split_array_as_array(r)

    @cached_property
    def boundary_edge_vertices(self):
        """

        Returns
        -------
        ndarray, [3, 2**level-1], int
        """
        r, c = np.nonzero(self.boundary_edge_vertices_chain)
        return npi.group_by(c).split_array_as_array(r)

    def triangle_normals(self, radius):
        """Compute triangle normals

        Parameters
        ----------
        radius : ndarray, [..., n_vertices]
            set of primal 0-forms describing height fields

        Returns
        -------
        normals : ndarray, [..., n_triangles, 3], float
            corresponding normals as seen from this Schwartz triangle
        """
        vertices = self.vertices[:, None, :] * radius[:, :, None]
        corners = vertices[self.topology.elements[2]]
        return segment_normals(np.swapaxes(corners, 1, 2))

    def vertex_normals(self, radius):
        """Compute non-normalized vertex normals

        Parameters
        ----------
        radius : ndarray, [n_vertices, ...]
            set of primal 0-forms describing height fields

        Returns
        -------
        normals : ndarray, [n_vertices, ..., 3], float
            corresponding normals as seen from this Schwartz triangle
        """
        triangle_normals = self.triangle_normals(radius)
        A = self.topology.matrix(2, 0)  # sparse with shape [n_vertices, n_triangles]
        n, m = A.shape
        return (A * triangle_normals.reshape(m, -1)).reshape((n,)+ triangle_normals.shape[1:])
