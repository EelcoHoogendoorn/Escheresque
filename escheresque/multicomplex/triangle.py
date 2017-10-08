
import numpy as np
from pycomplex.complex.simplicial.spherical import ComplexSpherical2
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
        return Schwartz(
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