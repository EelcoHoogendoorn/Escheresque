
import numpy as np
import scipy.sparse
from cached_property import cached_property


class Harmonics(object):
    """Harmonic properties of a single complex

    would like to have multigrid enabled harmonics solver too
    """

    def __init__(self, complex, k=10):
        """

        Parameters
        ----------
        complex : SymmetricComplex
        k : int
            number of harmonics to compute
        """
        self.complex = complex
        self.k = k

    @cached_property
    def precompute(self):
        laplacian = self.complex.laplacian_stitched_operator
        mass = self.complex.hodge_DP[0].flatten()
        inverse_mass_operator = scipy.sparse.diags(1 / mass)
        return laplacian, mass, inverse_mass_operator
    @cached_property
    def laplacian(self):
        return self.precompute[0]
    @cached_property
    def mass(self):
        return self.precompute[1]
    @cached_property
    def inverse_mass_operator(self):
        return self.precompute[2]

    @cached_property
    def largest_eigenvalue(self):
        """Compute the largest eigenvalue on the multicomplex

        Returns
        -------
        float
        """
        return scipy.sparse.linalg.eigsh(
            self.laplacian,
            M=scipy.sparse.diags(self.mass).tocsc(),
            k=1, which='LM', tol=1e-5, return_eigenvectors=False)

    @cached_property
    def smallest_eigenvalues(self):
        """Compute the lowest set of harmonics on the multicomplex

        Returns
        -------
        ndarray, [k, vertices, index]
        """
        # initialize with stitched vector
        v0 = np.random.randn(*self.complex.shape_p0)
        v0 = self.complex.stitcher_p0(v0)
        # v0 = v0 - v0.mean()
        l, v = scipy.sparse.linalg.eigsh(
            self.laplacian,
            M=scipy.sparse.diags(self.mass).tocsc(),
            k=self.k, which='SA', tol=1e-3, return_eigenvectors=True,
            v0=v0.flatten()
        )
        v = v.T.reshape(-1, *self.complex.shape_p0)
        return l, v#v0[None, ...]

    @cached_property
    def smallest_alt(self):
        """use dense logic instead. scales badly, but more numerically robust"""
        import scipy.linalg
        n = np.prod(self.complex.shape_p0)
        # get dense stitched laplacian\
        # self.complex.stitcher_p0(c) for c in
        A = self.laplacian(self.complex.stitcher_d2_flat * np.identity(n))
        print(np.round(A, 1))
        A = A + A.T
        assert (np.allclose(A, A.T))

        l, v = scipy.linalg.eigh(A, np.diag(self.mass))
        v = v.T.reshape(-1, *self.complex.shape_p0)
        mask = l > 0
        return l[mask], v[mask]#v0[None, ...]
