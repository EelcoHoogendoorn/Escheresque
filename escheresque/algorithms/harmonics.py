
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
        S = self.complex.stitcher_d2_flat

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
        #FIXME: not much success here; implement variable deduplicator instead?
        # may be less efficient, but so what for precomp
        # or can we do it lazy too? should be possible.
        # one complication; we reference only boundary elements now, not interior elements
        # FIXME: advantage of worling on sparse is that we can be sure that largest eigs are actually correct
        import scipy.linalg
        laplacian = self.complex.laplacian_stitched_operator
        groups = self.complex.stitch_groups
        idx = groups.index.sorter[groups.index.start]

        n = np.prod(self.complex.shape_p0)
        # get dense stitched laplacian\
        # self.complex.stitcher_p0(c) for c in
        q = np.identity(n)

        S = self.complex.stitcher_d2_flat + q   # add identity for interior / diag
        # q = (S * np.identity(n)) / (S * np.ones(n))[:, None]

        A = laplacian(S.T) #/ (S * np.ones(n))[:, None]

        print(np.round(A-A.T, 1))
        # A = A + A.T
        assert (np.allclose(A, A.T))
        M = np.diag(self.mass)

        M = np.dot(M, S.T)
        print(np.round(M-M.T, 1))
        assert (np.allclose(A, A.T))

        l, v = scipy.linalg.eigh(A, M)

        mult = np.asarray(S * np.ones((n, 1))).T
        v = v * mult
        v = v.T.reshape(-1, *self.complex.shape_p0)
        mask = l > l.max() / 1e9
        return l[mask], v[mask]#v0[None, ...]

