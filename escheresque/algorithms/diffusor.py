import numpy as np
import scipy.sparse
from cached_property import cached_property


class Diffusor(object):
    """Object to manage diffusion operations over primal 0-forms

    Currently implements explicit integration, no multigrid
    """

    def __init__(self, complex):
        """

        Parameters
        ----------
        complex : SymmetricComplex
        """
        self.complex = complex

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
        # compute largest eigenvalue, for optimally scaled explicit timestepping
        return scipy.sparse.linalg.eigsh(
            self.laplacian,
            M=scipy.sparse.diags(self.mass).tocsc(),
            k=1, which='LM', tol=1e-6, return_eigenvectors=False)

    def explicit_step(self, field, fraction=1):
        """Forward Euler timestep

        Parameters
        ----------
        field : ndarray, [n_vertices], float
            primal 0-form
        fraction : float, optional
            fraction == 1 is the stepsize that will exactly zero out the biggest eigenvector
            Values over 2 will be unstable

        Returns
        -------
        field : ndarray, [n_vertices], float
            primal 0-form
        """
        return field - ((self.inverse_mass_operator * self.laplacian(field.flatten())).reshape(self.complex.shape_p0)) * (fraction / self.largest_eigenvalue)

    def integrate_explicit(self, field, dt):
        """Integrate diffusion equation over a timestep dt

        Parameters
        ----------
        field : ndarray, [n_vertices], float
            primal 0-form
        dt : float
            timestep

        Returns
        -------
        field : ndarray, [n_vertices], float
            diffused primal 0-form

        """
        distance = self.largest_eigenvalue * dt
        steps = int(np.ceil(distance))
        fraction = distance / steps
        for i in range(steps):
            field = self.explicit_step(field, fraction)
        return field

    def integrate_explicit_sigma(self, field, sigma):
        """Integrate for such a length of time,
         as to be equivalent to a gaussian blur with the given sigma

        Parameters
        ----------
        field : ndarray, [n_vertices], float
            primal 0-form
        sigma : float
            sigma of gaussian smoothing kernel

        Returns
        -------
        field : ndarray, [n_vertices], float
            diffused primal 0-form

         """
        dt = sigma ** 2 / np.sqrt(np.pi)
        return self.integrate_explicit(field, dt)
