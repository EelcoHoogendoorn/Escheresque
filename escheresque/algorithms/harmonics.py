
from cached_property import cached_property


class Harmonics(object):

    def __init__(self, complex):
        """

        Parameters
        ----------
        complex: list of MultiComplex
            hierarchy of complexes
        """
        self.complex = complex

    @cached_property
    def largest_eigenvalue(self):
        """Compute the largest eigenvalue on the multicomplex

        Returns
        -------
        float
        """
        pass

    def harmonics(self, n):
        """Compute the lowest set of harmonics on the multicomplex

        Parameters
        ----------
        n : int

        Returns
        -------
        ndarray, [..., n]
        """