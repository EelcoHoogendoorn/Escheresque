import numpy as np

from escheresque.algorithms.diffusor import Diffusor


def perlin_noise(complex, octaves):
    """Generate Perlin noise over the given complex

    Parameters
    ----------
    complex : Complex
    octaves : iterable of (sigma, amplitude) tuples

    Returns
    -------
    ndarray
        primal 0-form

    """
    def normalize(x):
        x -= x.min()
        return x / x.max()
    def level(s, a):
        noise = np.random.rand(*complex.shape_p0)
        noise = complex.stitcher_p0(noise)
        return normalize(diffusor.integrate_explicit_sigma(noise, s)) ** 1.5 * a

    diffusor = Diffusor(complex)
    return normalize(sum(level(*o) for o in octaves))
