import numpy as np
import scipy.spatial


def identity():
    """Identity generator"""
    return np.eye(3)


def mirror():
    """Mirror generator"""
    return -identity()


def rotation(a, n):
    """Rotation generator around axis of order n"""
    angle = 2 *np.pi / n
    s, c = np.sin(angle), np.cos(angle)
    skew = np.array([
        [   0,  -a[2],  a[1]],
        [ a[2],    0,  -a[0]],
        [-a[1],  a[0],    0],
    ])
    return np.eye(3) * c + s * skew + (1 - c) * np.outer(a, a)


def generate(full_representation, generators):
    """Generate a representation from a set of subgroup generators

    Parameters
    ----------
    full_representation : ndarray, [full_order, 3, 3], float
    generators : ndarray, [n_generators, 3, 3], float

    Returns
    -------
    representation : [sub_order, 3, 3], float
    """
    generators = np.asarray(generators)
    representation = generators
    tree = scipy.spatial.cKDTree(full_representation.reshape(-1, 9))

    while True:
        new_representation = np.einsum('aij,bjk->abik', generators, representation).reshape(-1, 3, 3)
        idx = tree.query(new_representation.reshape(-1, 9))[1]
        idx = np.unique(idx)
        new_representation = full_representation[idx]

        if len(new_representation) == len(representation):
            return representation
        representation = new_representation