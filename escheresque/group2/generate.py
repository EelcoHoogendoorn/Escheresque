import numpy as np
import scipy.spatial


def identity():
    """Identity generator"""
    return np.eye(3)


def mirror():
    """Mirror generator

    Notes
    -----
    This is a reflection through the origin; this works for octahedra, but what about other groups?
    does not allow us to create a group with a single left-right symmetry only.
    need more flexible mirror-specification
    """
    return -identity()


def mirror_axis(a):
    return identity() - 2 * np.outer(a, a)


def rotation(a, n):
    """Rotation generator around axis of order n"""
    angle = 2 * np.pi / n
    s, c = np.sin(angle), np.cos(angle)
    skew = np.array([
        [   0,  -a[2],  a[1]],
        [ a[2],    0,  -a[0]],
        [-a[1],  a[0],    0],
    ])
    r = np.eye(3) * c + s * skew + (1 - c) * np.outer(a, a)
    if n < 0:
        r = r.dot(mirror_axis(a))
    return r


def generate(full_representation, generators):
    """Generate a representation from a set of subgroup generators, using a numerical brute force approach

    Parameters
    ----------
    full_representation : ndarray, [full_order, 3, 3], float
        At each step, we map back to this full representation, to avoid accumulation of numerical error
    generators : ndarray, [n_generators, 3, 3], float
        Set of generator elements of the group

    Returns
    -------
    representation : [sub_order, 3, 3], float
        set of all elements of the group spanned by the generators
    """
    generators = np.asarray(generators)
    representation = generators
    tree = scipy.spatial.cKDTree(full_representation.reshape(-1, 9))

    while True:
        new_representation = np.einsum('aij,bjk->abik', generators, representation)
        dist, idx = tree.query(new_representation.reshape(-1, 9))
        assert np.allclose(dist, 0) # make sure we are still inside the group
        idx = np.unique(idx)
        new_representation = full_representation[idx]
        if len(new_representation) == len(representation):
            return representation
        representation = new_representation
