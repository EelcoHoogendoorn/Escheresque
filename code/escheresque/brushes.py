"""
painting brush module

tools to rasterize a subdivision curve, and so forth.

how to rasterize?
collision with bary triangle is nontrivial
finding index and bary for each point is easy
but how to do lookup in bary space?
precmpute lookup for multicomplex?
could simply use 3d lookup structure. would affect distributabiliy however. dont want boost install

can we make simplest python equivalent of collision code?
cost of doing one triangle-test is similar to doing one level of depth iteration though



distance based dragging, temporal based dragging, and picking+moving towards target
"""


import numpy as np
from . import util

def pick(hierarchy, points):
    """
    pick a set of worldcoords on the sphere
    hierarchical structure of the triangles allows finding intersection triangle quickly
    this should be a method of complex hierarchy no?
    """
    points = util.normalize(points)

    complex = hierarchy[-1]
    group = complex.group

    domains, baries = group.find_support(points)

    local = np.dot(group.basis[0,0,0],baries.T).T

    #next, we need to find the triangle index for each point
    #we can use a simple tree method. simplicity hinges on same argument as multigrid, of recursive divisibility
    faces = np.zeros(len(points), np.int)
    for complex in hierarchy[1:]:   #skip non-subdivided level
        planes = complex.geometry.planes[faces]
        signs = np.einsum('ijk,ik->ij', planes, local) > 0  #where are we, with respect to the three planes defining the 4 triangles?
        #argmax is numerically stable; even in case of multiple true values, we get a valid result
        index = np.argmax(signs, axis=1) + 1        #corner tri index is argmax plus one, since middle triangle is zero
        index *= signs.sum(axis=1) > 0              #if we are infact in the middle of all planes; overwrite with zero
        faces = faces * 4 + index                   #convert local triangle index to global index

    #calc baries; simple linear bary computation is good enough for these purposes, no?
    baries = np.einsum('ijk,ik->ij', complex.geometry.inverted_triangle[faces], local)
    baries /= baries.sum(axis=1)[:, None]
    vertices = complex.topology.FV[faces]       #this is slightly redundant...
    return domains, faces, vertices, baries


def paint(hierarchy, trace):
    #convert picked points to well sampled trace
    edges = zip(trace[1:], trace[:-1])
    samples = np.linspace(0, 1, 11)
    samples = (samples[1:] + samples[:-1])/2
    weights = []
    positions = []
    for l,r in edges:
        L = np.linalg.norm(l-r)
        weights.extend([L]*len(samples))
        positions.extend( l[None, :] * samples[:, None] + r[None, :] * (1-samples[:, None]))
    weights = np.array(weights)
    positions = np.array(positions)#.reshape((-1,3))

    #pick the sampled trace
    domains, triangles, vertices, baries = pick(hierarchy, positions)

    #scatted picked info to brush
    complex = hierarchy[-1]
    brush = np.zeros(complex.shape, np.float)
    util.scatter(
        np.ravel_multi_index((vertices.ravel(),np.repeat(domains[0],3)), complex.shape),   #raveling is key to using efficient scatter here
        np.ravel(baries*weights[:,None]),
        np.ravel(brush))
    brush = complex.P0D2 * complex.boundify(brush)

    #apply diffusion
    from . import multigrid
    brush = multigrid.mg_diffuse_fixed(hierarchy, brush)

    return brush


def floodfill(hierarchy, seed):
    """
    can we do multigrid floodfill? only expand into a region if no blocks whatsoever. iterate to convergence on each level, starting at coarsest
    what if going coarse creates overlap with boundary? those cells need disabling. any overlap with boundary disqualifies
    coarse contribution is ORed in
    so we start at fine level, do some iterations to increase odds we hit a free cell at cooarse level,
    coarsen, and repeat to lowest level
    at each level we iterate till convergence, and OR into higher level
    flooding is easier on tris i suppose, since they map in a boolean hierachircla manner
    """



if __name__=='__main__':
    from tetrahedral import ChiralTetrahedral as Group
    import multicomplex
    complex = multicomplex.generate(Group(), 4)

    p = np.random.randn(10,3)
    pick(complex, p)
    ##pick(complex, np.array([[-3,-2,-1.0],[-1,-2,-3.0]]))

