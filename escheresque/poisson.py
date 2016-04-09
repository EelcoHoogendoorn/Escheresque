
"""
poisson code

solves immersed boundary conditions
that is, boundary conditions which need not coincide with the meshing,
and can have a certain softness to them


instead of iterative method, we may also compute a dense matrix of interaction strengths
only requires updates for updated points
just solve poisson for each point independently; global delta-displacement for a delta force at given point
this maps out the whole linear interaction between the points. this makes adejusting height matter of dense linalg
update needs only be done if curves are moved or added
this will make radius and force edits near instant

what about preconditioning?
how does height-residual transform into a search direction?
more closely spaced points should recieve a smaller contribution, no?
compute local density as precomputation step?
give each point a coeff of one, diffuse, and measure what is left
divide residual by this value. the smaller it is, the more the point is on its own
however, large-width points get lower weight, which is not very desirable
correct for this again?
so far, this doesnt work

use multigrid in outer solver as well?
coefficients obtained at the lower level should be meaningful at high level too...
coeffs stop being meaningful at some point though. also, no point in going coarser
if boundary points start to rival interior points
"""

from itertools import izip

import numpy as np

from escheresque import util
from escheresque import brushes
from escheresque import multigrid


def solve_poisson_multigrid(datamodel):
    """
    solve poisson with mg outer loop

    we could also prolongate search directions maybe?
    """

    edges = [edge.instantiate()[0,0] for edge in datamodel.edges if edge.driving]
    edges = [(edge[1:]+edge[:-1])/2 for edge in edges]


    coefficients, heightfield = solve_poisson_rec(
        datamodel.hierarchy,
        datamodel.complex.D2P0 * datamodel.forcefield,
        edges, None)

    #final high quality solve?

    #this is what we are here for
    return heightfield


def solve_poisson_rec(hierarchy, forcefield, edges, coefficients):
    """
    poisson solution on a single height level
    """
    complex = hierarchy[-1]

    #decide on recursion
    if len(hierarchy) > 4:
        coefficients, heightfield = solve_poisson_rec(
            hierarchy[:-1],
            hierarchy[-2].restrict_d2(forcefield),
            edges,
            coefficients
            )

    def smooth(x):
        width = 0.005
        return multigrid.diffuse(hierarchy, x, width**2)

    def height_from_force(force):
        return multigrid.solve_poisson(hierarchy, force)

    #external force field
    offset = 0
##    external_force = complex.D2P0 * (forcefield + offset)
    external_force = forcefield


    if len(edges)==0:
        #? why center is p0?
        f = complex.P0D2 * external_force
        f = f - f.mean()
        return None, height_from_force(complex.D2P0 * f) + 1
    else:
        #take midpoints of edges, for correct boundary handling
        edges = np.vstack(edges)
        radius = np.sqrt(util.dot(edges, edges))        #compute target radius at each point

        print 'outer unknowns'
        print len(radius)

        #pick edges against the sphere. give each edge its own mapping
        mapping = brushes.Mapping(hierarchy, edges)


    def sample_residual(height):
        """maps p0 height form to an absolute residual vector"""
        return radius - mapping.sample(height)


    if coefficients is None:
        #init coefs as average of force
        net_external_force = complex.deboundify(external_force).sum()
        base_coefficients = np.ones_like(radius) / len(radius) * net_external_force
    else:
        base_coefficients = coefficients

    print 'net force'
    print base_coefficients.sum()


    if True:
        force_curve = smooth(mapping.inject(base_coefficients))
        force = force_curve - external_force
    else:
        #if no boundary conditions, act only on homogenous part
        force = external_force.mean() - external_force

    base_height = height_from_force(force)

    height_dirs = []
    coeff_dirs  = []

    height       = base_height       + 0
    coefficients = base_coefficients + 0

    #this substraction is key; if we search in homogenous subspace, need to translate rhs there too!
    homogenous_radius = radius - mapping.sample(base_height)

    def find_homogenous_minimizer(basis):
        """
        find the linear combination of height basis vectors, such that
        sampled_basis dot coeff ~= radius
        perform fit modulo ones vector. sampled basis vectors need no sum to zero, but we dont want to fit this component
        """
        B = np.vstack([mapping.sample(b) for b in basis]+[np.ones_like(radius)])
        S = np.dot(B, B.T)
        r = np.dot(B, homogenous_radius)
        c = np.linalg.lstsq(S, r)[0]
##        print c

        return lambda I: sum(i*q for i,q in izip(I, c))


##    for i in range(18-len(hierarchy)):  #iterate until convergence
    for i in range(10):  #iterate until convergence
        print i
        #scale curves with coef estimates as preconditioner?
        coeff_dir = sample_residual(height) #/ scaling
        coeff_dir = coeff_dir - coeff_dir.mean()   #balance forces; move only in space of valid coefficients
        print 'outer convergence',  np.linalg.norm( coeff_dir)

        force_dir   = smooth(mapping.inject(coeff_dir))
        height_dir  = height_from_force(force_dir)

        height_dirs.append(height_dir)
        coeff_dirs .append(coeff_dir)

        minimizer = find_homogenous_minimizer(height_dirs)

        height       = base_height       + minimizer(height_dirs)
        coefficients = base_coefficients + minimizer(coeff_dirs)


    res = sample_residual(height)
    height = height + res.mean()


##    print len(hierarchy)
##    print coefficients


    if False:
        #visualize force field
        force -= force.min()
        force /= force.max()
        return force*0.1+1

    return coefficients, height




def solve_poisson(datamodel):
    """
    take a datamodel
    and return a heightfield that conforms to the immersed boundary conditions

    inner iteration is the mg solve from force to height

    outer iteration is a krylov subspace method
    is has several custom steps, to cope with the left and right nullspace problem

    """
    hierachy = datamodel.hierarchy
    complex = hierachy[-1]

    def smooth(x):
        width = 0.005
        return multigrid.diffuse(hierachy, x, width**2)

    def height_from_force(force):
##        return multigrid.solve_poisson(hierachy, force)
        return complex.P0D2 * multigrid.solve_poisson_full(hierachy, force)

    #external force field
    offset = 0
    scaling = -1
    external_force = complex.D2P0 * (datamodel.forcefield * scaling + offset)

##    def map_edge(edge):
##        points = edge.instantiate()[0,0]
##        points = (points[1:]+points[:-1])/2
##        radius = np.sqrt(util.dot(points, points))        #compute target radius at each point
##        mapping = brushes.Mapping(hierachy, points)
##
##    edges = [map_edge(edge) for edge in datamodel.edges if edge.driving]


    #instantiate all curves; pick first occurance
    edges = [edge.instantiate()[0,0] for edge in datamodel.edges if edge.driving]
    if len(edges)==0:
        f = complex.P0D2 * external_force
        f = f - f.mean()
        return height_from_force(complex.D2P0 * f) + 1
    #take midpoints of edges, for correct boundary handling
    edges = [(edge[1:]+edge[:-1])/2 for edge in edges]
    edges = np.vstack(edges)
    radius = np.sqrt(util.dot(edges, edges))        #compute target radius at each point

    print 'outer unknowns'
    print len(radius)

    #pick edges against the sphere. give each edge its own mapping
    mapping = brushes.Mapping(hierachy, edges)


    #preconditioner. less response in denser regions. somehow, this is a complete disaster
##    scaling = mapping.sample(smooth(mapping.inject(np.ones_like(radius))))
####    scaling = np.sqrt(scaling)
##    print 'scaling'
##    print scaling
##    print scaling.min(), scaling.max()


    def sample_residual(height):
        """maps p0 height form to an absolute residual vector"""
        return radius - mapping.sample(height)


    #init coefs as average of force
    net_external_force = complex.deboundify(external_force).sum()
    base_coefficients = np.ones_like(radius) / len(radius) * net_external_force

    if True:
        force_curve = smooth(mapping.inject(base_coefficients))
        force = force_curve - external_force
    else:
        #if no boundary conditions, act only on homogenous part
        force = external_force.mean() - external_force

    base_height = height_from_force(force)

    height_dirs = []
    coeff_dirs  = []

    height       = base_height       + 0
    coefficients = base_coefficients + 0

    #this substraction is key; if we search in homogenous subspace, need to translate rhs there too!
    homogenous_radius = radius - mapping.sample(base_height)

    def find_homogenous_minimizer(basis):
        """
        find the linear combination of height basis vectors, such that
        sampled_basis dot coeff ~= radius
        perform fit modulo ones vector. sampled basis vectors need no sum to zero, but we dont want to fit this component
        """
        B = np.vstack([mapping.sample(b) for b in basis]+[np.ones_like(radius)])
        S = np.dot(B, B.T)
        r = np.dot(B, homogenous_radius)
##        c = np.linalg.lstsq(S, r)[0]
        c = np.linalg.solve(S, r)
##        print c

        return lambda I: sum(i*q for i,q in izip(I, c))


    for i in range(15):  #iterate until convergence
        print i
        #scale curves with coef estimates as preconditioner?
        coeff_dir = sample_residual(height) #/ scaling
        coeff_dir = coeff_dir - coeff_dir.mean()   #balance forces; move only in space of valid coefficients
        print 'outer convergence'
        print np.linalg.norm( coeff_dir)

        force_dir   = smooth(mapping.inject(coeff_dir))
        height_dir  = height_from_force(force_dir)

        height_dirs.append(height_dir)
        coeff_dirs .append(coeff_dir)

        minimizer = find_homogenous_minimizer(height_dirs)

        height       = base_height       + minimizer(height_dirs)
        coefficients = base_coefficients + minimizer(coeff_dirs)


    res = sample_residual(height)
    height = height + res.mean()


    print coefficients


    if False:
        #visualize force field
        force -= force.min()
        force /= force.max()
        return force*0.1+1

    return height


