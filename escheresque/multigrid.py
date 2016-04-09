
"""
multigrid operations

two mg-algos are implemented here:
    a general poisson solver
    a pseudo-steady diffusion solver
    also, in harmonics.py, we have a multigrid eigensolver, that builds on functions here

not sure jacobi method is ideal. stick with residual descent for now

"""

import numpy as np



def solve_poisson_eigen(complex, rhs):
    """solve poisson linear system in eigenbasis"""
    def poisson(x, v):
        y = np.zeros_like(x)
        y[1:] = x[1:] / v[1:]   #linear solve is simple division in eigenspace. skip nullspace
        return y
    return complex.solve_eigen_d2(rhs, poisson)



def solve_poisson_rec(hierarchy, rhs, x):
    """recursive poisson solver step. inputs are D2-forms"""

    complex = hierarchy[-1]

    #coarse level break
    if complex.complete_eigen_basis:
        return solve_poisson_eigen(complex, rhs)


    def profile(func):
        """debug output for intermediate steps"""
        def inner(ix):
            ox = func(ix)
            err = np.linalg.norm(complex.poisson_residual(ix, rhs).ravel()) - \
                  np.linalg.norm(complex.poisson_residual(ox, rhs).ravel())
            print 'improvement', func.__name__
            print err
            return ox
        return inner

    knots = np.linspace(1, 4, 8, True)  #we need to zero out eigenvalues from largest to factor 4 smaller
##    knots = np.sqrt( (knots-1)) + 1
    def solve_poisson_overrelax(x): return complex.poisson_overrelax(x, rhs, knots)

##    iterations = 3
    def solve_poisson_iterate(x, iterations):
        for i in xrange(iterations):
##            x = complex.jacobi_d2(x, rhs)
            x = complex.poisson_descent(x, rhs)
        return x

    def coarsesmooth(x):
        coarse_complex = hierarchy[-2]
        fine_res = complex.poisson_residual(x, rhs)
        coarse_res = coarse_complex.restrict_d2(fine_res)
        coarse_error = solve_poisson_rec(
            hierarchy[:-1],
            coarse_res,
            np.zeros_like(coarse_res),
            )
        fine_error = coarse_complex.interpolate_d2(coarse_error)
        return x - fine_error      #residual correction scheme

    presmooth    = (solve_poisson_iterate)
    postsmooth   = (solve_poisson_iterate)
    coarsesmooth = (coarsesmooth)

##    x = presmooth(x, 5)
    x = solve_poisson_overrelax(x)
    x = coarsesmooth(x)
##    x = postsmooth(x, 5)
    x = solve_poisson_overrelax(x)

    return x


def solve_poisson(hierarchy, rhs, x0 = None):
    """
    v-cycle multigrid solver entry point.
    assumed input is now rhs=D2, x0=P0
    """
    complex = hierarchy[-1]
    if x0 is None:
        x = np.zeros_like(rhs)
    else:
        x = complex.D2P0 * x0

    for i in range(10):
        res = complex.poisson_residual(x, rhs)
        print i, np.linalg.norm(res.ravel())
        x = solve_poisson_rec(hierarchy, rhs, x)
    return complex.P0D2 * x

def solve_poisson_full(hierarchy, rhs):
    """
    full multigrid schema
    first restrict towards and solve on coarsest level
    then prolongate and do a single v-cycle on that level

    why is interpolation so disappointing?
    errors appear to have correct distribution
    try jacobi smoother again?

    """

    complex = hierarchy[-1]

    #walk down to coarsest
    if complex.complete_eigen_basis:
        return solve_poisson_eigen(complex, rhs)



##    print 'restricting'
    #get solution on coarser level first
    x = solve_poisson_full(
        hierarchy[:-1],
        hierarchy[-2].restrict_d2(rhs))
    x = hierarchy[-2].interpolate_d2(x)
##    print 'interpolating'


##    res = complex.poisson_residual(x, rhs)
##    print np.linalg.norm(res.ravel())
##
##    q = x - solve_poisson_rec(hierarchy, rhs, x)
##    print q.mean(), q.std(), q.min(), q.max()

##    print 'v-cycle'
    for i in xrange(2):
        x = solve_poisson_rec(hierarchy, rhs, x)

    res = complex.poisson_residual(x, rhs)
    print np.linalg.norm(res.ravel())

    return x










def diffusion_eigen(complex, d2, time):
    """
    timestep diffusion equation in eigenbasis
    """
    def diffuse(x, v):
        return x * np.exp(-v * time)

    return complex.solve_eigen_d2(d2, diffuse)



def diffuse_rec(hierarchy, x, time):

    """
    mg diffusion over a hierarchy of complexes

    at each level, we start by calculating time requirement for pre/postsmoothing
    if remaining time is greater, do recursive cycle
    otherwise iterate till time is up
    """

    complex = hierarchy[-1]

    if complex.complete_eigen_basis:
        return diffusion_eigen(complex, x, time)

    iterations = 5
    minimum_time = complex.time_per_iteration * iterations * 2  #pre-and postsmoothing

    def number_iterate(x):
        for i in xrange(iterations):
            x = complex.diffuse_normalized_d2(x)
        return x
    def fractional_iterate(x, t):
        xt = complex.diffuse_normalized_d2(x)
        f = t / complex.time_per_iteration
        return xt*f + x*(1-f)   #lerp overshoot for fractional iteration
    def period_iterate(x, t):
        while t > complex.time_per_iteration:
            x = complex.diffuse_normalized_d2(x)
            t = t - complex.time_per_iteration
        return fractional_iterate(x, t)

    if minimum_time > time:
        return period_iterate(x, time)
    else:
        #presmooth; needed to avoid excessive restriction smoothing
        x = number_iterate(x)

        coarse_complex = hierarchy[-2]
        x = coarse_complex.interpolate_d2(
                diffuse_rec(
                    hierarchy[:-1],
                    coarse_complex.restrict_d2(x),
                    time - minimum_time))

        #postsmoothing; needed to eliminate interpolation error
        x = number_iterate(x)
        return x

def diffuse(hierarchy, x, time):
    """
    long-term solution to the diffusion equation using a multigrid approach
    maps d2 to d2
    """
    return diffuse_rec(hierarchy, x.copy(), time)




def diffuse_fixed_rec(hierarchy, x, depth):

    complex = hierarchy[-1]

    iterations = 5
    def diffuse_iterate(x):
        for i in xrange(iterations):
            x = complex.diffuse_normalized_d2(x)
        return x

    def diffuse_coarse(fx):
        coarse_complex = hierarchy[-2]
        return \
            coarse_complex.interpolate_d2(
                diffuse_fixed_rec(
                    hierarchy[:-1],
                    coarse_complex.restrict_d2(x),
                    depth))

    if len(hierarchy)==depth:
        return diffuse_iterate(x)
    else:
        x = diffuse_iterate(x)
        x = diffuse_coarse(x)
        x = diffuse_iterate(x)
        return x

def diffuse_fixed(hierarchy, x, depth=6):
    """
    long-term solution to the diffusion equation using a multigrid approach
    diffuses a fixed amount; can only set level
    maps d2 to d2
    """
    return diffuse_fixed_rec(hierarchy, x.copy(), depth)


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
