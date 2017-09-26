"""
eigen module

find harmonic functions
builtin ones fail for multicomplex



we need biggest eigenvalue for optimal time integration
this should be easy to find
given that we have an efficient multigrid solver, the low eigenvectors should also be easy to find
smallest eigenvalue is good to have for the sake of precise scale-independent diffusion

can we speed up anisotropic diffusion with implicit method?
largest stable timestep depends on both matrix and iterand
the smoother the solution is, the bigger timesteps we may take


should add multigrid-enabled eigensolvers here
"""


import numpy as np








def cg_wrapper(operator, rhs, x0 = None):
    """
    do reshapes, and call cg; is that all there is to it?
    if rhs and x0 are in boundified subspace, we should not need to reapply this constraint
    """
    if not x0 is None: x0=np.ravel(x0)
    shape = rhs.shape
    from scipy.sparse.linalg import cg, LinearOperator, minres, gmres

    def flat_operator(x):
        return np.ravel( operator(x.reshape(shape)))
    N = np.prod(shape)
    wrapped_operator = LinearOperator((N,N), flat_operator, dtype=np.float)

    y = minres(wrapped_operator, np.ravel(rhs), x0, maxiter = 10)[0]
    return y.reshape(shape)





def solve_cg(deflate, operator, rhs, x0 = None):
    """
    solve operator(x) = rhs using conjugate gradient iteration
    operator must be symmetric

    it is understood that operator acts on 2d data
    can we simply wrap regular cg solver?

    """

    def dot(x,y):
        return np.dot(np.ravel(x), np.ravel(y))
##    def ortho(x,d):
##        x -= d * dot(x,d)
##    def project(x):
##        for d in deflation:
##            ortho(x, d)


    eps = 1e-9
    max_iter = 100

##    x = np.zeros_like(rhs) if x0 is None else complex.boundify(x0)
    x = np.copy( x0)
    deflate(x)
    r = rhs - operator(x)
    deflate(r)

    d = np.copy(r)
    delta_new = dot(r,r)
    delta_0 = delta_new

    for i in range(max_iter):
        deflate(d)
        q = operator(d)
        deflate(q)

        dq = dot(d, q)
        if np.abs(dq) < eps: break

        alpha = delta_new / dq

        x += alpha * d
        deflate(x)
        if i%50==49:
            r = rhs - operator(x)
        else:
            r -= alpha * q
        deflate(r)

        if delta_new/delta_0 < eps:
            break
        if dot(x0,x) / dot(x0,x0) > 1e30:        #guard against runaway
            break

        delta_old = delta_new
        delta_new = dot(r,r)
        beta = delta_new / delta_old
        d = r + beta * d
    return x



def inverse_iteration(complex, operator, shift, current = None):
    """
    find eigenvector by inverse iteration
    for some reason this isnt terribly stable
    """
##    q = np.sqrt(complex.P0D2)
    def shifted_operator(x):
        return operator(x) - shift * x
    def dot(x,y):
        return np.dot(np.ravel(x), np.ravel(y))
    def norm(x):
        return np.linalg.norm(np.ravel(x))

    if current is None:
        current = np.random.normal(complex.shape)
    deflation = np.ones(complex.shape)
    deflation /= norm(deflation)
    deflation = [deflation]

    def deflate(x):
##        x /= complex.P0D2
        for d in deflation:
            x -= d * dot(x, d)
##        x *= complex.P0D2
        return x

    eps = 1e-9

    for i in range(40):
        deflate(current)
##        current = cg_wrapper(shifted_operator, current, current)
        current = solve_cg(deflate, shifted_operator, current, current)
        deflate(current)
        old = dot(current, current)
        new = dot(deflate(operator(current)), current)
        if i>25:
            oldshift = shift
            shift = new/old

            if np.abs( shift-oldshift) < eps: break
        print(new/old)
        current /= current.max()

    return current, shift


##function x = rayleigh(A,epsilon,mu,x)

    epsilon = 1e-6
    x = current
    x = x / norm(x)
    y = cg_wrapper(shifted_operator, x, x)
    lamda = dot(y,x)
    shift = shift + 1 / lamda
    err = norm(y-lamda*x) / norm(y)
    while err > epsilon:
        x = y / norm(y)
        y = cg_wrapper(shifted_operator, x, x)
        lamda = dot(y,x)
        shift = shift + 1 / lamda
        err = norm(y-lamda*x) / norm(y)
        print(err)

    return x, shift


def eigs_wrapper(complex, operator, shift, current, k = 20):
    from scipy.sparse.linalg import eigs, LinearOperator
    repeats = 1
    def tile(x):
        return np.array([x]*repeats)

    shape =(repeats,) + complex.shape
    shape = complex.shape
    N = complex.size * repeats
    def flat(x):
        return np.ravel(( operator(x.reshape(shape))))
    A = LinearOperator((N,N), flat, dtype=np.float)
    i = int(shift)
    s, V = eigs(A, k=k, which='SM', tol=1e-7, v0=np.ravel(current))
    s = s.real
    V = V.real
    I = np.argsort(s)
    s = s[I]
    V = V[:,I]

    return V[:,i].reshape(shape), s[i]


def eigs_wrapper(complex, operator, shift, current, k = 20):
    from scipy.sparse.linalg import eigs, LinearOperator

    repeats = 1
    def tile(x):
        return np.array([x]*repeats)

    shape =(repeats,) + complex.shape
    shape = complex.shape
    N = complex.size * repeats
    def flat(x):
        return np.ravel(( operator(x.reshape(shape))))
    A = LinearOperator((N,N), flat, dtype=np.float)
    i = int(shift)
    s, V = eigs(A, k=k, which='SM', tol=1e-7, v0=np.ravel(current))
    s = s.real
    V = V.real
    I = np.argsort(s)
    s = s[I]
    V = V[:,I]

    return complex.P0D2[None,:,:] *  V.T.reshape((-1,)+ shape), s
##    return V[:,i].reshape(shape), s[i]





def largest_harmonic(complex):
    """find extreme eigenvalues of laplace operator"""
    from scipy.sparse import linalg
    A = linalg.LinearOperator((complex.size, complex.size), complex.wrap(complex.laplace_d2))

    v0 = complex.boundify( np.random.random(complex.shape))
    v0 -= v0.mean()
    v0 = np.ravel(v0)

    s, V = linalg.eigs(A, k=1, which='LR', tol=1e-5, v0=v0)
    return s.real[0]






def full_eigs(complex):
    """
    brute force full eigs computation for small size systems.

    we would like to obtain a symm matrix
    system matrix should be full matrix with boundary-violating directions clamped out
    boundify -> transform to other side should be rhs operator?
    boundify term should be in operator
    deboundify(boundify) is unity op for elements in bounded space
    laplace_P0D2 deboundify boundify d2 =  P0D2 d2

    work in special d2 space, needed for matrix symmetry
    convert implicit operator to dense matrix by multiplication with identity matrix
    """
    from scipy.linalg import eigh
    def L(x):
        x = x.T
        y = np.empty_like(x)
        for i in range(len(x)):
            y[i] = complex.laplace_P0D2_special(x[i].reshape(complex.shape)).ravel()
        return y.T
    def db(x):
        x = x.T
        y = np.empty_like(x)
        for i in range(len(x)):
            y[i] = complex.deboundify( complex.boundify(x[i].reshape(complex.shape))).ravel()
        return y.T
    #make sure we start with all components
    V0 = db(np.identity(complex.size))
    A =  L(V0)
    assert(np.allclose(A, A.T))
    P0D2 = np.repeat( complex.geometry.P0D2[:,None], complex.index, axis=1)
    B = np.diag(P0D2.ravel())
    v,V = eigh(A, B)        #solve generalized eigenproblem
    V = V.real
    v = v.real
    I = np.argsort(v)
    V = V[:,I]
    v = v[I]
    I = v>1e-3
    V = V[:,I]
    v = v[I]


    #print 'are we in subspace? need all zeros'
    assert(np.allclose(V, db(V)))


    #append zero eigenvector
    V = np.concatenate(((1 / P0D2).reshape(1,-1).T, V), axis=1)
    v = np.concatenate(([0], v))

    #map vectors to midspace
    V = np.dot( np.sqrt(B), V)

    V[:,0] /= np.linalg.norm(V[:,0])

    V = V.T

    V = V.reshape((-1,)+complex.shape)

    return V, v



def refine_eigs(complex, V0, v0):
    from . import multigrid
    """
    refine a set of eigenvectors.
    multigrid-type approach
    we rely on availability of inverse

    orthonormalize over all, for multiple vecs
    """
    def invert_special_mid(sm, v):
        D2 = complex.boundify( complex.d2s * sm)
##        D2 =  complex.D2s * sm
        D2 = multigrid.solve_poisson_rec( complex.hierarchy, D2, D2/v)
##        return complex.sD2 * D2
        return complex.sd2 * complex.deboundify( D2)
    def diffuse_special_mid(sm, v):
        """isnt this to be preferred over inversion?"""
        D2 = complex.boundify( complex.d2s * sm)
        for i in range(10):
            D2 = complex.diffuse_normalized_d2(D2)
        return complex.sd2 * complex.deboundify( D2)

    V = np.empty_like(V0)
    v = np.empty_like(v0)


    V[0] = V0[0] / np.linalg.norm(V0[0].ravel())
    v[0] = 0

    for i in range(1, len(v0)):
        V[i] = V0[i]

    for r in range(3):
        for i in range(1, len(v0)):
            q = V[i]
##            q = invert_special_mid(q, v0[i])
            q = diffuse_special_mid(q, v0[i])

            q = q / np.linalg.norm( q.ravel())

            #ensure orthonormalization against lower vectors
            for j in range(0, i):
                z = V[j]
                q = q - z * np.dot(q.ravel(), z.ravel())# / np.dot(z.ravel(),z.ravel()))
##            q = q - np.einsum(',->ni', q, V[:i])
            V[i] = q


    #compute eigenvalues for the new eigenvectors
    for i in range(1, len(v0)):
        z = V[i]
        q = complex.sP0 * complex.laplace_P0D2_special(complex.d2s * z)
        v[i] = np.linalg.norm(q.ravel()) / np.linalg.norm(z)

##    Q = V.reshape((len(v),-1))
##    print np.dot(Q, Q.T)

    return V, v




