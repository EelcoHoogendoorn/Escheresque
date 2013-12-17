"""
eigen module

find harmonic functions
builtin ones fail for multicomplex

actually, this is incorrect.

"""
import numpy as np

def implicit_diffusion(laplace, mu, rhs, steps = 1):
    """
    take in 2d vectors, interface with 1d cg code

    build operator; apply it several times
    """
    def operator(x):
        return laplace(x) - x

    for i in xrange(steps):
        x = cg_wrapper(operator, rhs, x)





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

    for i in xrange(max_iter):
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

    for i in xrange(40):
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
        print new/old
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
        print err

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
    print s[i]

    return V[:,i].reshape(shape), i







