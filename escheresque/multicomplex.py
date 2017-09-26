"""
simplicial multi-complex math

combines fumdanental domain triangle geometry and symmetry group
into an actionable multicomplex on the sphere

need to optimize normal computation
systematically create fallback code paths for numba code,
so numba dependency can be made optional
"""


import numpy as np
import itertools

from numba import jit, autojit
from numba.types import void, float32, float64, int32, pyobject

from escheresque import util
from escheresque import geometry
from escheresque import harmonics


@jit(void(pyobject, float32[:,:,:,:], float32[:,:,:]))
def boundify_normals_numba(complex, old, new):
    group = complex.group
    topology = complex.topology
    """
    boundify normals using rotation information in the edge datastucture
    can also rotate on the fly. probably more efficient. or maybe not. only if relative rotation matrix is passed in with type information
    can also optimize bound structures. can broadcast I over I; only compute each edge once
    this could be a nopython module
    """
    for e in range(3):
        E = group.edges[e]
        T = group.edge_transforms[e]
        be = topology.BE[e]
        for i in range(len(E)):
            for b in range(len(be)):
                v = be[b]
                for c in range(3):
                    new[i,v,c] = 0
                    for ii in range(E.shape[1]):
                        new[i,v,c] += old[E[i,ii], T[i,ii], v, c]
    for e in range(3):
        E = group.vertices[e]
        T = group.vertices_transforms[e]
        v = topology.BV[e]
        for i in range(len(E)):
            for c in range(3):
                new[i,v,c] = 0
                for ii in range(E.shape[1]):
                    new[i,v,c] += old[E[i,ii], T[i,ii], v, c]



##@jit(void(int32[:,:], int32[:,:], int32[:], float64[:,:,:], float64[:,:,:], float64[:,:,:]))
##def boundify_normals_numba_in(domain_index, domain_transform, vertex_index, transforms, old, new):
##    """
##    perform rotations on the fly
##    """
##
##    for b in range(len(vertex_index)):                 #loop through boundary vertices
##        v = vertex_index[b]                             #get vertex index
##        for c in range(3):                             #loop over components
##            for i in range(domain_index.shape[0]):         #loop over domains involved in this boundary
##                q = 0.0                                     #sum the values over all daomins involved in this boundary
##                for ii in range(domain_index.shape[1]):
##                    q += old[v,domain_index[i,ii],c]
##                new[v,i,c] = q




def boundify_python(group, primal, old):
    """applies boundary conditions to a dual 2-form"""
    #mostly, nothing changes
    new = np.copy(old)

    #loop through primal-mid-dual
    for e, E in enumerate(group.edges):
        be = primal.BE[e]
        for i, I in enumerate(E):
            new[be,i] = old[be][:,I].sum(axis=1)

    #loop through primal-mid-dual
    for v,V in enumerate(group.vertices):
        bv = primal.BV[v]
        #loop through indices
        for i,I in enumerate(V):
            new[bv,i] = old[bv,I].sum()

    return new



@jit(void(int32[:], int32[:,:], float64[:,:],float64[:,:]), locals=dict(q=float64, v=int32)) #, index=int32[:,:], vert=int32[:]
def boundify_numba_in(vertex_index, domain_index, old, new):
    """
    apply boundary conditions between domains, where compile time info of all arrays is known
    """
    for b in range(len(vertex_index)):                 #loop through boundary vertices
        v = vertex_index[b]                             #get vertex index
        #needs to be pruned for duplicates
        for i in range(domain_index.shape[0]):
            q = 0.0
            for ii in range(domain_index.shape[1]):    #perform reduction by summing over all domains
                q += old[v,domain_index[i,ii]]
            for ij in range(domain_index.shape[1]):    #broadcast to all participants
                new[v,domain_index[i,ij]] = q


def boundify_numba(complex, old):
    new = np.copy(old)
    group = complex.group
    topology = complex.topology


    def iter_PMD(vert3, index3, old, new):
        for v, i in zip(vert3, index3):     #loop over PMD; somehow slow in numba; seems like static type info is not inferred or used
            boundify_numba_in(v, i, old, new)

    iter_PMD(topology.BE, group.edges   , old, new)   #apply boundary to edges
    iter_PMD(topology.BV, group.vertices, old, new)   #apply boundary to corners
    return new






@jit(void(int32[:], int32[:,:], int32[:,:], float64[:,:,:], float64[:,:,:],float64[:,:,:]), locals=dict(q=float64, v=int32)) #, index=int32[:,:], vert=int32[:]
def boundify_normals_numba_in(vertex_index, domain_index, domain_transforms, rotated_vertex_normal, old, new):
    """
    apply boundary conditions between domains, where compile time info of all arrays is known
    """
    for b in range(len(vertex_index)):                 #loop through boundary vertices
        v = vertex_index[b]

        for i in range(domain_index.shape[0]):          #loop through constraints
            for c in range(3):                          #loop through dimensions
                q = 0.0
                for ii in range(domain_index.shape[1]): #loop over particpiants in constraint
                    q += old[domain_index[i,ii], domain_transforms[i,ii], v, c]
                for ij in range(domain_index.shape[1]):
                    new[ij,v,c] = q


def boundify_normals_dense_numba(complex, rotated_vertex_normal, old):
    """
    boundify normals
    this is a 'dense' implementation; normals are computed for each domain, not just a minimal fundamental domain
    """
    new = np.copy(old)
    group = complex.group
    topology = complex.topology

    def iter_PMD(vert3, index3, transform3, old, new):
        for v, i, t in zip(vert3, index3, transform3):     #loop over PMD; somehow slow in numba; seems like static type info is not inferred or used
            boundify_normals_numba_in(v, i, t, rotated_vertex_normal, old, new)

    iter_PMD(topology.BE, group.edges   , group.transform_edges,    old, new)   #apply boundary to edges
    iter_PMD(topology.BV, group.vertices, group.transform_vertices, old, new)   #apply boundary to corners

    return new

def boundify_normals_sparse_numba():
    """
    in this variant, we do not precompute all rotated normals
    rather, we rotate normal lookup on the fly
    """




@jit(float32[:,:,:](pyobject, float32[:,:]))
def vertex_normals_numba(complex, radius):

    group = complex.group
    geometry = complex.geometry
    topology = geometry.topology

    FV = topology.FV
    PP = geometry.decomposed
    B = group.basis[:,0,0]

    vertices, index = radius.shape
##    print vertices, index

    v_normal = np.zeros((index, vertices, 3), np.float32)

    for i in range(index):
##        b = group.basis[i,0,0]
##        b = util.normalize(b.T).T
##        mirror = np.sign(np.linalg.det(b))
####        print mirror
##
##        P = util.normalize(np.dot(b, PP.T).T) * radius[:,i][:, None]
##
##        tri_normal = np.cross(P[FV[:,1]]-P[FV[:,0]], P[FV[:,2]]-P[FV[:,0]]) * mirror

##        v_normal = np.zeros_like(P)
        for t in range(len(FV)):
            for j in range(3):
                v = FV[t,j]
                for c in range(3):
##                    v_normal[i,v,c] += tri_normal[t,c]
                    v_normal[i,v,c] += 1
    return v_normal




def triangle_normals(complex, radius, index):
    """triangle normals for a root index. do for each index?"""
    group = complex.group
    geometry = complex.geometry
    topology = geometry.topology

    FV = topology.FV
    PP = geometry.decomposed
    B = group.basis[:,0,0]      #grab all root bases

    b = util.normalize(B[index].T).T                   #now every row is a normalized vertex
    P = np.dot(b, PP.T).T * radius[:,index][:, None]   #go from decomposed coords to local coordinate system
    fv = FV[:,::int(np.sign(np.linalg.det(b)))]         #flip sign for mirrored domains
    return np.cross(P[fv[:,1]]-P[fv[:,0]], P[fv[:,2]]-P[fv[:,0]])

def vertex_normals_python(complex, radius):
    """
    efficiency here is essentially optimal
    could vectorize loop, but wouldnt matter much
    not much use for numba then
    """
    vert_normals = np.empty(complex.shape[::-1]+(3,))
    for i in range(complex.index):
        tri_normal = triangle_normals(complex, radius, i)
        vert_normals[i] = complex.topology.T02 * tri_normal     #sum all triangle contributions around vertex
    return vert_normals


@autojit
def laplace_numba(EV, D1P1, x):
    """
    combined gather/scatter; multi-index
    """
    y = np.zeros_like(x)
    index = x.shape[1]
    edges = len(EV)
    for e in range(edges):
        p = EV[e,0]
        m = EV[e,1]
        for i in range(index):
            d = (x[p,i]-x[m,i]) * D1P1[e]
            y[p,i] = y[p,i] + d
            y[m,i] = y[m,i] - d
##    y *= P0D2
    return y

class ImplicitLaplace(object):
    """wrapp laplace info in a class"""
    def __init__(self, EV, D1P1, P0D2):
        self.EV    = EV
        self.D1P1  = D1P1
        self.P0D2  = P0D2
        self.shape = (len(P0D2),)*2
    def __mul__(self, x):
        return laplace_numba(self.EV, self.D1P1, x)



class MultiComplex(object):
    """
    stitch fundamental domain triangles into a complete covering of the sphere
    operations on primal vertices are implemented
    """

    def __init__(self, group, geometry):
        self.group = group
        self.geometry = geometry        #triangle geometry, to be stiched into a covering of the sphere

        self.shape = self.topology.P0, self.index       #perhaps transposed layout is better? unless we unroll inner loop in implicit laplace
        self.size  = np.prod(self.shape)

        self.inverse_multiplicity = 1 / self.boundify(np.ones(self.shape))

        #main laplace object
        self._laplace = ImplicitLaplace(self.topology.EVi, self.geometry.D1P1, self.geometry.P0D2)

        #add boundary interactions to hodge; boundify on inverse; dual areas should get added
        self.D2P0 = self.boundify(np.ones(self.index)[None, :] / self.geometry.P0D2[:,None])        #each index should be the same, no?
        self.P0D2 = 1.0 / self.D2P0

        self.P0s = np.sqrt(self.geometry.P0D2)[:,None]
        self.sP0 = 1/self.P0s

        self.d2s = np.sqrt(self.geometry.D2P0)[:,None]      #these still require boundification
        self.sd2 = 1/self.d2s

##        self.D2s = np.sqrt(self.D2P0)      #these still require boundification
##        self.sD2 = np.sqrt(self.P0D2)

        self.precomp()


    def precomp(self):
        """precompute jacobi iteration matrices"""
        #precompute merged metric operations
        D1P1 = util.dia_simple(self.geometry.D1P1)
        P0D2 = util.dia_simple(self.geometry.P0D2)
        D21 = self.topology.D21
        P10 = self.topology.P10
        L = util.coo_matrix(D21 * D1P1 * P10 * P0D2)
        od = L.row != L.col
        self.off_diagonal = util.csr_matrix(
            (L.data[od], (L.row[od], L.col[od])),
            shape=(self.topology.P0, self.topology.P0))

        self._laplace_d2 = util.csr_matrix(L)

        self.inverse_diagonal = 1.0 / (L.data[od==False][:, None])


    @property
    def topology(self):
        return self.geometry.topology
    @property
    def index(self):
        return self.group.index

##    def laplace(self, state):
##        """laplace operator"""
##        dual_face = self._laplace * state
##        return ( self.boundify(dual_face))
    def laplace_D2P0(self, state):
        return ( self.boundify(self._laplace * state))
    def laplace_P0D2(self, state):
        return self.P0D2 * ( self.boundify(self._laplace * (self.P0D2 * state)))

    def laplace_d2(self, state):
        return  self.boundify(self._laplace * (self.P0D2 * state))
    def laplace_p0(self, state):
        return self.P0D2 * self.boundify(self._laplace * state)

    def laplace_P0D2_special(self, state):
        """laplace operator, with alternate deboundified d2 convention"""
        dual_face = self._laplace * (self.geometry.P0D2[:,None] * state)
        return self.geometry.P0D2[:,None] * self.deboundify( self.boundify(dual_face))
##    def laplace_D2P0_special(self, state):
##        """laplace operator, with alternate deboundified d2 convention"""
##        dual_face = self._laplace * (state)
##        return self.deboundify( self.boundify(dual_face))


    def laplace_normalized_d2(self, x):
        return self.laplace_d2(x) / self.largest
    def laplace_normalized_p0(self, x):
        return self.laplace_p0(x) / self.largest
##    def laplace_normalized_d2(self, x):
##        """
##        apply diffusion operator to vector.
##        take maximum stable timestep.
##        both input and output are D2-forms
##        """
##        return self.laplace(x) / self.largest

    def diffuse_normalized_d2(self, x):
        """
        apply diffusion operator to vector.
        take maximum stable timestep.
        both input and output are D2-forms
        """
        return x - self.laplace_d2(x) / self.largest

    def jacobi_d2(self, x, rhs):
        """jacobi iteration on a d2-form"""
        return self.inverse_diagonal * (rhs - self.boundify(self.off_diagonal * self.deboundify( x)))

    def poisson_residual(self, x, rhs):
        return  rhs - self.laplace_d2(x)
    def poisson_descent(self, x, rhs):
        """alternative to jacobi iteration"""
        r = self.poisson_residual(x, rhs)
        return x + r / (self.largest * 0.9)     #no harm in a little overrelaxation

    def poisson_overrelax(self, x, rhs, knots):
        """overrelax, forcing the eigencomponent to zero at the specified overrelaxation knots"""
        for s in knots / self.largest:
            x = x + self.poisson_residual(x, rhs) * s
        return x



    def wrap(self, func):
        """wraps internal state into a linear vector interface"""
        def inner(vec):
            return np.ravel(func(vec.reshape(self.shape)))
        return inner



    def normals(self, height):
        """
        compute triangle/vertex normal over indices
        rotate over all transforms -> gives transforms x index x vertex x 3 array
        precompute not only indices, but also transforms belonging to boundary operations
        use this to select normals to average over, in specialized boundary op

        extra cost is not too bad; only once per update, other boundary op is called far more often
        nonetheless, this thing is still slowing the interface down
        numba optimized version of this would not hurt
        also, implement 'lazy' version

        """

        v_normal = vertex_normals_python(self, height)

        transforms = self.group.transforms.reshape(-1,3,3)                                     #transforms x 3 x 3
        rotated_normal  = np.einsum('txy,ivy->itvx', transforms, v_normal).astype(np.float32)  #index x transform x vertex

        boundify_normals_numba(self, rotated_normal, v_normal)
##        boundify_normals_dense_numba(self, rotated_normal, v_normal)
##        self.boundify_normal(rotated_normal, v_normal)
        return util.normalize( v_normal)



    def boundify(self, vec):
        return boundify_numba(self, vec)
    def deboundify(self, vec):
        return vec * self.inverse_multiplicity




    def compute_harmonics(self):
        """
        precompute harmonics, for various purposes
        """
        if self.size < 400:     #need at least 360 dofs, so we can handle null symm icosahedral
            print('dense')
            self.complete_eigen_basis = True
            self.eigenvectors, self.eigenvalues = harmonics.full_eigs(self)
            self.largest = self.eigenvalues[-1]
        else:
            print('refine')
            self.complete_eigen_basis = False

            #compute largest harmonic for determining maximum timestep
            self.largest = harmonics.largest_harmonic(self)

            k = 50
            parent = self.parent
            V, v = parent.eigenvectors[:k], parent.eigenvalues[:k]

            V = np.array([parent.interpolate_special(q) for q in V ])
            self.eigenvectors, self.eigenvalues = harmonics.refine_eigs(self, V, v)

            #precompute for diffusion; what impact does smoothing step capable of setting highest eig to zero, on lowest eigencomponent?
            self.smallest = self.eigenvalues[1]
            smallest_multiplier = 1 - self.smallest / self.largest
            self.time_per_iteration = -np.log(smallest_multiplier) / self.smallest
            print(self.time_per_iteration)

        print(self.eigenvalues[:10])

    def solve_eigen_d2(self, x, func):
        """
        map a vector from d2 to eigenspace, and solve func in eigenspace
        used for poisson and diffusion solver
        """
        x = self.sd2 * self.deboundify(x)

        V, v = self.eigenvectors, self.eigenvalues
        y = np.einsum('vji,...ji->v', V, x)
        y = func(y, v)
        y = np.einsum('vji,v...->ji', V, y)

        return self.boundify (self.d2s * y)




    #mg transfer operators
    def restrict_p0(self, x):
        """
        restrict p0 values. not sure that there is a sensible way of doing this
        """
        parent = self
        child = parent.child
        restriction = parent.topology.restriction

        """
        simplest method;  exactly preserves constant functions
        """
        return self.boundify( restriction * x) / self.bounded_weighting

    def restrict_d2(self, x):
        """coarsen dual 2 form"""
        parent = self
        child = parent.child
        return parent.boundify( parent.geometry.restrict_d2(child.deboundify( x)))

    def interpolate_p0(self, x):
        """interpolate p0-form"""
        return self.topology.interpolation * x

    def interpolate_d2(self, x):
        """interpolate dual 2 form."""
        parent = self
        child = parent.child

        return child.boundify(  parent.geometry.interpolate_d2(parent.deboundify( x)))

    def interpolate_special(self, x):
        """interpolate deboundified midform via d2 pathway."""
        parent = self
        child = parent.child
        #/np.sqrt(self.inverse_multiplicity)
        d2 = (parent.P0s * x)
        d2 = parent.topology.interpolation *( d2)
        return child.sP0 * d2


        d2 = (parent.d2s * x)
##        d2 = parent.deboundify(D2)
        d2 = parent.geometry.interpolate_d2( d2)
        return child.sd2 * child.deboundify( child.boundify(d2))




def generate(group, levels):
    """
    add parent/child pointers, to make mg type code more clean
    """
    G = geometry.generate(group, levels)

    C = [MultiComplex(group, g) for g in G]
    #create parent/child relations
    for parent,child in zip(C[:-1],C[1:]):
        parent.child = child
        child.parent = parent
    for i, c in enumerate(C):
        c.hierarchy = C[:i+1]
        c.compute_harmonics()

    return C



# class Hierarchy(object):
#     def __init__(self):
#         self.group = None
#
#     def levels(self):
#         """number of subdivisions"""
#         return len(self.stack)-1
#     def __getitem__(self,idx):
#         return self.stack[idx]
#
#     def subdivide(self):
#         """extend heirarchy by one level"""
#         complex = self[-1]
#         t = complex.topology.subdivide_topology()
#         position, planes = complex.topology.subdivide_position(complex.geometry.primal)
#         g = geometry.Geometry(t, position, planes)
#         mc = MultiComplex(self.group, g)
#         self.stack.append(mc)
#
#
