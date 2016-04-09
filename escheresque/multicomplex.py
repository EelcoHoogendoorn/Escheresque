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

from escheresque import util



from numba import jit, autojit
from numba.types import void, float32, float64, int32, pyobject


@jit(void(int32[:,:,:],int32[:,:,:], int32[:], float32[:, :, :, :], float32[:, :, :]))
def boundify_edges_normals_numba(group_edges, group_edge_transforms, topology_BE, old, new):
    """
    boundify normals using rotation information in the edge datastucture
    can also rotate on the fly. probably more efficient. or maybe not. only if relative rotation matrix is passed in with type information
    can also optimize bound structures. can broadcast I over I; only compute each edge once
    this could be a nopython module
    """
    for e in range(3):
        E = group_edges[e]
        T = group_edge_transforms[e]
        be = topology_BE[e]
        for i in range(len(E)):
            for b in range(len(be)):
                v = be[b]
                for c in range(3):
                    new[i, v, c] = 0
                    for ii in range(E.shape[1]):
                        new[i, v, c] += old[E[i, ii], T[i, ii], v, c]

@jit(void(int32[:,:,:],int32[:,:,:], int32[:], float32[:, :, :, :], float32[:, :, :]))
def boundify_vertex_normals_numba(group_vertices, group_vertices_transforms, topology_BV, old, new):
    for e in range(3):
        E = group_vertices[e]
        T = group_vertices_transforms[e]
        v = topology_BV[e]
        for i in range(len(E)):
            for c in range(3):
                new[i, v, c] = 0
                for ii in range(E.shape[1]):
                    new[i, v, c] += old[E[i, ii], T[i, ii], v, c]



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

        self.precomp()


    def precomp(self):
        """only need to run this for levels on which we do math. this should be all, normally speaking"""
##        #precompute merged metric operations
##        D1P1 = util.dia_simple(self.geometry.D1P1)
##        D21 = self.topology.D21
##        P10 = self.topology.P10
##        self._laplace = csr_matrix(D21 * D1P1 * P10)        #symmetric part of the laplace matrix
        self._laplace = ImplicitLaplace(self.topology.EVi, self.geometry.D1P1, self.geometry.P0D2)

        #copy P0D2 hodge; once for each domain in index
        self.P0D2 = np.ascontiguousarray( np.column_stack([self.geometry.P0D2]*self.index))
        #add boundary interactions to hodge; boundify on inverse; dual areas should get added
        self.P0D2 = 1.0/self.boundify(1.0/self.P0D2)        #each index should be the same, no?
##        print self.P0D2
##        quit()

        #restriction weighting; add boundification
        self.bounded_weighting = self.boundify(np.column_stack([self.topology.weighting]*self.index))

        self.heightfield = np.zeros(self.shape)     #dont store state here; put in datamodel!

        try:
            self.largest_harmonic()
        except:
            pass


    @property
    def topology(self):
        return self.geometry.topology
    @property
    def index(self):
        return self.group.index

    def laplace(self, state):
        """laplace operator"""
        dual_face = self._laplace * state
        return self.boundify(dual_face)
    def laplace_2(self, state):
        """just a wrapper arond laplace"""
##        dual_face = self._laplace * (self.P0D2 * state)
        dual_face = self._laplace * state
        return self.P0D2 * self.boundify(dual_face)
##        return  self.boundify(dual_face)
    def diffuse(self, x):
        """just a wrapper arond laplace"""
        y = self.laplace(x.reshape(self.shape))
        y *= self.P0D2
        return y.flatten()
    def diffuse_2(self, x, steps=10):
        """
        apply diffusion operator to vector.
        take maximum stable timestep.
        both input and output are P0-forms
        """
        for i in xrange(steps):
            x = self.diffuse_normalized(x)
        return x
##        print self.largest, 'largest'
##        y = np.copy(x)
####        y = self.P0D2 * x
##        for i in xrange(steps):
##            y -= self.P0D2 * self.laplace(y) / self.largest
##        return y

    def diffuse_normalized(self, x):
        """
        apply diffusion operator to vector.
        take maximum stable timestep.
        both input and output are P0-forms
        """
        return x - self.P0D2 * self.laplace(x) / self.largest
    def laplace_normalized(self, x):
        return self.P0D2 * self.laplace(x) / self.largest

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

        # boundify_normals_numba(self, rotated_normal, v_normal)
        boundify_normals_numba(
            self,
            rotated_normal,
            v_normal
        )

        # boundify_edges_normals_numba(
        #     self.group.edges,
        #     self.group.edge_transforms,
        #     self.topology.BE,
        #     rotated_normal,
        #     v_normal
        # )
        # boundify_vertex_normals_numba(
        #     self.group.vertices,
        #     self.group.vertices_transforms,
        #     self.topology.BV,
        #     rotated_normal,
        #     v_normal
        # )
##        boundify_normals_dense_numba(self, rotated_normal, v_normal)
##        self.boundify_normal(rotated_normal, v_normal)
        return util.normalize( v_normal)



    def boundify(self, vec):
        return boundify_numba(self, vec)



    def harmonic(self, eig):

        #we explicitly give an initial guess, and boundify it
        self.heightfield = self.boundify( np.random.normal(size=self.shape))

        from . import harmonics

##        self.heightfield, eig = harmonics.inverse_iteration(self, self.laplace, eig, current = self.heightfield)
        self.heightfield, eig = harmonics.eigs_wrapper(self, self.laplace_2, eig, self.heightfield)
        return eig

    def largest_harmonic(self):
        """find extreme eigenvalues of laplace operator"""
        from scipy.sparse import linalg
        A = linalg.LinearOperator((self.size, self.size), self.wrap(self.laplace_2))

        v0 = self.boundify( np.random.random(self.shape))
        v0 -= v0.mean()
        v0 = np.ravel(v0)

        s, V = linalg.eigs(A, k=1, which='LR', tol=1e-5, v0=v0)
        self.largest = s.real[0]
        print 'largest', self.largest

        v0 = np.ravel(self.boundify( np.random.random(self.shape)))
        s, V = linalg.eigs(A, k=8, which='SR', tol=1e-5, v0=v0)
##        print np.sort( s.real)
        self.smallest = s.real[1]
        print 'smallest', self.smallest
        print 'ratio', self.largest / self.smallest


    #mg transfer operators
    def restrict(self, x):
        """
        full weighting op requires boundification.
        go through dual face?
        seems like improvement
        though not quite sure why presmoothing is required
        could we elimnate this with improved weighting?
        restriction of uniform function should be uniform; can we achieve this?
        also, can we make restriction pure inverse of interpolation?
        these two conditions boil down to the same thing no.
        no; latter is more strict; and clearly unattainable.
        unless we restrict by point sampling; but this is not conservative
        """
##        x *= complex.geometry.D2P0[:,None]
        return self.P0D2 * self.boundify( self.topology.restriction * (self.child.geometry.D2P0[:,None] * x))
##        return self.P0D2 * self.boundify( self.topology.restriction * x)
##        x = self.topology.restriction * x
        x = self.boundify(x)
        return x / self.bounded_weighting
    def interpolate(self, x):
        """this one is easy. or is it? now we interpolate P0. can we improve on this?"""
        return self.topology.interpolation * x




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
##    for b in xrange(len(vertex_index)):                 #loop through boundary vertices
##        v = vertex_index[b]                             #get vertex index
##        for c in xrange(3):                             #loop over components
##            for i in xrange(domain_index.shape[0]):         #loop over domains involved in this boundary
##                q = 0.0                                     #sum the values over all daomins involved in this boundary
##                for ii in xrange(domain_index.shape[1]):
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
    for b in xrange(len(vertex_index)):                 #loop through boundary vertices
        v = vertex_index[b]                             #get vertex index
        #needs to be pruned for duplicates
        for i in xrange(domain_index.shape[0]):
            q = 0.0
            for ii in xrange(domain_index.shape[1]):    #perform reduction by summing over all domains
                q += old[v,domain_index[i,ii]]
            for ij in xrange(domain_index.shape[1]):    #broadcast to all participants
                new[v,domain_index[i,ij]] = q


def boundify_numba(complex, old):
    new = np.copy(old)
    group = complex.group
    topology = complex.topology


    def iter_PMD(vert3, index3, old, new):
        for v, i in itertools.izip(vert3, index3):     #loop over PMD; somehow slow in numba; seems like static type info is not inferred or used
            boundify_numba_in(v, i, old, new)

    iter_PMD(topology.BE, group.edges   , old, new)   #apply boundary to edges
    iter_PMD(topology.BV, group.vertices, old, new)   #apply boundary to corners
    return new






@jit(void(int32[:], int32[:,:], int32[:,:], float64[:,:,:], float64[:,:,:],float64[:,:,:]), locals=dict(q=float64, v=int32)) #, index=int32[:,:], vert=int32[:]
def boundify_normals_numba_in(vertex_index, domain_index, domain_transforms, rotated_vertex_normal, old, new):
    """
    apply boundary conditions between domains, where compile time info of all arrays is known
    """
    for b in xrange(len(vertex_index)):                 #loop through boundary vertices
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
        for v, i, t in itertools.izip(vert3, index3, transform3):     #loop over PMD; somehow slow in numba; seems like static type info is not inferred or used
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

    for i in xrange(index):
##        b = group.basis[i,0,0]
##        b = util.normalize(b.T).T
##        mirror = np.sign(np.linalg.det(b))
####        print mirror
##
##        P = util.normalize(np.dot(b, PP.T).T) * radius[:,i][:, None]
##
##        tri_normal = np.cross(P[FV[:,1]]-P[FV[:,0]], P[FV[:,2]]-P[FV[:,0]]) * mirror

##        v_normal = np.zeros_like(P)
        for t in xrange(len(FV)):
            for j in xrange(3):
                v = FV[t,j]
                for c in xrange(3):
##                    v_normal[i,v,c] += tri_normal[t,c]
                    v_normal[i,v,c] += 1
    return v_normal





##def vertex_normals_python(complex, radius):
##    """
##    efficiency here is essentially optimal
##    could vectorize loop, but wouldnt matter much
##    """
##    group = complex.group
##    geometry = complex.geometry
##    topology = geometry.topology
##
##    FV = topology.FV
##    PP = geometry.decomposed
##    B = group.basis[:,0,0]      #grab all root bases
##
##    def comp_v_normal(h, b):
##        b = util.normalize(b.T).T               #now every row is a normalized vertex
##        mirror = np.sign(np.linalg.det(b))      #flip sign for mirrored domains
##        P = np.dot(b, PP.T).T * h[:, None]      #go from decomposed coords to local coordinate system
####        tri_normal = util.null(util.grab(topology.FV, topology.T10 * P))
##        tri_normal = np.cross(P[FV[:,1]]-P[FV[:,0]], P[FV[:,2]]-P[FV[:,0]])
##        tri_normal *= mirror
##        vert_normal = topology.T02 * tri_normal     #sum all triangle contributions around vertex
##        return vert_normal
##
##    #for each root fundamental domain; sure is ugly relative to proper vectorization..
##    index = complex.index
##    vert_normals = np.empty((index,)+PP.shape)
##    for i in xrange(index):
##        vert_normals[i] = comp_v_normal(radius[:,i], B[i])
##    return vert_normals
####    import itertools
####    return np.array( [comp_v_normal(h, b) for h,b in itertools.izip( radius.T, B)])     #index x verts x 3

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
    fv = FV[:,::np.sign(np.linalg.det(b))]         #flip sign for mirrored domains
    return np.cross(P[fv[:,1]]-P[fv[:,0]], P[fv[:,2]]-P[fv[:,0]])

def vertex_normals_python(complex, radius):
    """
    efficiency here is essentially optimal
    could vectorize loop, but wouldnt matter much
    not much use for numba then
    """
    vert_normals = np.empty(complex.shape[::-1]+(3,))
    for i in xrange(complex.index):
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
    for e in xrange(edges):
        p = EV[e,0]
        m = EV[e,1]
        for i in xrange(index):
            d = (x[p,i]-x[m,i]) * D1P1[e]
            y[p,i] += d
            y[m,i] -= d
##    y *= P0D2
    return y

class ImplicitLaplace(object):
    def __init__(self, EV, D1P1, P0D2):
        self.EV    = EV
        self.D1P1  = D1P1
        self.P0D2  = P0D2
        self.shape = (len(P0D2),)*2
    def __mul__(self, x):
        return laplace_numba(self.EV, self.D1P1, x)



from . import geometry
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

    return C



class Hierarchy(object):
    def __init__(self):
        self.group = None

    def levels(self):
        """number of subdivisions"""
        return len(self.stack)-1
    def __getitem__(self,idx):
        return self.stack[idx]

    def subdivide(self):
        """extend heirarchy by one level"""
        complex = self[-1]
        t = complex.topology.subdivide_topology()
        position, planes = complex.topology.subdivide_position(complex.geometry.primal)
        g = Geometry(t, position, planes)
        mc = MultiComplex(self.group, g)
        self.stack.append(mc)


