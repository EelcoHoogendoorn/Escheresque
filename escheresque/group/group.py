
"""
symmetry group datastructure

this is where a lot of the magic happens

many useful relations are precomputed, for clean and efficient subsequent code
the symmetries are encoded as relationships between the fundamental domains

constraints need to be changed. simply enable all 6 constraint types in all groups

add more asserts here; you never know what happens
"""

import numpy as np


class Group(object):
    """
    main symmetry group object, encoding all static information

    most important is the domains array; this encodes the symmetry group
    """

    def __init__(self):
#        self.symmetries = []

        self.geometry()

        if not hasattr(self, 'mid'):
            self.mid  = np.array([self.primal[e].sum(0) for e in self.edges])
        if not hasattr(self, 'dual'):       #supplied for dihedral special case
            self.dual = np.array([self.primal[f].sum(0) for f in self.faces])

        self._generate_domains()
        self._fundamental_domain()

        self._precompute()


    def nullify(self):
        """returns a group on the same basis, but with null symmetry"""

    def geometry(self):
        """specifies vertices and their connectivity. subclass must implement this"""
        return NotImplementedError()
    def fundamental_domain(self):
        """standard fundamental domain tiling gives full symmetry; all domains marked as being in the same transform group"""
        return self.full()

    def full(self):
        """standard fundamental domain tiling gives full symmetry; all domains marked as being in the same transform group"""
        return np.zeros(len(self.domains))
    def null(self):
        """each tile is given a unique index; no symmetries exist"""
        return np.arange(len(self.domains))
    def chiral(self):
        """all domains with the smae orientation are in the same rotation group"""
        return self.orientation_from_basis(self.basis_from_domains(self.domains))
    def origin(self):
        B = self.basis_from_domains(self.domains)
        O = self.orientation_from_basis(B)
        return [(b*100).astype(np.int).tostring() for b in O[:,None,None] * B]


    def _generate_domains(self):
        """generate all elementary domains"""
        self.domains = np.array( [(p,m,d)
                        for m, e in enumerate(self.edges)
                            for d, f in enumerate(self.faces)
                                if set(e).issubset(f)
                                    for p in e])


    def _fundamental_domain(self):
        """this specifies the symmetry group"""
        classify = self.fundamental_domain()            #get domain labelig from subclass
        self.index = len(np.unique(classify))
        self.order = len(classify) // self.index
        I = np.argsort(classify)
        D = self.domains[I].reshape((self.index, self.order, 3))
        self._set_domains(D)

    def _set_domains(self, domains):
        """
        given a seperation of domains in index and transforms
        split the transforms in rotations and mirrors, and enforce consistent ordering

        domains input:  index x transform x pmd
        domains output: index x mirror x rotation x pmd
        """

        #resort and reshape data according to mirror transforms
        O = self.orientation_from_basis(self.basis_from_domains(domains))
        I = np.argsort(O, axis=1)       #sort along transform axis
        mdim = len(np.unique(O[0]))     #number of orientations encounted in each transform group
        domains = np.array([d[i].reshape((mdim,-1, 3)) for i,d in zip(I, domains)]) #for each index, place different mirrors in different columns

        #make sure that the first index has consistent orientation
        O = self.orientation_from_basis(self.basis_from_domains(domains))
        domains = domains[np.argsort(O[:,0,0])]


        #reorder domains matrix so all rotations line up.
        _domains = np.empty_like(domains)
        basis = self.basis_from_domains(domains)
        b0 = basis[0,0,0]
        rotations = self.transforms_from_basis(self.basis_from_domains(domains[0,0,:]))
        for i,dd in enumerate(domains):
            for j,d in enumerate(dd):
                candidates = basis[i,j,:]
                for k,R in enumerate(rotations):
##                    match = np.einsum('ij,jk,kl,lm', R, M, J, b0)
                    match = np.einsum('ij,jk', R, candidates[0])
                    kk = np.argmin([np.linalg.norm(match-c) for c in candidates])
                    _domains[i,j,k] = domains[i,j,kk]

##        alltrans = self.transforms_from_basis(self.basis_from_domains(_domains))
##        print alltrans
        ref = self.transforms_from_basis(self.basis_from_domains(_domains[0]))
        for d in _domains:
            q = self.transforms_from_basis(self.basis_from_domains(d))
            assert(np.allclose(q, ref))

        self.domains = _domains


    def _precompute(self):
        """store precomputed quantites; this does not need to be saved to disk really; but it does take long to compute"""
        self.basis = self.basis_from_domains(self.domains)
        self.inverse = np.array([np.linalg.inv(b) for b in self.basis.reshape(-1,3,3)]).reshape(self.basis.shape)
        self.orientation = self.orientation_from_basis(self.basis)

        #transforms relative to first domain
        self.transforms = self.transforms_from_basis(self.basis[0])
        #rotations relative to first domain
        self.rotations = np.array([self.transforms_from_basis(b) for b in self.basis[0]])

        self.permutations = self.composition(self.transforms)

        self._edges()
        self._vertices()



    def _edges(self):
        """
        compute edges from group description
        for each item in group index, compute neighbor for each edge

        edges array = order x 3 (PMD)
        """

        def neighbor(index, edge):
            """find neighboring domain indces, and their assocaited transforms"""
            edgei = np.ones(3, np.bool)
            edgei[edge] = 0
            edge = self.domains[index, 0, 0, edgei]

            domains = self.domains.reshape(-1,3)[:,edgei]
            for i, neighbor in enumerate(domains):
                i,t = np.unravel_index(i, (self.index, self.order))
                if np.all(neighbor == edge):
                    yield i, t


        P = [ np.array( [list(iter(neighbor(i, p))) for i in xrange(self.index)], np.int32) for p in xrange(3)]
        self.edges            = tuple([p[:,:,0] for p in P])
        self.edge_transforms  = tuple([p[:,:,1] for p in P])

##        self.edges = np.array([[list(iter(neighbor(i, e))) for i in xrange(self.index)] for e in xrange(3)], np.int32)
##        self.edges, self.edge_transforms = self.edges[:,:,:,0], self.edges[:,:,:,1]


    def _vertices(self):
        """
        analogous to edges
        for each index, loop over verts
        find all indices this vert is shared with
        duplicate entries are no problem
        """
        def verts(index, point):
            domains = self.domains.reshape(-1,3)[:,point]
            point = self.domains[index, 0, 0, point]

            for j, neighbor in enumerate(domains):
                i,t = np.unravel_index(j, (self.index, self.order))
##                if i==index and t==0: continue      #not equal index and identity transform. could just compare whole domain array..
                if neighbor == point:
                    yield i,t

        P = [ np.array( [list(iter(verts(i, p))) for i in xrange(self.index)], np.int32) for p in xrange(3)]
        self.vertices            = tuple([p[:,:,0] for p in P])
        self.vertices_transforms = tuple([p[:,:,1] for p in P])


##        self.shared_vertex_indices    = tuple(p[:,:,0] for p in P)
##        self.shared_vertex_transforms = tuple(p[:,:,1] for p in P)


    def composition(self, T):
        """
        compose transformations. one may be though of as an absolute transform,
        but perhaps better to regard both as relative transforms to unity

        generate all permutations based on a list of transform operators
        what does the matrix represent? first col is arange
        on the one axis are domains, on the other axis are transforms
        the number in the table is the domain which the given domain/transform combo maps to
        """
        T = T.reshape(-1,3,3)

        perm = np.empty((len(T),)*2, np.int)
        for j, t in enumerate(T):
            for i,b in enumerate(T):
                q = np.dot(t, b)
                perm[i,j] = np.argmin([np.abs(q-qb).sum() for qb in T])
        return perm
    def relative(self, T):
        """
        find relative transformations; transform that will map one upon other

        generate all permutations based on a list of transform operators
        what does the matrix represent? first col is arange
        on the one axis are domains, on the other axis are transforms
        the number in the table is the domain which the given domain/transform combo maps to
        """
        T = T.reshape(-1,3,3)
        perm = np.empty((len(T),)*2, np.int)
        for j, t in enumerate(T):
            for i,b in enumerate(T):
                q = np.dot(t, b.T)
                perm[i,j] = np.argmin([np.abs(q-qb).sum() for qb in T])
        return perm




    @property
    def transform_shape(self):
        return self.transforms.shape[:2]
    def unravel(self, index):
        """convert linear transform index to mirror/rotation pair"""
        return np.unravel_index(index, self.transform_shape)
    def ravel(self, index):
        """convert mirror/rotation pair to linear transform index"""
        return np.ravel_multi_index(index, self.transform_shape)

    def permute_forward(self, a, b):
        a,b = self.ravel(a), self.ravel(b)
        return self.unravel(self.permutations[a, b])
    def permuted_basis(self, index, mr):
        """"""
        P = self.permutations[self.ravel(mr)]
        return self.basis[index].reshape((-1,3,3))[P].reshape(self.transform_shape+(3,3))


    def find_support(self, position):
        """
        find convex support for a list of points on the sphere.
        that means, returns the domain in which a point has only positive barycentric support; plus said baries
        essentially a brute force algo, but need to test against 120 domains max
        """
        position = np.atleast_2d(position)
        baries = np.dot(self.inverse, position.T)   #shape = i, m, r, 3, points
        neg = baries.copy()
        neg[neg>0] = 0
        neg = neg.sum(axis=-2)
        shape = neg.shape[:3]
        neg = neg.reshape(-1,len(position))
        linear_index = neg.argmax(axis=0)
        dindex = np.unravel_index(linear_index, shape)
        bary = baries[dindex[0],dindex[1],dindex[2],:,np.arange(len(position))]     #fancy indexing indeed...
        bary[bary<0] = 0
        return np.squeeze(dindex), np.squeeze(bary)


    def basis_from_domains(self, domains):
        """basis has shape i,m,r,c,pmd; not having components as last index is atypical"""
        shape = domains.shape[:-1]
        p,m,d = domains.reshape((-1,3)).T
        return np.dstack( (self.primal[p],  self.mid[m], self.dual[d])).reshape(shape +(3,3))

    def transforms_from_basis(self, basis):
        shape = basis.shape[:-2]
        basis = basis.reshape(-1, 3,3)
        """generate all linear transforms associated with this domain"""
        ib = np.linalg.inv(basis[0])
        return np.array( [np.dot(b, ib) for b in basis]).reshape(shape+(3,3))

    def orientation_from_basis(self, basis):
        shape = basis.shape[:-2]
        return np.sign( np.array( [np.linalg.det(b) for b in basis.reshape(-1,3,3)])).reshape(shape)




##class Symmetry(object):
##    """maybe just use a triplet of bools as a constraint? easier to pickle without version conflict"""
##    def __init__(self, description, constraints):
##        self.description = description + str(tuple( constraints))
##        self.constraints = np.array(constraints)
##
##symmetries = [Symmetry('primal', (1,0,0)), Symmetry('middle', (0,1,0)), Symmetry('dual  ', (0,0,1)),
##              Symmetry('mirror', (0,1,1)), Symmetry('mirror', (1,0,1)), Symmetry('mirror', (1,1,0))]



##from collections import OrderedDict
##constraints_dict = OrderedDict()
##for i, c in enumerate( constraints):
##    constraints_dict[c] = '{i}: {n}'.format(i=i,n=str(c))



##
##symdict[None] = 'None'
##for s in symmetries:
##    symdict[s] = s.description
##symdict = {s:s.description for s in symmetries}
