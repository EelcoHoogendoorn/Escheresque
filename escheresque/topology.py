"""
topology module
ordering of unknowns is based on recursive scheme
would not be too hard to make topology operators entirely implicit

constructing in terms of sparse operators is wrong; need to use index tables to enforce some required orderings
only construct sparse operators lazily, if needed for certain applications
in the end, sorted infex array provides fastest operator anyway
maintain well defined orderings.
edges are always opposite of vertices of a triangle, and thus both ordered
"""

import numpy as np

from escheresque import util

from scipy.sparse import csr_matrix

def sparse_matrix_from_array(index, weight=None):
    if weight is None:
        weight = np.ones_like(index)
    counter = np.arange(index.size)//index.shape[1]
    index = index.flatten()
    weight = weight.flatten().astype(np.float)
    return csr_matrix((weight, (index, counter)))


##class TransposedLoopup(object):
##    """
##    given a datastructure allowing fast boundary lookup, precompute fast incidence lookup information
##    """
##    def __init__(self, boundary, elements):
##        si = np.argsort(boundary.flatten())
##        self.pivots = np.searchsorted(boundary.flatten()[si], np.arange(elements+1))
##        I = np.arange(boundary.size) // boundary.shape[1]
##        self.si = I[si]
##    def __getitem__(self, ind):
##        """return an array of incident primitives"""
##        return self.si[self.pivots[ind]:self.pivots[ind+1]]



class Topology(object):
    """topology describing a single triangular fundamental domain
    it follows a hierarchical subdivision strategy; each triangle maps exactly to 4 subdivided triangles
    this is useful both for multigrid as well as collision detection

    need to add transfer operators between different levels of subdivision, for MG applications

    ordering information is kept metriculously
    """

    def __init__(self, position, FE, EV, FV):
        self.position = position
        self.FE = FE
        self.EV = EV
        self.FV = FV

        self.P2 = len(self.FEi)
        self.P1 = len(self.EVi)
        self.P0 = len(position)

        self.D0 = self.P2
        self.D1 = self.P1
        self.D2 = self.P0

        #maybe these should be constructed lazily, on demand? only needed ocassionally anyway, right? no, FV is needed in normal calc. could use scatter op here though...
        self._T01 = sparse_matrix_from_array(*EV)
        self._T12 = sparse_matrix_from_array(*FE)
        self.T02  = sparse_matrix_from_array(self.FV)

        #precompute boundary indices
        self.BV = tuple(bv.astype(np.int32) for bv in np.arange(3).reshape((3,1)))
        self.BE = tuple( [np.nonzero(p==0)[0][2:].astype(np.int32) for p in position.T])

        self.transfer_operators()


    def subdivide_topology(self):
        """
        create the next level of subdivision

        fully vectorized version
        this has been a holy grail of mine for quite some time
        would make for a nice programming challenge methinks
        """
        #positive and negatively rolled FE info is needed several times; compute once
        FEp = np.roll(self.FE, +1, axis=2)
        FEm = np.roll(self.FE, -1, axis=2)

        #new positions in bary-space; one for each point, one for each edge
        vP = self.position
        eP = self.edge_mid * self.position

        #create edge edges; two for each parent edge
        eEV  = np.empty((2, self.P1, 2, 2), np.int)
        new_edge_vertex = np.arange(self.P1) + self.P0
        eEV[0,:,0,0] = self.EV[0,:,0]
        eEV[0,:,0,1] = new_edge_vertex
        eEV[0,:,1,0] = new_edge_vertex
        eEV[0,:,1,1] = self.EV[0,:,1]
        #edge-edge sign info
        eEV[1,:,0,0] = +self.EV[1,:,0]
        eEV[1,:,0,1] = -self.EV[1,:,0]
        eEV[1,:,1,0] = -self.EV[1,:,1]
        eEV[1,:,1,1] = +self.EV[1,:,1]

        #3 new edges per face
        fEV = np.empty((2, self.P2, 3, 2), np.int)      #edge-vertex info added as consequence of faces
        fEV[0,:,:,0] = FEm[0] + self.P0     #edge-vertices can be found by rotating and offsetting parent edges
        fEV[0,:,:,1] = FEp[0] + self.P0
        fEV[1,:,:,0] = -1                   # this is a sign convention we are free to choose; neg-to-pos edges are maintained globally
        fEV[1,:,:,1] = +1

        #4 new faces per face
        fFE  = np.empty((2, self.P2, 4, 3), np.int)

        #add middel (zero) tri
        fFE[0,:,0,:] = np.arange(3*self.P2).reshape((self.P2, 3)) + self.P1*2     #add middle connectivity; three new edges for each face
        fFE[1,:,0,:] = 1                                  #orientation of middle tri is given by convention

        #first edge of each outer tri is connected to the middle; copy index, invert sign
        fFE[0,:,1:,0] =  fFE[0,:,0,:]
        fFE[1,:,1:,0] = -fFE[1,:,0,:]

        #now do hard part; hook up corner triangles
        plusverts = self.EVi[self.FEi,1]
        #getting the right indices requires some logic; look up if triangle vertex is on the plus end of edge or not; if so, it has the higher (uneven) index
        fFE[0,:,1:,+1] = FEm[0]*2 + (np.roll(plusverts, -1, axis=1) == self.FV)*1
        fFE[0,:,1:,-1] = FEp[0]*2 + (np.roll(plusverts, +1, axis=1) == self.FV)*1
        #weights are simply inherited, using the same roll logic
        fFE[1,:,1:,+1] = FEm[1]
        fFE[1,:,1:,-1] = FEp[1]

        #FV is implied by FE; but much easier to calc by subdivision too
        FV = np.empty((self.P2, 4, 3), np.int)
        FV[:,0 , :] = self.FE[0] + self.P0      #middle tri; translate parent edge index to edge-vertex
        FV[:,1:, 0] = self.FV                   #corner tri; inherit from parent
        FV[:,1:,+1] = FEp[0] + self.P0          #middle edge connections; same as fEV; rotate parent edge, translate to edge-vert
        FV[:,1:,-1] = FEm[0] + self.P0


        return Topology(
            position = np.concatenate( (vP, eP), axis=0),
            FE       = fFE.reshape(2,-1,3),
            EV       = np.concatenate((
                            eEV.reshape(2,-1,2),
                            fEV.reshape(2,-1,2)), axis=1),
            FV       = FV.reshape(-1,3)
            )

    @property
    def edge_mid(self):
        """edge midpoint operator"""
        return np.abs(self.T10)/2


    def subdivide_position(self, position):
        """calc primal coords from parent"""
        #one child for each parent
        vertex_vertex = position
        #each new vert lies at midpoint
        edge_vertex = util.normalize(self.edge_mid*vertex_vertex)
        #ordering convention is vertex-vertex + edge-vertex
        position = np.vstack((vertex_vertex, edge_vertex))

        #calc subdivision planes
        central = util.grab(self.FEi, edge_vertex)
        planes  = util.adjoint(central)

        return position, planes


    def transfer_operators(self):
        """
        generate transfer operators for the given layer, assuming the given conventions
        interpolation is identity concatted with averaging matrix

        how to organize mg?
        maintain complex hierarchy? this is the only place we truely need it no?
        otoh, with geometry all required info is first available no?
        """
        vertex_vertex = util.coo_matrix( util.dia_simple(np.ones(self.P0)))
        edge_vertex   = util.coo_matrix( self.edge_mid)
        edge_vertex   = util.coo_shift( edge_vertex,self.P0, 0)

        self.interpolation = util.coo_append(vertex_vertex, edge_vertex)
        self.restriction   = self.interpolation.T
        self.weighting     = self.restriction * np.ones(self.P0+self.P1)      #redution of operator






    @property
    def EVi(self):
        return self.EV[0]
    @property
    def EVs(self):
        return self.EV[1]
    @property
    def FEi(self):
        return self.FE[0]
    @property
    def FEs(self):
        return self.FE[1]


    @property
    def T01(self):
        """right multiplication gives the boundary of the edge chain"""
        return self._T01
    @property
    def T12(self):
        """right multiplication gives the boundary of the face chain"""
        return self._T12
    @property
    def T10(self):
        """right multiplication gives the edges indicent to the given vertices"""
        return self._T01.T
    @property
    def T21(self):
        """right multiplication gives the faces indicent to the given edges"""
        return self._T12.T

    @property
    def P21(self):
        """discrete differential operator; divergence of flux in P1 to faces in P2"""
        return self.T21
    @property
    def P10(self):
        """discrete differential operator; gradient of scalar in P0 to edges in P1"""
        return self.T10
    @property
    def D21(self):
        """discrete differential operator; curl of flux in D1 around faces in D2"""
        return self.T01
    @property
    def D10(self):
        """discrete differential operator; curl of scalar in D0 to edges in D1"""
        return self.T12



    @staticmethod
    def triangle():
        """create a triangle with standard orientation"""
        P = np.eye(3)
        FE = np.array([[0,1,2]]), np.array([[1,1,1]])
        EV = np.array([[1,2],[2,0],[0,1]]), np.array([[-1,+1],[-1,+1],[-1,+1]])
        FV = np.array([[0,1,2]])
        return Topology(P, np.array(FE), np.array(EV), FV)




def generate(levels):
    """
    generate a topology hierachy by subdivision.
    """
    T = [Topology.triangle()]
    for i in range(levels):
        t = T[-1].subdivide_topology()
        T.append(t)

    return T


