"""
geometry classes
take topology object
plus a set of triangle coordinates


seperate hierarchy/mg stuff
basic geom class should be able to stand by itself
hierachy should be added on externaly
"""

import numpy as np
from . import util



##def tri_area(tri):
##    edge = util.normalize( np.cross(tri, tri[[1,2,0]]))     #edge planes
##    angles = -np.sum(edge*edge[[1,2,0]], axis=1)            #dot of edge plane with neighbors
##    return np.arccos( angles).sum()-np.pi
##def edge_length(edge):
##    return np.arccos( np.linalg.norm(np.dot(*edge)))

def edge_length_vec(*edge):
    """assumes a pair of noramlized Vx3 matrices"""
    return np.arccos(util.dot(*edge))
def tri_area_vec(*tri):
    edge_planes = [util.normalize(np.cross(tri[v-2], tri[v-1])) for v in xrange(3)] #p3 x [faces, c3]
    angles      = [util.dot(edge_planes[p-2], edge_planes[p-1]) for p in xrange(3)] #a3 x [faces, c3]
    areas       = sum(np.arccos(-a) for a in angles) - np.pi                        #faces
    return areas




class Geometry(object):
    """
    describes the geometry of a single fundamental domain triangle
    """

    def __init__(self, topology, primal, planes):
        """topology plus primal positions gives a geometry"""
        self.topology = topology
        self.primal = primal
        self.planes = planes

        self.dual_position()
        self.metric()


    def dual_position(self):
        """calc dual coords from primal; interestingly, this is idential to computing a triangle normal"""
        #calc direction orthogonal to normal of intesecting plane
        diff = self.topology.T10 * self.primal
        #collect these on per-tri basis, including weights, so ordering is correct
        tri_edge = util.grab(self.topology.FEi, diff) * self.topology.FEs[:,:,None]
        #for above, could also do cyclical diff on grab(FV, primal)
        #dual vert les where three mid edge planes intesect
        self.dual = util.normalize(-util.null(tri_edge))
##        midpoint = util.grab(self.topology.FV, self.primal).sum(axis=1)
##        print (self.dual* midpoint).sum(axis=1)

    def decompose(self):
        """find decomposition of coords onto corner points. this is useful for mapping to other domains without floating point 'leaks' in the mesh"""
        primal = self.primal
        corner = primal[:3]
        PP = np.dot( np.linalg.inv(corner).T, primal.T).T
        #this is an important constraint, to get edges free of numerical leaks; points on edge must be fuction of egd
        PP[self.topology.position==0] = 0
        self.decomposed = PP

    def triangle_invert(self):
        """
        pre-invert triangle support. (adjoint is fine; we normalize coords anyway)
        """
        tri_coords = util.grab(self.topology.FV, self.primal)
        self.inverted_triangle = util.adjoint(tri_coords)


    def metric(self):
        """
        calc metric properties and hodges; nicely vectorized
        """
        topology = self.topology

        #metrics
        MP0 = np.ones (topology.P0)
        MP1 = np.zeros(topology.P1)
        MP2 = np.zeros(topology.P2)
        MD0 = np.ones (topology.D0)
        MD1 = np.zeros(topology.D1)
        MD2 = np.zeros(topology.D2)

        #precomputations
        EVP  = util.grab(topology.EVi, self.primal)
        FEVP = util.grab(topology.FEi, EVP)         #[faces, e3, v2, c3]
        FEM  = util.normalize(FEVP.sum(axis=2))
        FEV  = util.grab(topology.FEi, topology.EVi)

        #calculate areas; devectorization over e makes things a little more elegant, by avoiding superfluous stacking
        for e in xrange(3):
            areas = tri_area_vec(FEVP[:,e,0,:], FEVP[:,e,1,:], self.dual)
            MP2 += areas                    #add contribution to primal face
            util.scatter(                   #add contributions divided over left and right dual face
                FEV[:,e,:],                 #get both verts of each edge
                np.repeat(areas/2, 2),      #half of domain area for both verts
                MD2)

        #calc edge lengths
        MP1 += edge_length_vec(EVP[:,0,:], EVP[:,1,:])
        for e in xrange(3):
            util.scatter(
                topology.FEi[:,e],
                edge_length_vec(FEM[:,e,:], self.dual),
                MD1)

        #hodge operators
        self.D2P0 = MD2 / MP0
        self.P0D2 = MP0 / MD2

        self.D1P1 = MD1 / MP1
        self.P1D1 = MP1 / MD1

        self.D0P2 = MD0 / MP2
        self.P2D0 = MP2 / MD0



from . import topology
def generate(group, levels):
    """
    create geometry hierarchy from topology hierarchy and pmd basis

    these computations are hard to cache, since they are different for each group,
    so there are too many possible combinations to make this appealing
    as such, this code could use some more optimization
    """
    T = topology.generate(levels)

    pmd = util.normalize( group.basis[0,0,0].T)

    _t = T[0]
    geom = [Geometry(_t, pmd, None)]
    _g = geom[0]
    for t in T[1:]:
        position, planes = _t.subdivide_position(_g.primal)
        _g = Geometry(t, position, planes)
        _t = t
        geom.append(_g)

    #for picking and rendering
    _g.decompose()
    _g.triangle_invert()

    return geom


