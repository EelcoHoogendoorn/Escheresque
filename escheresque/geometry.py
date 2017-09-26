"""geometry class

The geometry class combined the topology of a single fundamental triangle,
with a set of coordinates for the primal vertices

From this, metric properties on the sphere are computed, such as the
primal-dual transfer operators, as well as the multigrid transfer operators
"""

import numpy as np
import numpy_indexed as npi
from escheresque import util


def edge_length(*edge):
    """compute spherical edge length.

    Parameters
    ----------
    edge : 2 x ndarray, [n, 3], float
        arc segments described by their start and end position

    Returns
    -------
    lengths: ndarray, [n], float
        length along the unit sphere of each segment
    """
    return np.arccos(util.dot(*edge))

def triangle_area_from_normals(*edge_planes):
    """compute spherical area from triplet of great circles

    Parameters
    ----------
    edge_planes : 3 x ndarray, [n, 3], float
        edge normal vectors of great circles

    Returns
    -------
    areas : ndarray, [n], float
        spherical area enclosed by the input planes
    """
    edge_planes = [util.normalize(ep) for ep in edge_planes ]
    angles      = [util.dot(edge_planes[p-2], edge_planes[p-1]) for p in range(3)] #a3 x [faces, c3]
    areas       = sum(np.arccos(-a) for a in angles) - np.pi                        #faces
    return areas

def triangle_area_from_corners(*tri):
    """compute spherical area from triplet of triangle corners

    Parameters
    ----------
    tri : 3 x ndarray, [n, 3], float
        corners of each triangle

    Returns
    -------
    areas : ndarray, [n], float
        spherical area enclosed by the input corners
    """
    return triangle_area_from_normals(*[np.cross(tri[v-2], tri[v-1]) for v in range(3)])

def triangle_areas_around_center(center, corners):
    """given a triangle formed by corners, and its dual point center,
    compute spherical area of the voronoi faces

    Parameters
    ----------
    center : ndarray, [..., 3], float
    corners : ndarray, [..., 3, 3], float

    Returns
    -------
    areas : ndarray, [..., 3], float
        spherical area opposite to each corner
    """
    areas = np.empty(corners.shape[:-1])
    for i in range(3):
        areas[:,:,i] = triangle_area_from_corners(center, corners[:,:,i-2], corners[:,:,i-1])
    #swivel equilaterals to vonoroi parts
    return (areas.sum(axis=2)[:,:,None]-areas) / 2


class Geometry(object):
    """
    describes the geometry of a single fundamental domain triangle
    geometry is idnetical between fundamental domains
    """

    def __init__(self, topology, primal, planes):
        """topology plus primal positions defines a geometry"""
        self.topology = topology
        self.primal = primal
        self.planes = planes

        self.dual_position()
        self.metric()
        self.decompose()
        self.triangle_invert()

    def dual_position(self):
        """calc dual coords from primal; interestingly, this is idential to computing a triangle normal"""
        #calc direction orthogonal to normal of intesecting plane
        diff = self.topology.T10 * self.primal
        #collect these on per-tri basis, including weights, so ordering is correct
        tri_edge = util.gather(self.topology.FEi, diff) * self.topology.FEs[:, :, None]
        #for above, could also do cyclical diff on grab(FV, primal)
        #dual vert les where three mid edge planes intesect
        self.dual = util.normalize(-util.null(tri_edge))
##        midpoint = util.grab(self.topology.FV, self.primal).sum(axis=1)
##        print (self.dual* midpoint).sum(axis=1)

    def decompose(self):
        """
        find decomposition of primal coords onto corner points.
        this is useful for mapping to other domains without floating point 'leaks' in the mesh
        """
        primal = self.primal
        corner = primal[:3]
        PP = np.dot( np.linalg.inv(corner).T, primal.T).T
        #this is an important constraint, to get edges free of numerical leaks; points on edge must be fuction of egd
        PP[self.topology.position==0] = 0
        self.decomposed = PP

    def triangle_invert(self):
        """
        pre-invert triangle support.
        this is useful for computing projections on triangles
        adjoint is fine; we normalize coords anyway
        """
        tri_coords = util.gather(self.topology.FV, self.primal)
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
        EVP  = util.gather(topology.EVi, self.primal)
        FEVP = util.gather(topology.FEi, EVP)         #[faces, e3, v2, c3]
        FEM  = util.normalize(FEVP.sum(axis=2))
        FEV  = util.gather(topology.FEi, topology.EVi)

        #calculate areas; devectorization over e makes things a little more elegant, by avoiding superfluous stacking
        for e in range(3):
            areas = triangle_area_from_corners(FEVP[:,e,0,:], FEVP[:,e,1,:], self.dual)
            MP2 += areas                    #add contribution to primal face
            util.scatter(                   #add contributions divided over left and right dual face
                FEV[:,e,:],                 #get both verts of each edge
                np.repeat(areas/2, 2),      #half of domain area for both verts
                MD2)

        #calc edge lengths
        MP1 += edge_length(EVP[:,0,:], EVP[:,1,:])
        for e in range(3):
            util.scatter(
                topology.FEi[:,e],
                edge_length(FEM[:,e,:], self.dual),
                MD1)

        #hodge operators
        self.D2P0 = MD2 / MP0
        self.P0D2 = MP0 / MD2

        self.D1P1 = MD1 / MP1
        self.P1D1 = MP1 / MD1

        self.D0P2 = MD0 / MP2
        self.P2D0 = MP2 / MD0



    def transfer_operators(self):
        """
        construct metric transfer operators, as required for dual-transfer on pseudo-regular grid
        we need to calculate overlap between fine and coarse dual domains

        the crux here is in the treatment of the central triangle

        holy shitballs this is a dense function.
        there is some cleanup i could do, but this is also simply some insanely hardcore shit

        algebraicly optimal multigrid transfer operators on a pseudo-regular grid, here we come
        """
        coarse = self
        fine = self.child


        all_tris = np.arange(fine.topology.P2).reshape(coarse.topology.P2, 4)
        central_tris = all_tris[:,0]
        corner_tris  = all_tris[:,1:]
        #first, compute contribution to transfer matrices from the central refined triangle

        coarse_dual   = coarse.dual
        fine_dual     = fine.dual[central_tris]
        face_edge_mid = util.gather(fine.topology.FV[0::4], fine.primal)

        fine_edge_normal = [np.cross(face_edge_mid[:,i-2,:], face_edge_mid[:,i-1,:]) for i in range(3)]
        fine_edge_mid    = [(face_edge_mid[:,i-2,:] + face_edge_mid[:,i-1,:])/2      for i in range(3)]
        fine_edge_dual   = [np.cross(fine_edge_mid[i], fine_edge_normal[i])          for i in range(3)]
        fine_edge_normal = np.array(fine_edge_normal)
        fine_edge_mid    = np.array(fine_edge_mid)
        fine_edge_dual   = np.array(fine_edge_dual)

        coarse_areas     = [triangle_area_from_corners(coarse_dual, face_edge_mid[:,i-2,:], face_edge_mid[:,i-1,:]) for i in range(3)]
        fine_areas       = [triangle_area_from_corners(fine_dual  , face_edge_mid[:,i-2,:], face_edge_mid[:,i-1,:]) for i in range(3)]
        fine_areas       = [(fine_areas[i-2]+fine_areas[i-1])/2 for i in range(3)]
        coarse_areas     = np.array(coarse_areas)
        fine_areas       = np.array(fine_areas)

        #normal of edge midpoints to coarse dual
        interior_normal = np.array([np.cross(face_edge_mid[:,i,:], coarse_dual) for i in range(3)])

        #the 0-3 index of the overlapping domains
        #biggest of the subtris formed with the coarse dual vertex seems to work; but cant prove why it is so...
        touching = np.argmax(coarse_areas, axis=0)
##        print touching
##        print fine_areas
##        print coarse_areas

        #indexing arrays
        I = np.arange(len(touching))
        m = touching        #middle pair
        l = touching-1      #left-rotated pair
        r = touching-2      #right-rotated pair

        #compute sliver triangles
        sliver_r = triangle_area_from_normals(
            +fine_edge_normal[l, I],
            +fine_edge_dual  [l, I],
            +interior_normal [r, I])
        sliver_l = triangle_area_from_normals(
            +fine_edge_normal[r, I],
            -fine_edge_dual  [r, I],
            -interior_normal [l, I])

##        print 'slivers'
##        print sliver_l
##        print sliver_r

        assert(np.all(sliver_l>-1e-10))
        assert(np.all(sliver_r>-1e-10))


        #assemble area contributions of the middle triangle
        areas = np.empty((len(touching),3,3))     #coarsetris x coarsevert x finevert
        #the non-overlapping parts
        areas[I,l,l] = 0
        areas[I,r,r] = 0
        #triangular slivers disjoint from the m,m intersection
        areas[I,r,l] = sliver_l
        areas[I,l,r] = sliver_r
        #subset of coarse tri bounding sliver
        areas[I,r,m] = coarse_areas[r,I] - sliver_l
        areas[I,l,m] = coarse_areas[l,I] - sliver_r
        #subset of fine tri bounding sliver
        areas[I,m,l] = fine_areas[l,I] - sliver_l
        areas[I,m,r] = fine_areas[r,I] - sliver_r
        #square middle region; may compute as fine or caorse minus its flanking parts
        areas[I,m,m] = coarse_areas[m,I] - areas[I,m,l] - areas[I,m,r]

        #we may get numerical negativity for 2x2x2 symmetry, with equilateral fundemantal domain,
        #or high subdivision levels. or is error at high subdivision due to failing of touching logic?
        assert(np.all(areas > -1e-10))

        #areas maps between coarse vertices and fine edge vertices.
        #add mapping for coarse to fine vertices too

        #need to grab coarsetri x 3coarsevert x 3finevert arrays of coarse and fine vertices
        fine_vertex   = np.repeat( fine  .topology.FV[0::4, None,    :], 3, axis=1)
        coarse_vertex = np.repeat( coarse.topology.FV[:   , :   , None], 3, axis=2)

        def coo_matrix(data, row, col):
            """construct a coo_matrix from data and index arrays"""
            return util.coo_matrix(
                (data.ravel(),(row.ravel(), col.ravel())),
                shape=(coarse.topology.D2, fine.topology.D2))

        center_transfer = coo_matrix(areas, coarse_vertex, fine_vertex)


        #add corner triangle contributions; this is relatively easy
        #coarsetri x 3coarsevert x 3finevert
        corner_vertex = util.gather(corner_tris, fine.topology.FV)
        corner_dual   = util.gather(corner_tris, fine.dual)
        corner_primal = util.gather(corner_vertex, fine.primal)

        #coarsetri x 3coarsevert x 3finevert
        corner_areas    = triangle_areas_around_center(corner_dual, corner_primal)
        #construct matrix
        corner_transfer = coo_matrix(corner_areas, coarse_vertex, corner_vertex)
        self.transfer = util.csr_matrix(center_transfer + corner_transfer)

        #calc normalizations
        self.coarse_area = self.transfer   * np.ones(fine  .topology.D2)
        self.fine_area   = self.transfer.T * np.ones(coarse.topology.D2)

        self.f = np.sqrt( self.fine_area)[:,None]
        self.c = np.sqrt( self.coarse_area)[:,None]

        #test for consistency with metric calculations
        assert(np.allclose(self.coarse_area, coarse.D2P0, 1e-10))
        assert(np.allclose(self.fine_area  , fine  .D2P0, 1e-10))



    def restrict_d2(self, x):
        return (self.transfer   * (x / self.fine_area  [:,None]))
    def interpolate_d2(self, x):
        return (self.transfer.T * (x / self.coarse_area[:,None]))
##    def interpolate_d2(self, x):
##        return (self.transfer.T * (x / self.c)) / self.f
##    def interpolate_d2(self, x):
##        return (self.transfer.T * x) / self.fine_area[:,None]

    def generate_vertices(self, group):
        """instantiate a full sphere by repeating the transformed fundamental domain

        Returns
        -------
        ndarray, [n, 3], float
            all points in the geometry, on a unit sphere
        """
        points = np.empty((group.index, group.order, self.topology.P0, 3), np.float)
        PP = self.decomposed
        for i, B in enumerate(group.basis):
            for t, b in enumerate(B.reshape(-1, 3, 3)):
                b = util.normalize(b.T).T  # now every row is a normalized vertex
                P = np.dot(b, PP.T).T  # go from decomposed coords to local coordinate system
                points[i, t] = P

        # make single unique point list
        return npi.unique(points.reshape(-1, 3))


from escheresque import topology
def generate(group, levels):
    """
    create geometry hierarchy from topology hierarchy
    and group and a pmd basis of its first fundamental domain
    """
    T = topology.generate(levels)

    pmd = util.normalize( group.basis[0,0,0].T)

    _t = T[0]
    G = [Geometry(_t, pmd, None)]
    _g = G[0]
    for t in T[1:]:
        position, planes = _t.subdivide_position(_g.primal)
        _g = Geometry(t, position, planes)
        _t = t
        G.append(_g)

    #hook up parent-child relations in geometry list
    for parent,child in zip(G[:-1],G[1:]):
        parent.child = child
        child.parent = parent
        parent.transfer_operators()


    return G


