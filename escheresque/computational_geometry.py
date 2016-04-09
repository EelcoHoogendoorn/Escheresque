
"""
mesh cutting and handling functionality
this module performs constrained triangulation of a spherical mesh, partitioned by
an arbitrary set of non-intersecting curves on said sphere

im a very pleased with how simple yet robust this piece of code has turned out to be
no external libraries are used, mesh quality is top notch, there is no risk of numerical errors,
the code is squeeky clean, it has every feature i want,
and performance is of optimal time complexity and reasoable in an absolute sense

while stuggeling with this problem, i would at some point have settled for any single one of the traits in this list,
so to get them all in the end, makes me one very happy programmer

the crux here is in bootstrapping scipy's convex hull functionality to do our triangulation for us
the convex hull of a set of spherical points is also a delauney triangulation.
and with a little cleverness, even the constrained triangulation we are after.
constrained triangulation is a messy problem, and on the sphere ne might think itd soon be nightmarish,
but by exploiting this duality we can kill two t-rexes with one pebble.
"""

from itertools import izip

import numpy as np
import numpy_indexed as npi

from scipy.spatial import cKDTree as KDTree

from escheresque import util



def merge_geometry(points, idx=None):
    """marge identical points in an NxM point array, reducing an indiex array into the point array at the same time"""
    _, _idx, _inv = npi.unique(points, return_index=True, return_inverse=True)
    if idx is None:
        return points[_idx]
    else:
        return points[_idx], _inv[idx]


def squeeze_geometry(points, idx):
    """compact geometry description, removing unused points, and adjusting index accordingly"""
    active_points, inv = np.unique(idx, return_inverse=True)
    return points[active_points], inv.reshape(idx.shape)





class TransposedLoopup(object):
    """
    given a datastructure allowing fast boundary lookup, precompute fast incidence lookup information
    """
    def __init__(self, boundary, elements):
        si = np.argsort(boundary.flatten())
        self.pivots = np.searchsorted(boundary.flatten()[si], np.arange(elements+1))
        self.si = (np.arange(boundary.size) // boundary.shape[1])[si]
    def __getitem__(self, ind):
        """return an array of incident primitives; edges, given vertex"""
        return self.si[self.pivots[ind]:self.pivots[ind+1]]



def trim_curve(points, curve, curve_idx, radius):
    """
    remove points too close to the cut curve. they dont add anything, and only lead to awkward triangles
    """
    #some precomputations
    tree   = KDTree(points)
    cp     = curve[curve_idx]
    normal = util.normalize(np.cross(cp[:,0], cp[:,1]))
    mid    = util.normalize(cp.sum(axis=1))
    diff   = np.diff(cp, axis=1)[:,0,:]
    edge_radius = np.sqrt(util.dot(diff, diff)/4 + radius**2)

    index = np.ones(len(points), np.bool)

    #eliminate near edges
    def near_edge(e, p):
        return np.abs(np.dot(points[p]-mid[e], normal[e])) < radius
    for i,(p,r) in enumerate(izip(mid, edge_radius)):
        coarse = tree.query_ball_point(p, r)
        index[[c for c in coarse if near_edge(i, c)]] = 0
    #eliminate near points
    for p in curve:
        coarse = tree.query_ball_point(p, radius)
        index[coarse] = 0

    return points[index]





def intersect_curve(curve_p, curve_idx):
    """
    test curve of arc-segments for intersection
    raises exception in case of intersection
    alternatively, we might resolve intersections by point insertion
    but this is unlikely to have any practical utility, and more likely to be annoying
    """
    tree   = KDTree(curve_p)
    cp     = util.gather(curve_idx, curve_p)
    normal = util.normalize(np.cross(cp[:,0], cp[:,1]))
    mid    = util.normalize(cp.sum(axis=1))
    diff   = np.diff(cp, axis=1)[:,0,:]
    radius = np.sqrt(util.dot(diff, diff)) / 2 * 1.01

    projector = [np.linalg.pinv(q.T) for q in cp]

    incident = TransposedLoopup(curve_idx, len(curve_p))      #each vertex may have an aribtrary number of edges

    def intersect(i,j):
        """test if spherical line segments intersect. bretty elegant"""
        intersection = np.cross(normal[i], normal[j])                               #intersection direction of two great circles; sign may go either way though!
        return all(np.prod(np.dot(projector[e], intersection)) > 0 for e in (i,j))  #this direction must lie within the cone spanned by both sets of endpoints
    for ei,(p,r,cidx) in enumerate(izip(mid, radius, curve_idx)):
        V = [v for v in tree.query_ball_point(p, r) if v not in cidx]
        edges = np.unique([ej for v in V for ej in incident[v]])
        for ej in edges:
            if len(np.intersect1d(curve_idx[ei], curve_idx[ej])) == 0:      #does not count if edges touch
                if intersect(ei, ej):
                    raise Exception('The boundary curves intersect. Check your geometry and try again')




def refine_curve(points, curve_p, curve_idx):
    """
    refine the contour such as to maintain it as a constrained boundary under triangulation using a convex hull
    this is really the crux of the method pursued in this module
    we need to 'shield off' any points that lie so close to the edge such as to threaten our constrained boundary
    by adding a split at the projection of the point on the line, for all vertices within the swept circle of the edge,
    we may guarantee that a subsequent convex hull of the sphere respects our original boundary
    """
    allpoints = np.vstack((curve_p, points))
    tree = KDTree(allpoints)

    cp     = util.gather(curve_idx, curve_p)
    normal = util.normalize(np.cross(cp[:,0], cp[:,1]))
    mid    = util.normalize(cp.sum(axis=1))
    diff   = np.diff(cp, axis=1)[:,0,:]
    radius = np.sqrt(util.dot(diff, diff)) / 2

    def insertion_point(e, c):
        """calculate insertion point"""
        coeff = np.dot( np.linalg.pinv(cp[e].T), allpoints[c])
        coeff = coeff / coeff.sum()
        return coeff[0], np.dot(cp[e].T, coeff)
##        n = normal[e]
##        p = allpoints[c]
##        newp = p - n * np.dot(p,n)
##        delta = diff[e]
##        d = np.dot(delta, newp-cp[e,0]) / np.dot(delta, delta)
##        return d, newp #/ np.linalg.norm(newp)

    #build new curves
    _curve_p = [c for c in curve_p]
    _curve_idx = []
    for e,(m,r,cidx) in enumerate(izip( mid, radius, curve_idx)):
        try:
            d,ip = min(     #codepath for use in iterative scheme; only insert the most balanced split; probably makes more awkward ones obsolete anyway
                [insertion_point(e,v) for v in tree.query_ball_point(m, r) if not v in cidx],
                key=lambda x:(x[0]-0.5)**2)     #sort on distance from midpoint
            nidx = len(_curve_p)
            _curve_idx.append((cidx[0], nidx))  #attach on both ends
            _curve_idx.append((nidx, cidx[1]))
            _curve_p.append(ip)                   #append insertion point
        except:
            _curve_idx.append(cidx)             #if edge is not split, just copy it

    return np.array(_curve_p), np.array(_curve_idx)


def triangulate(points, curve, curve_idx):
    """
    return a triangulation of the pointset points,
    while being constrained by the boundary dicated by curve
    """
    #test curve for self-intersection
    print 'testing curve for self-intersection'
    intersect_curve(curve, curve_idx)

    #trim the pointset, to eliminate points co-linear with the cutting curve
    print 'trimming dataset'
    diff   = np.diff(curve[curve_idx], axis=1)[:,0,:]
    length = np.sqrt(util.dot(diff, diff))
    points = trim_curve(points, curve, curve_idx, length.mean()/4)

    #refine curve iteratively. new points may both obsolete or require novel insertions themselves
    #so only do the most pressing ones first, then iterate to convergence
    old_curve = curve
    old_curve_idx = curve_idx
    for i in range(10):
        newcurve, newcurve_idx = refine_curve(points, curve, curve_idx)
        #print len(newcurve)
        if len(newcurve)==len(curve):
            break
        print 'curve refined'
        curve, curve_idx = newcurve, newcurve_idx


    """
    we use the nifty property, that a convex hull of a sphere equals a delauney triangulation of its surface
    if we have cleverly refined our boundary curve, this trinagulation should also be 'constrained', in the sense
    of respecting that original boundary curve
    this is the most computationally expensive part of this function, but we should be done in a minute or so

    qhull performance; need 51 sec and 2.7gb for 4M points
    that corresponds to an icosahedron with level 8 subdivision; not too bad
    editor is very unresponsive at this level anyway
    """
    print 'triangulating'
    allpoints = np.vstack((curve, points))    #include origin; facilitates clipping
    from scipy.spatial import ConvexHull
    hull = ConvexHull(util.normalize(allpoints))
    triangles = hull.simplices

    #order triangles coming from the convex hull
    print 'ordering triangles'
    FP        = util.gather(triangles, allpoints)
    mid       = FP.sum(axis=1)
#    normal    = util.adjoint( FP - np.roll(FP, 1, axis=1)).sum(axis=1)
    normal    = util.normals(FP)
    sign      = util.dot(normal, mid) > 0
    triangles = np.where(sign[:,None], triangles[:,::+1], triangles[:,::-1])

    return allpoints, triangles, curve, curve_idx




def triangle_edges(triangles):
    """construct 3 edges for each triangle"""
    edges = np.empty((len(triangles), 3, 2), np.int32)
    for i in range(3):
        edges[:,i,0] = triangles[:,i-2]
        edges[:,i,1] = triangles[:,i-1]
    return edges.reshape(-1,2)
def order_edges(edges):
    """place edges in standarized order. is this actually faster than sorting over last axis?"""
    return np.where((edges[:,0]<edges[:,1])[:,None], edges[:,::+1], edges[:,::-1])
##def label_edges(idx):
##    """convert Nx2 edge index array to N unique labels. could reaplce with voidview if searchosrted works on it"""
##    return idx.astype(np.int32).ravel().view(np.int64).reshape(idx.shape[:-1])
# def label_edges(edges):
#     """return unique bytecode for each edge, independent of vertex ordering"""
#     return voidview(order_edges(edges))


def partition(points, triangles, curve):
    """
    triangles is a list of vertex-index triples
    curve is an index array of vertex-doublets

    the triangles and curve need to be consistent; that is, any two adjecent curve points do need to be connected by two tris
    and the triangulation needs to be closed; no open edges

    returned is a list of triangle arrays, as partitioned by the curve

    brute force floodfillling dominates for larger meshes; this is the only O(N^2) step;
    and apparently N is big enough to make it noticable
    we may also use np.maximum.at type functionality, no?
    this may be more efficient, especially on a sparse vector
    work with a subset of sparse elements?
    need a way to efficiently update the active set
    expand by newly updated idx
    from expanded set, trim those that didnt change
    or do pointer based method?
    multigrid-type method would be ideal
    iterate a few steps, then find subgraphs by building dict of pairs
    """
    print 'partitioning mesh'

    #construct edge information

    unique, inverse = npi.unique(
        order_edges(triangle_edges(triangles)),
        return_inverse = True)

    FE =            inverse .reshape(-1,3)
    EF = np.argsort(inverse).reshape(-1,2) // 3   #this gives incidence relations

    #determine which edges are curve edges
    # curve_edges = np.searchsorted(unique, label_edges(curve))
    curve_edges = npi.indices(unique, order_edges(curve))


    if True:
        def associative_floodfill():
            """
            not optimally efficient, but at least should run in linear time
            """
            curve_edges_set = set(curve_edges)
            unused = set(np.arange(len(FE)))
            used = set()
            parts = []
            while len(unused):
                active = set([unused.pop()])
                part   = set(active)
                while len(active):
                    edges  = set(FE[list(active)].flatten()).difference(curve_edges_set)
                    active = set(EF[list(edges )].flatten()).difference(part)
                    part = part.union(active)
                parts.append(part)
                print 'part found'
                unused = unused.difference(part)
            return parts

        partitions = [squeeze_geometry(points, triangles[list(p)]) for p in associative_floodfill()]
    else:
        """
        floodfill, respecting boundaries. numerically robust and fast way to do segmentation (as opposed to point-in-poly tests)
        however, this version has superlinear time complexity
        """
        tri_labels = np.arange(len(triangles), dtype=np.int32)
        while True:
            edge_labels = tri_labels[EF].max(axis=1)
            edge_labels[curve_edges] = 0                    #halt propagation at boundary
            new_labels = edge_labels[FE].max(axis=1)
            print np.count_nonzero(new_labels!=tri_labels)
            if np.alltrue(new_labels==tri_labels): break    #convergence condition
            tri_labels = new_labels

        #return triangle array, split in seperate arrays
        partitions = [squeeze_geometry(points, triangles[tri_labels==l]) for l in np.unique(tri_labels)]

    return sorted(partitions, key=lambda p:len(p[0]))

def multiplicity(keys):
    unique, inverse = np.unique(keys, return_inverse=True)
    count = np.zeros(len(unique),np.int)
    np.add.at(count, inverse, 1)
    return count[inverse]

def extrude(points, triangles, outer, inner):
    """
    radially extrude a surface mesh into a solid
    given a bounded surface, return a closed solid
    create extra options; rather than plain radial extrude,
    we can also do swepth sphere extrude; better for casting
    """
    #construct edge information
    tri_edges_all = triangle_edges(triangles)       #nx2 pairs of indices
    # tri_edges_id  = label_edges(tri_edges_all)

    #count the number of times each edge occurs. cant we use searchsorted here? surely this isnt most elegant way?
    #use group.multiplicity here
##    edges, inverse  = np.unique(tri_edges_id, return_inverse = True)
##    _, idx          = np.unique(np.sort(inverse), return_index=True)
##    incidence_count = np.diff(np.append(idx,len(inverse)))[inverse]
##
##    #find boundary indices, in original oriented ordering
##    boundary = incidence_count == 1
    boundary = npi.multiplicity(tri_edges_all) == 1
    eb = tri_edges_all[boundary]
    if len(eb) == 0: raise Exception('Surface to be extruded is closed, thus does not have a boundary')

    #construct closed solid
    boundary_tris = np.vstack((
        np.concatenate((eb[:,::-1], eb[:,0:1]+len(points)),axis=1),
        np.concatenate((eb[:,::+1]+len(points), eb[:,1:2]),axis=1)))

    copy_tris = triangles[:,::-1]+len(points)

    solid_points = np.vstack((points*np.atleast_1d([outer])[:,None], points*np.atleast_1d(inner)[:,None]))
    solid_tris   = np.vstack((triangles, copy_tris, boundary_tris))

    return solid_points, solid_tris


def extrude(outer, inner, triangles):
    """
    radially extrude a surface mesh into a solid
    given a bounded surface, return a closed solid
    create extra options; rather than plain radial extrude,
    we can also do swepth sphere extrude; better for casting
    """
    points = len(outer)

    #construct edge information
    tri_edges_all = triangle_edges(triangles)       #nx2 pairs of indices

    boundary = npi.multiplicity(tri_edges_all) == 1
    eb = tri_edges_all[boundary]
    if len(eb) == 0: raise Exception('Surface to be extruded is closed, thus does not have a boundary')

    #construct closed solid
    boundary_tris = np.vstack((
        np.concatenate((eb[:,::-1], eb[:,0:1]+points),axis=1),
        np.concatenate((eb[:,::+1]+points, eb[:,1:2]),axis=1)))

    copy_tris = triangles[:,::-1]+points

    solid_points = np.vstack((outer, inner))
    solid_tris   = np.vstack((triangles, copy_tris, boundary_tris))

    return solid_points, solid_tris


def swept_extrude(outer, inner, thickness):
    """
    outer is a copy of inner, possibly with added detail, but with identical boundary
    we seek to create a castable object with a constant thickness 'thickness'
    to that end, we need to match the boundary points to make a closed extrusion
    extrusion is done iteratively
    we init by radially shinking the inner mesh by thickness
    """
    def radial_displace(p, d):
        """move points radially inwards"""
        l = np.linalg.norm(p, 2, axis=1)
        return p * ((l-d) / l)[..., None]

    def angle(A, B):
        return util.dot( util.normalize(A), util.normalize(B))

    #find boundary edges
    op, ot = outer
    ip, it = inner
    #find mapping between boundary points and boundary edges

    #move inner points radially inwards as speedup step
    ip = radial_displace(ip, thickness)
    #incremental updates
    otree = KDTree(op)
    for i in range(3):
        #compute radial updates
        itree = KDTree(ip)
        pairs = itree.sparse_distance_matrix(otree, thickness, 2)
        idx  = np.array(pairs.keys())
        dist = np.array(pairs.values())

        delta = ip[idx[:,0]] - op[idx[:,1]]

        dec = (thickness-dist) / angle(ip[idx[:,0]], delta)
        uidx, maxdec = group_by(idx[:,0]).max(dec)
        ip[uidx] = radial_displace(ip[uidx], maxdec)

        #perform light smoothing
        #do we need original space for this?

    #return inner surface swepth by thickness
    return ip, it



if __name__=='__main__':
    #test triangulation

    #random points on the sphere
    points = util.normalize(np.random.randn(10000,3))

    #build curve. add sharp convex corners, as well as additional cuts
    N = 267#9
    radius = np.cos( np.linspace(0,np.pi*2*12,N, False)) +1.1
    curve = np.array([(np.cos(a)*r,np.sin(a)*r,1) for a,r in izip( np.linspace(0,np.pi*2,N, endpoint=False), radius)])
    curve = np.append(curve, [[1,0,-4],[-1,0,-4]], axis=0)      #add bottom slit
    curve = util.normalize(curve)
#    print curve
    curve_idx = np.arange(N)
    curve_idx = np.ascontiguousarray(np.vstack((curve_idx, np.roll(curve_idx, 1))).T)
#    curve_idx = np.append(curve_idx, [[45,N-45]], axis=0)
#    curve_idx = np.append(curve_idx, [[N/24+45,N/24-45]], axis=0)
    curve_idx = np.append(curve_idx, [[N,N+1]], axis=0)




    #do triangulation
    allpoints, triangles, curve, curve_idx = triangulate(points, curve, curve_idx)
    #add partitioning of points here too?
    partitions = partition(triangles, curve_idx)




    if False:
        from mayavi import mlab
        for p,c in izip( partitions, [(1,0,0), (0,1,0),(0,0,1),(0,1,1),(1,0,1),(1,1,0)]):
            x,y,z= allpoints.T
            mlab.triangular_mesh(x,y,z, p, color = c,       representation='surface')
            mlab.triangular_mesh(x,y,z, p, color = (0,0,0), representation='wireframe')
        mlab.show()


#    filename = r'C:\Users\Eelco\Dropbox\Escheresque\examples\part{0}.stl'
    filename = r'C:\Users\Eelco Hoogendoorn\Dropbox\Escheresque\examples\part{0}.stl'

    if True:
        radius = np.linspace(0.99, 1.01, len(partitions))
        from mayavi import mlab
        for i,(p,c) in enumerate(izip( partitions, [(1,0,0), (0,1,0),(0,0,1),(0,1,1),(1,0,1),(1,1,0)])):
            solid_points, solid_triangles = extrude(allpoints, p, radius[i], 0.8)

            x,y,z= solid_points.T

            mlab.triangular_mesh(x,y,z, solid_triangles, color = c,       representation='surface')
            mlab.triangular_mesh(x,y,z, solid_triangles, color = (0,0,0), representation='wireframe')

            import stl
            stl.save_STL(filename.format(i), util.grab(solid_triangles,solid_points))
        mlab.show()
    quit()
