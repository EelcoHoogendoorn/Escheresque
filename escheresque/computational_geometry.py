
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
import scipy.sparse
import scipy.spatial
import openmesh

from escheresque import util

class PolyData(object):

    def merge(self, other):
        vertices = np.concatenate([self.vertices, other.vertices], axis=0)
        faces = np.concatenate([self.faces, other.faces + len(self.vertices)], axis=0)
        _, _idx, _inv = npi.unique(vertices, return_index=True, return_inverse=True)
        return type(self)(vertices[_idx], _inv[faces])

    def squeeze(self):
        """compact geometry description, removing unused vertices, and adjusting faces accordingly"""
        active, inv = np.unique(self.faces, return_inverse=True)
        return type(self)(self.vertices[active], inv.reshape(self.faces.shape))

    @staticmethod
    def order_edges(edges):
        return np.where((edges[:, 0] < edges[:, 1])[:, None], edges[:, ::+1], edges[:, ::-1])


class Curve(PolyData):
    def __init__(self, vertices, faces=None):
        if faces is None:
            faces = np.arange(len(vertices))[:, None]
            faces = np.concatenate([faces[:-1], faces[1:]], axis=1)

        vertices = np.asarray(vertices)
        faces = np.asarray(faces)

        assert vertices.shape[1] == 3
        assert faces.shape[1] == 2

        self.vertices = vertices
        self.faces = faces

    def self_intersect(self):
        """
        test curve of arc-segments for intersection
        raises exception in case of intersection
        alternatively, we might resolve intersections by point insertion
        but this is unlikely to have any practical utility, and more likely to be annoying
        """
        vertices = self.vertices
        faces = self.faces
        tree   = KDTree(vertices)
        # curve points per edge, [n, 2, 3]
        cp     = util.gather(faces, vertices)
        # normal rotating end unto start
        normal = util.normalize(np.cross(cp[:,0], cp[:,1]))
        # midpoints of edges; [n, 3]
        mid    = util.normalize(cp.sum(axis=1))
        # vector from end to start, [n, 3]
        diff   = np.diff(cp, axis=1)[:,0,:]
        # radius of sphere needed to contain edge, [n]
        radius = np.linalg.norm(diff, axis=1) / 2 * 1.01

        # FIXME: this can be vectorized by adapting pinv
        projector = [np.linalg.pinv(q) for q in np.swapaxes(cp, 1, 2)]

        # incident[vertex_index] gives a list of all indicent edge indices
        incident = npi.group_by(faces.flatten(), np.arange(faces.size))

        def intersect(i,j):
            """test if spherical line segments intersect. bretty elegant"""
            intersection = np.cross(normal[i], normal[j])                               #intersection direction of two great circles; sign may go either way though!
            return all(np.prod(np.dot(projector[e], intersection)) > 0 for e in (i,j))  #this direction must lie within the cone spanned by both sets of endpoints
        for ei,(p,r,cidx) in enumerate(izip(mid, radius, faces)):
            V = [v for v in tree.query_ball_point(p, r) if v not in cidx]
            edges = np.unique([ej for v in V for ej in incident[v]])
            for ej in edges:
                if len(np.intersect1d(faces[ei], faces[ej])) == 0:      #does not count if edges touch
                    if intersect(ei, ej):
                        raise Exception('The boundary curves intersect. Check your geometry and try again')

    def refine(self, points):
        """
        refine the contour such as to maintain it as a constrained boundary under triangulation using a convex hull
        this is really the crux of the method pursued in this module
        we need to 'shield off' any points that lie so close to the edge such as to threaten our constrained boundary
        by adding a split at the projection of the point on the line, for all vertices within the swept circle of the edge,
        we may guarantee that a subsequent convex hull of the sphere respects our original boundary
        """
        allpoints = np.vstack((self.vertices, points))
        tree = KDTree(allpoints)

        cp     = util.gather(self.faces, self.vertices)
        normal = util.normalize(np.cross(cp[:,0], cp[:,1]))
        mid    = util.normalize(cp.sum(axis=1))
        diff   = np.diff(cp, axis=1)[:,0,:]
        radius = np.linalg.norm(diff, axis=1) / 2

        def insertion_point(e, c):
            """calculate insertion point"""
            coeff = np.dot( np.linalg.pinv(cp[e].T), allpoints[c])
            coeff = coeff / coeff.sum()
            return coeff[0], np.dot(cp[e].T, coeff)

        #build new curves
        _curve_p = [c for c in self.vertices]
        _curve_idx = []
        for e,(m,r,cidx) in enumerate(izip( mid, radius, self.faces)):
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

        return Curve(_curve_p, _curve_idx)

    def trim(self, points, radius):
        """
        remove points too close to the cut curve. they dont add anything, and only lead to awkward faces
        """
        #some precomputations
        tree   = KDTree(points)
        cp     = self.vertices[self.faces]
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
        for p in self.vertices:
            coarse = tree.query_ball_point(p, radius)
            index[coarse] = 0

        return points[index]


class Mesh(PolyData):
    def __init__(self, vertices, faces):
        vertices = np.asarray(vertices)
        faces = np.asarray(faces)

        assert vertices.shape[1] == 3
        assert faces.shape[1] == 3

        self.vertices = vertices
        self.faces = faces

    def edges(self):
        """construct 3 edges for each triangle"""
        edges = self.faces[:, [[1, 2], [2, 0], [0, 1]]]
        return edges.reshape(-1, 2)

    def ordered_edges(self):
        return self.order_edges(self.edges())

    def compute_face_incidence(self):
        # get edges with consistent ordering
        ordered_edges = self.ordered_edges()

        # construct full incidence matrix
        unique_edges, edge_indices = npi.unique(ordered_edges, return_inverse=True)
        face_indices = np.arange(self.faces.size) // 3
        incidence = scipy.sparse.csr_matrix((np.ones_like(edge_indices), (edge_indices, face_indices)))
        return incidence, unique_edges

    def is_orientated(self):
        return npi.all_unique(self.edges())

    def boundary(self):
        edges = self.edges()
        return edges[npi.multiplicity(self.order_edges(edges)) == 1]

    def partition(self, curve):
        """partition the connected component, seperated by the cutting curves

        curve : ndarray, [n_edges, 2], int

        returned is a list of (point, faces) tuples, denoting the partitions
        """
        # get edges with consistent ordering
        ordered_curve = self.order_edges(curve.faces)

        incidence, unique_edges = self.compute_face_incidence()

        # filter out the curve edges
        incidence = incidence[~npi.in_(unique_edges, ordered_curve)]
        adjecency = incidence.T * incidence
        # find and split connected components
        n_components, labels = scipy.sparse.csgraph.connected_components(adjecency)
        partitions = [self.faces[labels == l] for l in range(n_components)]
        partitions = [Mesh(self.vertices, triangles).squeeze() for triangles in partitions]
        return sorted(partitions, key=lambda m: len(m.vertices))

    def is_orientated(self):
        return npi.all_unique(self.edges())

    def swept_extrude(self, thickness):
        """
        outer is a copy of inner, possibly with added detail, but with identical boundary
        we seek to create a castable object with a constant thickness 'thickness'
        to that end, we need to match the boundary points to make a closed extrusion
        extrusion is done iteratively
        we init by radially shinking the inner mesh by thickness
        """
        assert thickness > 0
        outer = self.vertices
        tree = KDTree(outer)

        outer_radius = np.linalg.norm(outer, axis=1)
        inner = outer

        #incremental updates
        while True:
            # find nearest point for each inner point
            dist, idx = tree.query(inner, k=1)

            inner_radius = np.linalg.norm(inner, axis=1)
            radial_dist = inner_radius - outer_radius[idx]
            ortho_dist2 = dist**2 - radial_dist**2
            new_radius = outer_radius[idx] - np.sqrt(1 - ortho_dist2 / thickness ** 2) * thickness

            if np.allclose(inner_radius, new_radius):
                break
            inner = inner / (inner_radius / new_radius)[:, None]

        #return inner surface swepth by thickness
        return self.extrude(inner)

    def extrude(self, displaced):
        """
        radially extrude a surface mesh into a solid
        given a bounded surface, return a closed solid
        create extra options; rather than plain radial extrude,
        we can also do swepth sphere extrude; better for casting
        """
        points = len(self.vertices)

        #construct edge information
        edges = self.edges()

        boundary = npi.multiplicity(self.order_edges(edges)) == 1
        eb = edges[boundary]
        if len(eb) == 0: raise Exception('Surface to be extruded is closed, thus does not have a boundary')

        #construct closed solid
        boundary_tris = np.concatenate((
            np.concatenate((eb[:,::-1], eb[:,0:1]+points),axis=1),
            np.concatenate((eb[:,::+1]+points, eb[:,1:2]),axis=1)
        ))

        copy_tris = self.faces[:, ::-1] + points

        solid_points = np.concatenate((self.vertices, displaced))
        solid_tris   = np.concatenate((self.faces, copy_tris, boundary_tris))

        return Mesh(solid_points, solid_tris)

    def to_openmesh(self):
        mesh = openmesh.TriMesh()
        vhs = [mesh.add_vertex(openmesh.TriMesh.Point(*vertex)) for vertex in self.vertices.astype(np.float64)]

        for triangle in self.faces:
            mesh.add_face([vhs[v] for v in triangle])
        return mesh

    @staticmethod
    def from_openmesh(mesh):
        vertices = [list(mesh.point(vh)) for vh in mesh.vertices()]
        faces = [[v.idx() for v in mesh.fv(fh)] for fh in mesh.faces()]
        return Mesh(vertices, faces)

    def decimate(self, n_faces):
        mesh = self.to_openmesh()

        decimater = openmesh.TriMeshDecimater(mesh)
        modquadrichandle = openmesh.TriMeshModQuadricHandle()
        decimater.add(modquadrichandle)

        decimater.initialize()
        # print(decimater.decimate(1000))
        decimater.decimate_to_faces(n_faces)
        mesh.garbage_collection()

        del decimater
        import gc
        gc.collect()

        return Mesh.from_openmesh(mesh)

    def plot(self):
        from mayavi import mlab
        q = self.vertices.mean(axis=0)
        x, y, z = (self.vertices + q / 4).T
        mlab.triangular_mesh(x, y, z, self.faces)
        # mlab.triangular_mesh(x, y, z, t, color=(0, 0, 0), representation='wireframe')
        mlab.show()




def triangulate(points, curve):
    """
    return a triangulation of the pointset points,
    while being constrained by the boundary dicated by curve
    """
    #test curve for self-intersection
    print 'testing curve for self-intersection'
    curve.self_intersect()

    #trim the pointset, to eliminate points co-linear with the cutting curve
    print 'trimming dataset'
    diff   = np.diff(curve.vertices[curve.faces], axis=1)[:,0,:]
    length = np.linalg.norm(diff, axis=1)
    points = curve.trim(points, length.mean()/4)

    #refine curve iteratively. new points may both obsolete or require novel insertions themselves
    #so only do the most pressing ones first, then iterate to convergence
    while True:
        newcurve = curve.refine(points)
        if len(newcurve.vertices)==len(curve.vertices):
            break
        print 'curve refined'
        curve = newcurve


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
    allpoints = np.concatenate((curve.vertices, points))    #include origin; facilitates clipping
    hull = scipy.spatial.ConvexHull(util.normalize(allpoints))
    triangles = hull.simplices

    #order faces coming from the convex hull
    print 'ordering faces'
    FP        = util.gather(triangles, allpoints)
    mid       = FP.sum(axis=1)
    normal    = util.normals(FP)
    sign      = util.dot(normal, mid) > 0
    triangles = np.where(sign[:,None], triangles[:,::+1], triangles[:,::-1])

    mesh = Mesh(allpoints, triangles)
    assert mesh.is_orientated()

    return mesh, curve



if __name__=='__main__':

    if True:
        from escheresque.datamodel import DataModel
        from mayavi import mlab
        from escheresque import stl, brushes
        import os

        path = r'C:\Users\Eelco\Dropbox\Git\Escheresque\data'
        filename = 'turtles.sch'

        datamodel = DataModel.load(os.path.join(path, filename))
        # datamodel.generate(5)
        partitions = datamodel.partition()

        filename = r'C:\Users\Eelco\Dropbox\Git\Escheresque\data\part{0}.stl'

        for i, mesh in enumerate(partitions):
            mesh.vertices *= datamodel.sample(mesh.vertices)[:, None]
            thickness = 0.07
            mesh = mesh.swept_extrude(thickness)
            assert mesh.is_orientated()
            stl.save_STL(filename.format(i), mesh)
            break

        mesh.plot()
        quit()





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
            stl.save_STL(filename.format(i), util.gather(solid_triangles, solid_points))
        mlab.show()
    quit()
