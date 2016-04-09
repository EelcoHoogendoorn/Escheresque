
"""
datamodel class

contains a group, and its associated data

this contains all information that might need to be stored to disk.

all points and edges are stored here, too

store entire multicomplex here too?
at least store level, and heightmap
generalize to editing on a composition of layers?
"""

import numpy as np

from escheresque.subdivision import Curve
from escheresque.util import normalize
from escheresque import multicomplex
from escheresque import brushes
from escheresque import poisson


class DataModel(object):

    #heightmapping
    radius = None       #[index, vertices] array with per vertex radius values.

    def __init__(self, group):
        print 'creating dm'
        self.group = group      #immutable group info


        self.points = []
        self.edges = []

        self.generate(6)

    @property
    def complex(self):
        return self.hierarchy[-1]

    def generate(self, level):
        self.level = level
        self.hierarchy = multicomplex.generate(self.group, self.level)

        self.forcefield = np.zeros (self.complex.shape)
        self.heightfield = np.ones (self.complex.shape)

        self.update()



    def regenerate(self, level):
        oldlevel = self.level
        old = self.hierarchy
        forcefield = self.complex.D2P0 * self.forcefield

        self.level = level
        self.hierarchy = multicomplex.generate(self.group, self.level)

##        self.forcefield  = np.zeros(self.complex.shape)

        #scale up forcefield
        if self.level > oldlevel:
            for l in range(oldlevel, self.level):
                forcefield = self.hierarchy[l].interpolate_d2(forcefield)
        if self.level < oldlevel:
            for l in range(oldlevel, self.level, -1):
                forcefield = old[l-1].restrict_d2(forcefield)

        self.forcefield = self.complex.P0D2 * forcefield
        self.heightfield = np.ones (self.complex.shape)

        self.update()


    def primal(self):
        return normalize(self.group.primal)
    def mid(self):
        return normalize(self.group.mid)
    def dual(self):
        return normalize(self.group.dual)

    def save(self, filename):
        import cPickle as pickle
        pickle.dump(self, file(filename, 'wb'), -1)
    @staticmethod
    def load(filename):
        import cPickle as pickle
        return pickle.load(file(filename, 'rb'))

    def add_point(self, pos):
        """add a point to datamodel, given a world space coordinate"""
##        domain, bary = self.group.find_support(pos)
        domain, bary = [np.squeeze(r) for r in self.group.find_support(pos)]

        point = Point(self, bary, domain)
        self.points.append(point)
        return point

    def add_edge(self, l, r):
        lp,li = l
        rp,ri = r

        point = lp.point,rp.point
        index = li,ri

        edge = Edge(point, index)
        self.edges.append(edge)
        return edge


    def partition(self):
        """
        take a datamodel, and subdivide it into surface meshes, according to a list of boundary curves
        take standard boundary prop for now
        """
##        level = 7
##        from . import multicomplex
##        hierarchy = multicomplex.generate(self.group, level)
        hierarchy = self.hierarchy
        complex = hierarchy[-1]

        from escheresque import computational_geometry

        #concat all curves
        curve_p   = [transform for e in self.edges if e.boundary for mirrors in e.instantiate() for transform in mirrors]
        offset    = np.cumsum([len(p) for p in curve_p])
        curve_idx = [np.arange(o-len(p), o) for p,o in zip( curve_p, offset)]
        curve_idx = [np.vstack((i[:-1], i[1:])).T for i in curve_idx]
        curve_p   = np.vstack(curve_p)
        curve_idx = np.vstack(curve_idx)

        curve_p, curve_idx = computational_geometry.merge_geometry(curve_p, curve_idx)

        #instantiate a geometry
        geometry = complex.geometry
        group = self.group
        points        = np.empty((group.index, group.order, geometry.topology.P0, 3), np.float)
        PP = geometry.decomposed        #hide this sort of thing behind instantion function
        for i,B in enumerate(group.basis):
            for t, b in enumerate(B.reshape(-1,3,3)):
                b = normalize(b.T).T                      #now every row is a normalized vertex
                P = np.dot(b, PP.T).T                      #go from decomposed coords to local coordinate system
                points[i,t] = P

        #make single unique point list
        points = computational_geometry.merge_geometry(points.reshape(-1, 3))

        #triangulate
        allpoints, triangles, curve_p, curve_idx = computational_geometry.triangulate(points, curve_p, curve_idx)
        #get partitions
        partitions = computational_geometry.partition(allpoints, triangles, curve_idx)

        return partitions


    def sample(self, points):
        return brushes.Mapping(self.hierarchy, points).sample(self.heightfield)

    def update(self):
        self.heightfield = poisson.solve_poisson(self)



#valid constraint values (enumerated for UI convenience)
Constraints  = [None,
                (1,0,0),
                (0,1,0),
                (0,0,1),
                (0,1,1),
                (1,0,1),
                (1,1,0)]


class Point(object):
    """
    point specified as bary weighting over fundamental domain
    first and last half of points may have different orientation
    """
    def __init__(self, datamodel, bary, domain, constraint = None):
        self.datamodel = datamodel
        self.group = datamodel.group
        self.domain = domain                #domain subtile; tile, orientation, rotation
        self.constraint = constraint        #tuple of bool pmd or none
        self.bary = self.normalize(np.array(bary, np.float))
        self.radius = 1

    #access domain properties
    @property
    def index(self):
        return self.domain[0]
    @property
    def transform(self):
        return self.domain[1:]

    #property to enforece casting to tuple
    def set_domain(self, d):
        self._domain = tuple(d)
    def get_domain(self):
        return self._domain
    domain = property(get_domain, set_domain)

    #property to auto-norm bary coords?
    def normalize(self, bary):
        return bary/bary.sum()

    def instantiate(self):
        """return position array"""
        return self.instantiate_offset(self.transform)
    def instantiate_offset(self, transform):
        """return position array. instantiate with index point being the first"""
        basis = self.group.permuted_basis(self.index, transform)
        return normalize(np.dot(basis, self.bary))


    def match(self, selected_transform, position):
        """
        match the point representation to the given world position
        """
        #remap position, as if we are manipulating the first point
        transform = self.group.transforms[selected_transform]
        position = np.dot(transform.T, position)        #this is the position the first untransformed point should conform to

        #try recalc baries assuming we are still in same domain tile
        self.bary = np.dot( self.group.inverse[self.domain], position)

        #find new domain for zero point if out of current one
        if np.any(self.bary<0):
            self.domain, self.bary = [np.squeeze(r) for r in self.group.find_support(position)]

        #apply constraint
        position = self.constrain(position)
        #return retransformed point
        return np.dot(transform, position)


    def constrain(self, position):
        """
        constrain bary coords, given world coords of zero transform point
        return constrained point in world coords
        """
#        print self.domain
        B = self.group.basis[self.domain]
        B = B * self.constraints[np.newaxis,:]      #zero out deactived basis points
        bary = np.linalg.lstsq(B, position)[0]
        self.bary = self.normalize(bary)
        return normalize( np.dot(B, self.bary))


    domain = property(get_domain, set_domain)

    def get_constraint(self):
        if self.constraint is None:
            return np.ones(3)
        else:
            return np.array(self.constraint)

    def set_constraint(self, c):
        """set constraint, and adjust baries accordingly"""
        self.constraint = c
        return self.constrain(self.instantiate()[0,0])






class Edge(object):
    """
    connects any two points
    """

    #tuple of points, and the instance index to which the edge is rooted
    points     = (None, None)
    transforms = ((0,0), (0,0))  #transform relative to root domain, which define the root edge

    curve      = None           #subdivision curve; this should really be here. controlpoints are part of datamodel. cmputation and caching should be handled by the editor
    color      = (0,0,0)        #for giggles
    boundary   = True
    driving    = True           #driving edges enforce their radius on the relief
    width      = 0.01           #smoothing width to apply to get stencil

    def __init__(self, points, transforms = ((0,0), (0,0))):
        self.points     = points
        self.transforms = transforms
        self.curve      = Curve(self)

    def basis(self):
        """return start, end and orientation vectors for each edge"""
        L, R = [p.instantiate_offset(self.group.permute_forward(p.transform, t))
                    for p,t in zip(self.points, self.transforms)]

        o = np.array([+1,-1])[:len(L)]
        S = np.cross(L, R) * o[:, None, None]
        return L, R, S

    def instantiate(self):
        """"returns actual curve data by subdivision?"""
        #this is a hack... find better way to propagate point radii
        self.curve.radius[0] = self.points[1].radius
        self.curve.radius[-1] = self.points[0].radius

        return self.curve.curve()
    def instantiate_cp(self):
        """"returns actual curve data by subdivision?"""
        #this is a hack...
        self.curve.radius[0] = self.points[1].radius
        self.curve.radius[-1] = self.points[0].radius

        return self.curve.controlpoints()

    @property
    def group(self):
        return self.points[0].group
    @property
    def datamodel(self):
        return self.points[0].datamodel


class SubdivisionCurve(Edge):
    """
    subdivision
    extend edge with control points
    control points are permanent
    subdivided points are not
    """
    control = None

    @property
    def points(self):
        return len(self.control)