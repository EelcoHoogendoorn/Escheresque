"""
painting brush module
rename: the essence of this module is in mapping between intrinsic mesh space and embedding R3 space
painting is just one application thereof
"""

import itertools

import numpy as np

from escheresque import util
from escheresque import multigrid


def pick_primal_triangles(hierarchy, points):
    faces = np.zeros(len(points), np.int)           #all points start their search at the root triangle
    for complex in hierarchy[1:]:                   #recurse, skipping root level
        planes = complex.geometry.planes[faces]     #grab the three precomputed planes subdividing each triangle
        signs = np.einsum('ijk,ik->ij', planes, points) > 0  #where are we, with respect to the three planes?
        #only one sign should be positive .argmax is numerically stable; even in case of multiple positive values, we get a valid result
        index = np.argmax(signs, axis=1) + 1        #corner tri index is argmax plus one, since middle triangle is zero
        index *= signs.any(axis=1)                  #if we are infact in the middle of all planes; overwrite with zero
        faces = faces * 4 + index                   #convert local triangle index to global index according to recursive convention
    return faces


class Mapping(object):
    """
    object that maps between a set of points in R3
    and an intrinsic representation on a meshed sphere
    """

    def __init__(self, hierarchy, points):
        """pick a set of worldcoords on the sphere

        Parameters
        ----------
        hierarchy : list of MultiComplex
        points : ndarray, [n, 3], float

        Notes
        -----
        hierarchical structure of the triangles allows finding intersection triangle quickly
        """
        self.hierarchy = hierarchy
        self.points = util.normalize(np.atleast_2d(points))

        complex = hierarchy[-1]
        group = complex.group

        domains, baries = group.find_support(self.points)    #find out which fundamental domain tile each point is in
        local = np.dot(group.basis[0,0,0],baries.T).T   #map all points to the root domain for computations; all domains have identical tesselation anyway

        # get the face index for each point
        faces = pick_primal_triangles(hierarchy, local)

        #calc baries; simple linear bary computation is good enough for these purposes, no?
        baries = np.einsum('ijk,ik->ij', complex.geometry.inverted_triangle[faces], local)
        self.baries = baries / baries.sum(axis=1)[:, None]

        self.raveled_indices = np.ravel_multi_index(
            (complex.topology.FV[faces].ravel(), np.repeat(domains[0],3)),
            complex.shape)
        self.complex = complex

    def sample(self, field):
        """map p0 form on sphere to curve"""
        return (field.ravel()[self.raveled_indices].reshape(self.baries.shape) * self.baries).sum(axis=1)

    def inject(self, weights):
        """map weights on curve to d2 form on sphere"""
        brush = np.zeros(self.complex.shape, np.float)   #brush is d2; it s a conserved quantity
        util.scatter(
            self.raveled_indices,
            self.baries*weights[:,None],
            brush)
        return self.complex.boundify(brush)




class Mapping_d2(object):
    """
    create a mapping object to resample velocity components for flow simulation
    init only with datamodel; update with a new set of points, allowing for caching of properties

    these interpolations should happen in a maximally mesh-independent manner
    as a stress-case, a rectangular grid interpolation should not be biased towards the midline
    multigrid transfer should provide clues here

    is there anything wrong with advecting scalar vorticity directly via d2 mechanism?
    may not generalize to 3d, but whatever
    would need to be able to map velocity to p0 at least
    average tangential fluxes over boundary of d2?
    not sure this is possible; one cannot really map the normal to tangent fluxes on the advected mesh
    one cannot escape interpolating velocity it seems
    but velocity from d1 adjecnt to d0 produces massive jump on diagonal flip, unless it tapers to zero
    taperiong to zero hardly makes sense, however

    sum of all tangent fluxes around d2 makes more sense methinks
    this also needs custom boundification. same as normal really, no?

    d0-method has advantage of locally conserving vorticity by construction
    do we conserve vorticity? yeah, since it is always zero anyway in a closed domain
    but are we doing so in any local sense?
    not materially more or less so, i think
    he question is, what happens to a remapping of velo field, while keeping sampling points constant
    d0 vecloity leads to unchanging vorticity under these condition
    whereas d2 vecloity seems like it would result in diffusion
    bfecc might counteract that.

    given good integration, jump in velo would be less of an issue. would like to sample integrated quantity,
    but we cannoot construct a streamfunction

    BFECC should be easy to add as well.

    philosophical conclusion: a single mesh may not have all desired properties
    that is, we may need to rely on the randomness of the mesh for independence from its particulars,
    rather than obtaining proper operators for all mesh configurations
    """
    def __init__(self, datamodel, points):
        """precomputations which are identical for all mappings"""
        self.datamodel = datamodel
        self.hierarchy = self.datamodel.hierarchy
        self.complex   = self.hierarchy[-1]
        self.group     = self.complex.group

        #cache the index a point is in. if it remains unchanged, no need to update
        count = len(points)
        self.index = np.zeros((3, count), np.int8)

        #precompute subtriangles
        primal = self.complex.geometry.primal[ self.complex.topology.FV]
        mid    = util.normalize(np.roll(primal, +1, 1) + np.roll(primal, -1, 1))
        dual   = self.complex.geometry.dual

        basis = np.empty((self.complex.topology.D0, 3, 2, 3, 3))
        for i in range(3):
            basis[:,i,0,0,:] = primal[:,i  ,:]
            basis[:,i,1,0,:] = primal[:,i  ,:]
            basis[:,i,0,1,:] = mid   [:,i-2,:]
            basis[:,i,1,1,:] = mid   [:,i-1,:]
        basis[:,:,:,2,:]     = dual  [:,None, None,:]     #each subtri shares the dual vert

        self.subdomain = util.adjoint(basis)                #precompute all we can for subdomain compu
        self.subdomain[:,:,1,:,:] *= -1                     #flip sign
        self.subdomain = self.subdomain.reshape(-1,6,3,3)   #fold sign axis

        self.update(points)

    def update(self, points):
        """bind a new set of points
        presumably, repeatd update calls have coherency that can be exploited
        """

        #check which points need a domain update
        baries = np.einsum('pij,pj->pi', self.group.inverse[[i for i in self.index]], points)
        update = np.any(baries<0, -1)
        if np.any(update):
            index, baries[update] = self.group.find_support(points[update])
            for si, i in itertools.zip( self.index, index):
                si[update] = i

        #map all points to the root domain for computations; all domains have identical tesselation anyway
        local = np.dot(self.group.basis[0,0,0],baries.T).T

        #add caching here as well? many d2 points may simply remain within their triangle
        faces = pick_primal_triangles(self.hierarchy, local)
        print(faces)

        #now do fundamental domain picking. need to have edge midpoints, triangle corners, and dual
        #endresult should be a primal-mid-dual index for each point, for fast indexing
        #we can calc baries easily; does that help?
        #not really; need pos relative to dual
        #could also just preinvert all subtris; only 9x6 flops.
        #3x6 flops is optimal; but will need baries eventuall anyway
        #then just need to map back from positive
        sd = self.subdomain[faces]
        sub_baries = np.einsum('psbj,pj->psb', sd, local)
        sub_domain = np.all(sub_baries>=0, -1)
##        print np.nonzero(sub_domain)
        linear_index = sub_domain.argmax(axis=1)
##        dindex = np.unravel_index(linear_index, shape)
        v_idx = linear_index // 2
        e_idx = (linear_index - 3) // 2

        print(linear_index)
        print(v_idx)
        print(e_idx)

        self.v_idx = self.complex.topology.FV [faces,v_idx]
        self.e_idx = self.complex.topology.FEi[faces,e_idx]
        self.f_idx = faces
        print(self.v_idx)
        print(self.e_idx)
        print(self.f_idx)

        self.sub_baries = sub_baries[np.arange(len(linear_index)), linear_index]
        self.sub_baries /= self.sub_baries.sum(axis=-1)[:, None]
        print(self.sub_baries)

    def sample_d2(self, field):
        """sample a d2-form"""

    def sample_velocity(self, velocity_p1):
        """sample a velocity field"""
        #calc velocity representation on relevant subforms (primal-mid-dual vert)
        velocity_d1 = velocity_p1
        velocity_d2 = None

        #sum field over the picked baries





def generate_trace(trace):
    """
    convert picked points to well sampled trace of positions and weights
    """
    edges = np.array( zip(trace[1:], trace[:-1]))
    edges = util.normalize(edges)
    samples = np.linspace(0, 1, 11)
    samples = (samples[1:] + samples[:-1])/2
    weights = []
    positions = []
    for l,r in edges:
        L = np.linalg.norm(l-r)
        weights.extend([L]*len(samples))
        positions.extend( l[None, :] * samples[:, None] + r[None, :] * (1-samples[:, None]))
    weights = np.array(weights)
    positions = np.array(positions)#.reshape((-1,3))
    return positions, weights




def paint(hierarchy, trace, width):
    """
    take a point trace, and paint it onto a brush
    """
    complex = hierarchy[-1]
    #refine the trace
    positions, weights = generate_trace(trace)

    brush = Mapping(hierarchy, positions).inject(weights)

    brush = multigrid.diffuse(hierarchy, brush, width**2)

    return complex.P0D2 * brush





