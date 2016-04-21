
"""
review overall workflow.
poisson editor is a great start for sculpting
but it would be nice if we had good post-editing features
blender fills these functions pretty nicely though
can we export vertex weight to blender, for fixing?


main entry point

editors should be integrated better; that should be the prime focus now
edge editor needs to be improved; direct click, not-shitty picking
should we implement custom picking based on KDTrees?

brushes need major improvemnt too

some bug appear pickling-related. want to avoid class-properties
what about cursor movement bug?


perhaps do heightfield editing per-parition?
certainly preferable on rendering performance grounds
could be implemented quite efficiently; just edit the underlying sphere, and
remap using fixed transform. still; premature optimization...





add feedback mechanism to general drawing class too?
select edge, then give it a target height; solve locally for the edge
or solve for constant differential along the line

create diffusion barriers in painting tools? messes with multigrid, but could be useful for limited scenarios


add ability to dynamically adjust subdivision level. maintain max level in datamodel

is popup related to bypassing mlabsource in cursor?


dual-based mapping object?
after last step, test for fundamental domain?
easy to find d2 we are in. but how to interpolate?
could calc fundamental domain of fine tri we are in
then interpolate one additional level down, to get d2 at mid of primal edge
only need d2 at d0 now, and we can linearly interpolate
can calc this as edge midpoint values bary-weighted by coarse_areas
will get thin-sliver tris though; is this a problem?
also, triangle-count will incrase sixfold
and vertex-normal compu will need rethinking
but only for sampling it is not too bad
still piecewise linear. do we get meaningfull improvement in any sense?

may help with poisson-mg. infact; high iteration without mg also shows oscilation
appears to be an issue of medium-convergence
enough to start fitting at the small scale, but not enough to do it right
get long range oscilation that increases with iteration on turtle... hmmm


add circulation preserving fluids?
compute velocity on d0. each d0 has three incident tangential fluxes constrained by incompressibility
which should give us two velocity components
we can then interpolate over fundamental domains
why does it not work on squares? we have 4 instead of 3 incident pieces
this can still 'almost' happen, leading to a sort of discontinuity in the flowfield
for each point, check first if not still in original group. only if not, recompute
http://www.geometry.caltech.edu/pubs/ETKSD07.pdf
of course do all this symmetrically too

general multigrid and soft bc's idea; can we do nested linear solver to get mg performance on complex geom?


shit; d2 interpolation requires boundary communication...
we may simply gather all from d2 on one side, and then boundify no?
yeah; but still something appears not quite right
why did harmonic calculation suddenly go wrong?


high-subdivision, supposedly linearly interolated d2 functions,
hint at a bug in interpolation code. surface gets oscilations...


write about relief editor. natural notion of curvature, and all
kinda like a soap membrane; except with a different notion of flatness
"""

backend = 'qt4'
if backend == 'qt4':
    import os
    os.environ['ETS_TOOLKIT'] = 'qt4'


from escheresque.datamodel import DataModel


def run():
    """
    starting the main thread needs to be isolated from module level,
    to avoid import deadlock issues
    """
    if False:
        from escheresque.group.tetrahedral import ChiralTetrahedral as Group
        ##        from tetrahedral import Null as Group
        ##        from octahedral import Tetrahedral as Group
        ##        from octahedral import Null as Group
        from escheresque.group.octahedral import Pyritohedral as Group
        ##        from octahedral import ChiralOctahedral as Group
        ##        from octahedral import Tetrahedral as Group
        ##        from octahedral import Origin as Group
        ##        from dihedral import ChiralDihedral as Group
        ##        from icosahedral import ChiralIcosahedral as Group
        ##        from icosahedral import Icosahedral as Group
        dm = DataModel(Group())
        # dm.generate(6)
    else:
        path = r'..\data'
        filename = 'turtles.sch'
##        filename = 'angles_and_demons.sch'

        dm = DataModel.load(os.path.join(path, filename))
        # dm.regenerate(5)








    if True:
##        from .interface.edge_editor import EdgeEditor
##        editor = EdgeEditor(dm)
        from escheresque.interface.poisson_editor import PoissonEditor
        editor = PoissonEditor(dm)
    else:
        from escheresque.interface.height_editor import HeightEditor
        editor = HeightEditor(dm)
##        from .interface.pressure_editor import PressureEditor
##        editor = PressureEditor(dm)
    editor.configure_traits()


if __name__ == '__main__':
    run()
