
"""
main entry point

for 3d printing, we need both relief and segmentation
these editors should be integrated better; that should be the prime focus now
edge editor needs to be improved; direct click, not-shitty picking
should we implement custom picking based on KDTrees?

brushes need major improvemnt too

some bug appear pickling-related. want to avoid class-properties

make posisson editor main type? not sure; interior boundaries are nontrivial
use immersed boundary conditions? or implement all boundaries with penalties?
single basic boundary is no different than extended ones

perhaps do heightfield editing per-parition?
certainly preferable on rendering performance grounds
could be implemented quite efficiently; just edit the underlying sphere, and
remap using fixed transform. still; premature optimization...

use pressure-bc poisson solver? create guide-curves with subdiv editor
surface is moulded such as to respect guide curves
solve for a conforming geometry in realtime
cache raserization of each curve
also cache the blurred version of the line
per-controlpoint width isnt really going to happen; blurring is a global op
need to reblur every line at every iteration, no?
yeah; only way to update per-point pressure is to re-raster (cached) and reblur

mulitigrid should work well on these type of problems. force vec is easily coarsened


"""

backend = 'qt4'
if backend == 'qt4':
    import os
    os.environ['ETS_TOOLKIT'] = 'qt4'


import numpy as np

from .datamodel import DataModel




def run():
    """
    starting the main thread needs to be isolated from module level,
    to avoid import deadlock issues
    """
    if True:
        from .group.tetrahedral import ChiralTetrahedral as Group
        ##        from tetrahedral import Null as Group
        ##        from octahedral import Tetrahedral as Group
        ##        from octahedral import Null as Group
        from .group.octahedral import Pyritohedral as Group
        ##        from octahedral import ChiralOctahedral as Group
        ##        from octahedral import Tetrahedral as Group
        ##        from octahedral import Origin as Group
        ##        from dihedral import ChiralDihedral as Group
        ##        from icosahedral import ChiralIcosahedral as Group
        ##        from icosahedral import Icosahedral as Group
        dm = DataModel(Group())
    else:
        ##        dm = DataModel.load(r'C:\Users\Eelco\Dropbox\Escheresque\examples\fishes.sch')
        ##        dm = DataModel.load(r'C:\Users\Eelco\Dropbox\Escheresque\examples\angles_and_demons.sch')
        dm = DataModel.load(r'C:\Users\Eelco\Dropbox\Escheresque\code\escheresque\turtles.sch')
        ##        dm = DataModel.load(r'C:\Users\Eelco\Dropbox\Escheresque\code\v2\fishes.sch')


    if False:
        from .interface.edge_editor import EdgeEditor
        editor = EdgeEditor(dm)
    else:
        from .interface.height_editor import HeightEditor
        editor = HeightEditor(dm)
    editor.configure_traits()


