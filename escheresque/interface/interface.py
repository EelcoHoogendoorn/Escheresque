
"""
some common interface code

"""

#flag to disable threaded functions that get tripped up by mayavi
MAYAVI_THREADING = True

import numpy as np

from traits.api import HasTraits, Instance, Button, \
    on_trait_change, Range, Bool, Enum, List, Any, Int, Color, Str, Float
from traitsui.api import View, Item, HSplit, Group, EnumEditor, Handler, HGroup
from traitsui.api import EnumEditor, VGroup, VSplit, Label, spring


from mayavi import mlab
from mayavi.core import lut_manager
from mayavi.core.ui.api import MlabSceneModel, SceneEditor
from mayavi.core.ui.mayavi_scene import MayaviScene
from threading import Thread
from collections import defaultdict

from pyface.api import FileDialog, OK, confirm, YES, GUI

from tvtk.api import tvtk

from traits.api import push_exception_handler



push_exception_handler( handler = lambda o,t,ov,nv: None,
                        reraise_exceptions = True,
                        main = True,
                        locked = True )



class MouseHandler(object):
    """
    interactor style for escher application. how to do right clicking on mousepad?
    create toggle button for mousepad?

    rotation needs be improved; pick point on right click and match it to raycasted point
    left click; for spline manipulation, raycast to sphere
    for height editor, pick world coord on surface
    """
    def __init__(self, interactor):
        self.interactor = interactor



def rotate_actor(actor, R):
    """set rotation; could also extract orientation, and use this?"""
    m = actor._get_matrix()
    r = m.to_array()
    r[:3,:3] = R
    m.from_array(r)
    actor._set_user_matrix(m)


class RenderLock(object):
    """context lock to make changes to scene"""
    def __init__(self, scene):
        self.scene = scene
    def __enter__(self):
        self.scene.disable_render = True
        return self.scene
    def __exit__(self, type, value, traceback):
        self.scene.disable_render = False






class ThreadedAction(Thread):
    """
    put callback to UI update in a thread
    used for recalc while dragging points
    or generating new group
    """
    running = True
    def __init__(self, interface, **kwargs):
        Thread.__init__(self, **kwargs)
        self.interface = interface

    def run(self):
        step = 0
        while (self.running):
            self.interface.datamodel.Update()
            step += 1
            if step>=self.interface.steps:
                step=0
                GUI.invoke_later(self.interface.callback)
        GUI.invoke_later(self.interface.redraw_scene)

    def cancel(self):
        self.running = False





"""
not sure what is wrong here; lack of threadsafety might be the issue
"""


scale_flag = defaultdict(lambda: False)
def lock_scale(obj, name, old, new):
    print 'locking s'
    if scale_flag[obj]:
        scale_flag[obj] = False
        return
##    if np.all(new==1): return
    def icb():
        scale_flag[obj] = True
        obj.scale = np.array(old)
    GUI.invoke_later(icb)

position_flag = defaultdict(lambda: False)
def lock_position(obj, name, old, new):
    print 'locking p'
    if position_flag[obj]:
        position_flag[obj] = False
        return
##    if np.all(new==0): return
    def icb():
        position_flag[obj] = True
        """crash seems to originate here? lost as to why"""
        obj.position = np.array(old)
##        obj.position = 0,0,0
    GUI.invoke_later(icb)

orientation_flag = defaultdict(lambda: False)
def lock_orientation(obj, name, old, new):
    print 'locking o'
    if orientation_flag[obj]:
        orientation_flag[obj] = False
        return
    def icb():
        orientation_flag[obj] = True
        obj.orientation = np.array(old)
    GUI.invoke_later(icb)

def lock_actor(actor, orientation=True, scale=True, position=True):
    if orientation: actor.on_trait_change(lock_orientation, 'orientation')
    if scale:       actor.on_trait_change(lock_scale,       'scale')
    if position:    actor.on_trait_change(lock_position,    'position')



#some color defs
gray = (0.8,)*3
black = (0,)*3



class GroupInfo(HasTraits):
    """
    simple ui window to display a groups information
    """
    index = Int
    order = Int
    chiral = Bool
    polyhedron = Str
    subgroup = Str

    def __init__(self, group):
        self.group = group
        self.index = group.index