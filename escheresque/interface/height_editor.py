
"""
heightfield interface code


add pulldown menus to select harmonic and RD options

create general inversion code on multicomplex
useful for implicit RD and inverse iteration

create suite of editing tools
on left click, start curve. at each mouse move, add control point at picked location
end when mouse released
then activate a set of controls to manipulate


camera has orientation_wxyz prop; if that is not a quat, what is?
"""

from escheresque.interface.selectgroup import SelectGroup
from escheresque.interface.interface import *
from escheresque.datamodel import DataModel
from escheresque import util
from escheresque.quaternion import Quaternion

from escheresque import multicomplex
from escheresque import reaction_diffusion


class EdgeHandler(Handler):
    """
    example of a handler for smoothing over saving and shit
    """

    def object_filename_changed(self, info):
        filename = info.object.filename
        if filename is "":
            filename = "<no file>"
        info.ui.title = "Editing: " + filename

    def close(self, info, isok):
        # Return True to indicate that it is OK to close the window.
        response = confirm(info.ui.control,
                        "Are you sure you want to close?")
        closing = response == YES
        try:
            info.object.ta.cancel()
        except:
            pass
        return closing

##
##        if not info.object.saved:
##            response = confirm(info.ui.control,
##                            "Value is not saved.  Are you sure you want to exit?")
##            return response == YES
##        else:
##            return True
##


class ThreadedRDAction(Thread):
    """
    put callback to UI update in a thread
    used for recalc while dragging points
    or generating new group
    """
    running = True
    def __init__(self, interface, datamodel, callback, **kwargs):
        Thread.__init__(self, **kwargs)
        self.interface = interface
        self.datamodel = datamodel
        self.callback = callback

        from ..reaction_diffusion import ReactionDiffusion
        self.rd = ReactionDiffusion(datamodel.complex)
        try:
            self.rd.coeff = self.rd.params[self.interface.reaction_selected]
        except:
            pass


    def run(self):
        step = 0
        while (self.running and step < 100):
##            print np.ones(1)
##            from scipy.sparse import csr_matrix
##            csr_matrix(np.eye(3))
            self.rd.simulate(30)
            self.datamodel.heightfield = self.rd.state[1].reshape(self.datamodel.complex.shape)

            GUI.invoke_later(self.callback)
            step += 1

##        GUI.invoke_later(self.interface.domain.redraw())

    def cancel(self):
        self.running = False








class Domain(object):
    """
    plot element for all domains

    needs to be linked to datamodel, no?

    yeah; heightfield needs to be in datamodel
    """

    def __init__(self, parent):
        super(Domain, self).__init__()
        self.parent  = parent

    @property
    def scene(self):
        return self.parent.scene
    @property
    def group(self):
        return self.parent.group
    @property
    def datamodel(self):
        return self.parent.datamodel
    @property
    def complex(self):
        return self.parent.complex

    def draw(self):
        """
        setup plots
        """

        def add_base(index, mirror, B):
            """
            build pipeline, for given basis
            """
            B = util.normalize(B.T).T

            PP = np.dot(B, self.complex.geometry.decomposed.T).T       #primal coords transformed into current frame


            x,y,z = PP.T
            FV = self.complex.topology.FV[:,::np.sign(np.linalg.det(B))]        #reverse vertex order depending on orientation
            source  = self.scene.mlab.pipeline.triangular_mesh_source(x,y,z, FV)

            #put these guys under a ui list
            l=lut_manager.tvtk.LookupTable()
            lut_manager.set_lut(l, lut_manager.pylab_luts['jet'])
##            lut_manager.set_lut(l, lut_manager.pylab_luts.values()[0])

            #add polydatamapper, to control color mapping and interpolation
            mapper = tvtk.PolyDataMapper(lookup_table=l)

            from tvtk.common import configure_input
            configure_input(mapper, source.outputs[0])

            ##            mapper = tvtk.PolyDataMapper(input=source.outputs[0])
            mapper.interpolate_scalars_before_mapping = True
            mapper.immediate_mode_rendering = False

            return mirror, source, mapper

        #add all root pipelines; one for all indices and mirrors. only pure rotations are stenciled
        self.root = np.array([[add_base(i, m, M[0]) for m,M in enumerate(T)] for i, T in enumerate( self.group.basis)])


        def add_instance(mirror, root, R):
            """add a rotated actor for each rotation, using a set of datasources"""
            mirror, source, mapper = root

            a = tvtk.Actor(mapper=mapper)
            self.scene.add_actor(a)

            #set rotation matrix
            rotate_actor(a, R)

            a.dragable = 0
            a.pickable = True
            a.property.backface_culling = True
            a.property.representation = 'wireframe' if self.parent.wireframe else 'surface'

            return a

        #add all rotated instances
        self.instances = []
        for T in self.root:
            for m,(root, rotations) in enumerate( zip(T, self.group.rotations)):
                for R in rotations:
                    self.instances.append( add_instance(m, root, R) )

        #always redraw
        self.redraw()


    def redraw(self):
        """
        set coords and scalar data of roots
        """
        hf = self.datamodel.heightfield
        hfl, hfh = hf.min(), hf.max()
        if hfh==hfl:
            hf[:] = 1
        else:
            hf = (hf-hfl) / (hfh-hfl)       #norm to 0-1 range

        hmin, hmax = 0.95, 1.05
        radius = hf*(hmax-hmin)+hmin

        #precompute normal information
        if self.parent.mapping_height:
            if self.parent.vertex_normal:
                normals = self.complex.normals(radius)
        else:
            N = util.normalize(self.complex.geometry.primal)
            normals = np.array([np.dot( N, T) for T in self.group.transforms.reshape(-1,3,3) ])

        if not self.parent.mapping_color:
            hf = np.ones_like(hf)

        for index,T in enumerate( self.root):
            for mirror,M in enumerate(T):
                _, source, mapper = M

##                mapper.lookup_table = None
                #set positions
                B = self.group.basis[index,mirror,0]
                B = util.normalize(B.T).T
                PP = np.dot(B, self.complex.geometry.decomposed.T).T       #primal coords transformed into current frame
                if self.parent.mapping_height: PP *= radius[:, index][:, None]
                source.mlab_source.set(points=PP)

                #set colors
                source.mlab_source.set(scalars=hf[:,index])

                #set normals
                if self.parent.vertex_normal:
                    M = self.group.transforms[mirror,0]
                    source.data.point_data.normals = np.dot(M, normals[index,:,:].T).T
                    source.data.cell_data.normals = None
                else:
                    source.data.point_data.normals = None
                    source.data.cell_data.normals = None

                source.mlab_source.update()


        for i in self.instances:
            i.property.representation = 'wireframe' if self.parent.wireframe else 'surface'





    def remove(self):
        for i in self.instances:
            self.scene.remove_actors(i)
##        for T in self.root:
##            for m in T:
##                mirror, source, mapper = m
##                mapper.remove()
##                source.remove()




class HeightEditor(HasTraits):
    """
    editor for heightfield
    """

    scene               = Instance(MlabSceneModel, args=())

    primal_visible      = Bool('primal')
    mid_visible         = Bool('mid')
    dual_visible        = Bool('dual')

    harmonics           = List()

    new                 = Button()
    save                = Button()
    save_as             = Button()
    load                = Button()

    wireframe           = Bool(False)


    subdivision         = Range(0, 8, 6)
    regenerate          = Button()

    reaction            = Button()
    reaction_type       = List()
    reaction_selected   = Str()

    brush_width         = Range(0.0, 0.5, 0.1)
    harmonic            = Button()
    eigen               = Float(1)

    vertex_normal       = Bool(False)
    mapping_height      = Bool(True)
    mapping_color       = Bool(True)

    datamodel           = Instance(DataModel)
    orientation         = Instance(Quaternion, (0,0,0,1))
    domain              = Instance(Domain)

    zoom                = Float(1.0)

    filedir             = Str('')
    filename            = Str('')



    def __init__(self, datamodel):
        super(HeightEditor, self).__init__()

        self.datamodel = datamodel

        self.reaction_type = reaction_diffusion.ReactionDiffusion.params.keys()
        self.reaction_selected = self.reaction_type[0]


    #forward datamodel properties
    @property
    def hierarchy(self):
        return self.datamodel.hierarchy
    @property
    def complex(self):
        return self.datamodel.complex
    @property
    def group(self):
        return self.datamodel.group


    def _new_fired(self):
        #pop up file dialog
        sg = SelectGroup()
        sg.configure_traits(kind='livemodal')
        if sg.selected_group:

            print('generating new group; this may take a few seconds')

            def worker():

                if MAYAVI_THREADING:
                    print('this is reached')
                    print(np.ones(4))        #already here we have trouble!
                    print('this is not')

                """The threaded function"""
                kwargs = dict(N=sg.N) if sg.first=='Dihedral' else {}

                group = sg.selected_group(**kwargs)
                dm = DataModel(group)

                def inner():
                    ee = HeightEditor(dm)
                    ee.configure_traits()

                GUI.invoke_later(inner)

            if MAYAVI_THREADING:
                self.thr = Thread(target=worker)
                self.thr.start()
            else:
                worker()

##            return
##            print 'generating new group; this may take a few seconds'
##            kwargs = dict(N=sg.N) if sg.first=='Dihedral' else {}
##            from height_editor import HeightEditor
##            ee = EdgeEditor( HeightEditor(sg.selected_group(**kwargs)))
##            ee.configure_traits()
##    def _new_fired(self):
##        def dummy():
##            print 'dummy start'
##            print np.ones(4)
##            print 'dummy end'
##        self.thr = Thread(target=dummy)
##        self.thr.start()


    def _save_fired(self):
        try:
            import os.path
            self.datamodel.save(os.path.join(self.filedir, self.filename))
        except:
            self._save_as_fired()
    def _save_as_fired(self):
        file_wildcard = '*.*'
        dialog = FileDialog(action="save as", wildcard=file_wildcard, default_directory = self.filedir)
        dialog.open()
        if dialog.return_code == OK:
            try:
                self.datamodel.save(dialog.path)
                self.filedir = dialog.directory
                self.filename = dialog.filename
            except:
                import traceback
                traceback.print_exc()
                print('error saving file')
    def _load_fired(self):
        file_wildcard = '*.*'
        dialog = FileDialog(action="open", wildcard=file_wildcard, default_directory = self.filedir)
        dialog.open()
        if dialog.return_code == OK:
            try:
                ee = EdgeEditor(DataModel.load(dialog.path))
                ee.filedir = dialog.directory
                ee.filename = dialog.filename
                ee.configure_traits()
            except:
                import traceback
                traceback.print_exc()
                print('error loading file')



    def _regenerate_fired(self):
        print('regenerating')

        def worker():
            self.datamodel.regenerate(self.subdivision)
            def inner():
                with RenderLock(self.scene):
                    if self.domain: self.domain.remove()
                    self.domain = Domain(self)
                    self.domain.draw()

            GUI.invoke_later(inner)

        if MAYAVI_THREADING:
            self.thr = Thread(target=worker).start()
        else:
            worker()


    def _reaction_fired(self):
        def callback():
            with RenderLock(self.scene):
                self.domain.redraw()

        self.ta = ThreadedRDAction(self, self.datamodel, callback)
        self.ta.start()


    def _harmonic_fired(self):
        with RenderLock(self.scene):
##            self.eigen = self.complex.harmonic(self.eigen)
            self.datamodel.heightfield = self.complex.P0s * self.complex.eigenvectors[self.eigen]
            self.domain.redraw()




    def draw_scene(self):

        with RenderLock(self.scene):

            #fixed points
            self.primal = self.scene.mlab.points3d(
                *self.datamodel.primal().T,
                resolution=10,
                scale_factor = 0.05,
                color = (1,0,0))
            a = self.primal.actor.actors[0]
            a.pickable = False

            self.mid = self.scene.mlab.points3d(
                *self.datamodel.mid().T,
                resolution=10,
                scale_factor = 0.05,
                color = (0,1,0))
            a = self.mid.actor.actors[0]
            a.pickable = False

            self.dual = self.scene.mlab.points3d(
                *self.datamodel.dual().T,
                resolution=10,
                scale_factor = 0.05,
                color = (0,0,1))
            a = self.dual.actor.actors[0]
            a.pickable = False


            #enable optional symmetry plane/axis renderings. how to render point reflection though? nevermind i guess

            #cp cursor
            self.cursor = self.scene.mlab.points3d(
                *[0,0,0],
                resolution=10,
                scale_factor = 0.04,
                color = (1,0,1))
            self.cursor.visible = True


            #draw actual mesh
            self.domain = Domain(self)
            self.domain.draw()


    mouse_left_down  = False
    mouse_right_down = False
    anchor = None


    @on_trait_change('scene.activated')
    def post_activation(self):
        """stuff that needs to be run after mayavi init"""
        self.draw_scene()




        def get_world_to_view_matrix():
            """returns the 4x4 matrix that is a concatenation of the modelview transform and
            perspective transform. Takes as input an mlab scene object."""
##
##            if not isinstance(mlab_scene, MayaviScene):
##                raise TypeError('argument must be an instance of MayaviScene')


            # The VTK method needs the aspect ratio and near and far clipping planes
            # in order to return the proper transform. So we query the current scene
            # object to get the parameters we need.
            scene_size = tuple(self.scene.get_size())
            clip_range = self.scene.camera.clipping_range
            aspect_ratio = float(scene_size[0])/float(scene_size[1])

            # this actually just gets a vtk matrix object, we can't really do anything with it yet
            vtk_comb_trans_mat = self.scene.camera.get_composite_projection_transform_matrix(
                                        aspect_ratio, clip_range[0], clip_range[1])

             # get the vtk mat as a numpy array
            np_comb_trans_mat = vtk_comb_trans_mat.to_array()

            return np_comb_trans_mat


        def get_view_to_display_matrix():
            """ this function returns a 4x4 matrix that will convert normalized
                view coordinates to display coordinates. It's assumed that the view should
                take up the entire window and that the origin of the window is in the
                upper left corner"""
##
##            if not (isinstance(mlab_scene, MayaviScene)):
##                raise TypeError('argument must be an instance of MayaviScene')

            # this gets the client size of the window
            x, y = tuple(self.scene.get_size())

            # normalized view coordinates have the origin in the middle of the space
            # so we need to scale by width and height of the display window and shift
            # by half width and half height. The matrix accomplishes that.
            view_to_disp_mat = np.array([[x/2.0,      0.,   0.,   x/2.0],
                                         [   0.,   y/2.0,   0.,   y/2.0],
                                         [   0.,      0.,   1.,      0.],
                                         [   0.,      0.,   0.,      1.]])

            return view_to_disp_mat


##        print get_view_to_display_matrix(self.scene)
##        print get_world_to_view_matrix(self.scene)



        def point_picker_callback(picker):
            pos = picker._get_pick_position()
            print(pos)
##            pos = np.array( pos+(0,))
##            pos = np.linalg.solve(get_world_to_view_matrix(), pos)
##            print pos
##            pos = np.linalg.solve(get_view_to_display_matrix(), pos)
##            print pos








        def on_point_pick(vtk_picker, event):
            """rather superfluous"""
            picker = tvtk.to_tvtk(vtk_picker)
            point_picker_callback(picker)



        #hook up picker\


        picker = self.scene.picker.pointpicker

        picker = self.scene.picker.pointpicker
        picker.add_observer('EndPickEvent', on_point_pick)
        picker.tolerance = 3e-3

##        picker = self.scene.picker.worldpicker
##        picker.add_observer('EndPickEvent', worldpicker_callback)

        def mousewheel_forward(obj, event):
##            self.zoom *= 1.1
            self.scene.camera.zoom(1.1)
            GUI.invoke_later(self.scene.render)
        def mousewheel_backward(obj, event):
##            self.zoom /= 1.1
            self.scene.camera.zoom(0.9)
            GUI.invoke_later(self.scene.render)

        def mouse_right_press(obj, event):
            self.mouse_right_down = True
            self.anchor = surface_pick(obj.GetEventPosition())

        def mouse_right_release(obj, event):
            self.mouse_right_down = False
            self.anchor = None


        def sphere_pick(display_coords, radius = 1):
            """pick point on sphere by raycasting"""

            nx, ny = display_coords

            view_display = get_view_to_display_matrix()
            world_view = get_world_to_view_matrix()

            near, far = self.scene.camera.clipping_range

            pos = np.array(((nx,ny,near,1),(nx,ny,far,1)) , dtype=np.float).T        #pos in display coords
            pos = np.linalg.solve(view_display, pos)
            pos = np.linalg.solve(world_view, pos).T
            pos = pos[:, :3] / pos[:, 3:]
            d = pos[1]-pos[0]
            o = pos[0]
            A = np.dot(d,d)
            B = np.dot(o,d)*2
            C = np.dot(o,o) - radius**2
            D = B**2-4*A*C
            if D > 0:
                t = (-B-np.sqrt(D))/(2*A)
                p = o+d*t
                return p

        def surface_pick(display_coords):
            """pick point on surface"""
            nx, ny = display_coords

            q = self.scene.picker.pick_cell(nx,ny)
            if q.valid==1:
                return q.coordinate


        def mouse_left_press(obj, event):
            self.mouse_left_down = True
            p = sphere_pick( obj.GetEventPosition())
            if not p is None:
                self.trace = [p]


        def mouse_left_release(obj, event):
            self.mouse_left_down = False

            if len(self.trace) > 1:
                from .. import brushes
                brush = brushes.paint(self.hierarchy, self.trace, self.brush_width)

                self.datamodel.heightfield += brush

                with RenderLock(self.scene):
##                    self.cursor.actor.actors[0].position = p
                    self.domain.redraw()




        def mouse_move(obj, event):
            """
            general idea; pick point on first click; then match point to cursor by tracing ray from camera to sphere
            """
            if self.mouse_left_down:
                p = sphere_pick( obj.GetEventPosition())
                if not p is None:
                    self.trace.append(p)


            if self.mouse_right_down:
                """
                view_up, position and focal point
                """
                if not self.anchor is None:


                    target = sphere_pick(obj.GetEventPosition(), np.linalg.norm(self.anchor))
                    if target is None: return

                    delta_orientation = Quaternion.from_pair(target, self.anchor)
##                    delta_orientation = util.Quaternion.from_pair(self.anchor, target)

##                    self.orientation = self.orientation * delta_orientation
                    self.orientation = delta_orientation * self.orientation

                    self.anchor = target


                    view_up = np.eye(3)[1]
                    position = np.eye(3)[2]*6
                    M = self.orientation.to_matrix()
                    with RenderLock(self.scene):
                        self.scene.camera.position = np.dot(M, position)
                        self.scene.camera.view_up = np.dot(M, view_up)


                    target = sphere_pick(obj.GetEventPosition(), np.linalg.norm(self.anchor))
                    if target is None: return
                    self.anchor = target

                return



                view_up = np.eye(3)[1]
                position = np.eye(3)[2]*6
                nx, ny = obj.GetEventPosition()
                lx, ly = obj.GetLastEventPosition()
                dx, dy = nx-lx, ny-ly

##                self.orientation = self.orientation * util.Quaternion.from_axis_angle(0, dx*0.1) * util.Quaternion.from_axis_angle(1, dy*0.1)
                scale = 0.2
                delta_orientation = Quaternion.from_axis_angle(1, -dx*scale) * Quaternion.from_axis_angle(0, dy*scale)
##                self.orientation = delta_orientation * self.orientation
                self.orientation = self.orientation * delta_orientation

##                self.orientation = self.orientation * ( delta_orientation* self.orientation.inv()) * self.orientation
##                self.orientation = self.orientation * ( self.orientation.inv() * delta_orientation) * self.orientation

##                print self.orientation
##                a,b,c = self.orientation.to_spherical()
##                print 'orientation'
##                print a,b,c
##                self.scene.mlab.view(
##                    azimuth=a,
##                    elevation=b,
##                    roll=c,
##                    distance=None)

##                self.scene.camera.user_view_transform = self.scene.camera.view_transform_matrix.from_array( self.orientation.to_matrix())


                M = self.orientation.to_matrix()

##                print M
##                print self.scene.camera.position
##                print self.scene.camera.view_up

                with RenderLock(self.scene):
                    self.scene.camera.position = np.dot(M, position)
                    self.scene.camera.view_up = np.dot(M, view_up)
##                    self.scene.camera.direction_of_projection = \\-util.normalize( self.scene.camera.position)
##                    self.scene.camera.focal_point = (0,0,0)
##                print self.scene.camera.focal_point

                return

                view_display = get_view_to_display_matrix()
                world_view = get_world_to_view_matrix()
##                pos = np.array((nx,ny,1,1) , dtype=np.float)        #pos in display coords
                pos = np.array(((nx,ny,1,1),(nx,ny,10,1)) , dtype=np.float).T        #pos in display coords
                pos = np.linalg.solve(view_display, pos)
                print(pos)                                         #pos in viewport
                pos = np.linalg.solve(world_view, pos).T
##                print pos[:3]/pos[3] - self.scene.camera.position
                pos = pos[:, :3] / pos[:, 3:]
                print(pos)
                d = pos[1]-pos[0]
                o = pos[0]
                A = np.dot(d,d)
                B = np.dot(o,d)*2
                C = np.dot(o,o) - 1
                D = B**2-4*A*C
                print(D)

##                print pos[:3,0] / pos[3,0]
##                print pos[:3,1] / pos[3,1]
                print(self.scene.camera.position)

                return

                pos = np.array((0.0,0,0,1) )
                pos[:3] = self.scene.camera.position
                pos[2]-=1
                pos = np.dot(world_view, pos)
                print(pos)
                pos = pos / pos[3]
                print(pos)
                pos = np.dot(view_display, pos)
                print(pos)



##                print self.scene.camera.screen_top_right
##                print self.scene.camera.get()

##                print tvtk.to_vtk(self.scene.camera)
##                print get_world_to_view_matrix()

##                GUI.invoke_later(self.scene.render)
##                print self.scene.camera.get()
                return

##
##                #find camera position such that picked world point stays constant upon dragging?
##                print nx, ny
##                picker.pick((nx, ny, 0), self.scene.renderer)
##
##                pos = np.array((nx,ny,0,1) )
##                pos = np.linalg.solve(get_view_to_display_matrix(), pos)
##                print pos
##                pos = np.linalg.solve(get_world_to_view_matrix(), pos)
##                print pos
##                print np.linalg.norm(pos)
##
##
##                a,b,z,focal = self.scene.mlab.view()
##                self.scene.mlab.view(a-dx*0.1, b+dy*0.1)
##                GUI.invoke_later(self.scene.render)
##                ()

        def dummy(*args, **kwargs):
            print(args, kwargs)

        """
        setup interactor matters here
        """
        print(self.scene.mlab.view())
        if True:
##            self.scene.interactor.remove_all_observers()
            self.scene.interactor.interactor_style = tvtk.InteractorStyleUser()


            self.scene.interactor.add_observer('MouseWheelForwardEvent', mousewheel_forward)
            self.scene.interactor.add_observer('MouseWheelBackwardEvent', mousewheel_backward)
            #do a pick here
            self.scene.interactor.add_observer('LeftButtonPressEvent', mouse_left_press,1)
            self.scene.interactor.add_observer('LeftButtonReleaseEvent', mouse_left_release,1)
            self.scene.interactor.add_observer('RightButtonPressEvent', mouse_right_press, 1)
            self.scene.interactor.add_observer('RightButtonReleaseEvent', mouse_right_release, 1)
            self.scene.interactor.add_observer('MouseMoveEvent', mouse_move, 1)


        print(self.scene.camera.get())
        self.scene.camera.clipping_range = (1,10)
        cam =  tvtk.to_vtk( self.scene.camera)
##        print cam.GetScreenBottomLeft()




##        self.scene.interactor.add_observer('MouseMoveEvent',
##                                self.on_mouse_move)
        self.scene.interactor.add_observer('LeftButtonPressEvent',
                                self.on_button_press)


        #start working on improved mesh, once drawing done
##        self._regenerate_fired()



    def on_mouse_move(self, interactor, event):
        print(interactor.GetEventPosition())

    def on_button_press(self, interactor, event):
        pos = interactor.GetEventPosition()
        self.scene.picker.worldpicker.pick(pos+(0,), self.scene.renderer)
        self.scene.picker.pointpicker.pick(pos+(0,), self.scene.renderer)




    def _primal_visible_changed(self, new):
        self.primal.actor.visible = new
    def _mid_visible_changed(self, new):
        self.mid.actor.visible = new
    def _dual_visible_changed(self, new):
        self.dual.actor.visible = new

    def _vertex_normal_changed(self):
        with RenderLock(self.scene):
            self.domain.redraw()
    def _mapping_height_changed(self):
        with RenderLock(self.scene):
            self.domain.redraw()
    def _mapping_color_changed(self):
        with RenderLock(self.scene):
            self.domain.redraw()
    def _wireframe_changed(self):
        with RenderLock(self.scene):
            self.domain.redraw()





##    def _reaction_selected_changed(self):
##        reaction_type[]


    view = View(
                HSplit(
                    Group(
                        Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                             height=1000, width=1000, show_label=False)),
                    Group(
                        Group(
                            Item('primal_visible', label='Primal'),
                            Item('mid_visible',    label='Mid'),
                            Item('dual_visible',   label='Dual'),
                            Item('wireframe'),
                            Item('vertex_normal'),
                            Item('mapping_height'),
                            Item('mapping_color'),
                            show_border = True,
                            label = 'Visibility'
                        ),
                        Group(
                            Item('brush_width'),
                            Item('subdivision'),
                            Item('regenerate', enabled_when='complex!=None'),
                            Item(name='reaction_selected', show_label=False, editor=EnumEditor(name='reaction_type')),#, mode='list' #height=0.5,
                            Item('reaction', enabled_when='complex!=None'),
                            Item('harmonic', enabled_when='complex!=None'),
                            Item('eigen'),

                            show_border = True,
                            label = 'Reaction diffusion'
                        ),
                        HGroup(
                            Item('new'),
                            Item('save'),
                            Item('save_as'),
                            Item('load'),
                            show_border = True,
                            show_labels = False,

                        ),
                    )
                ),
            resizable=True,
            handler=EdgeHandler(),
            )




