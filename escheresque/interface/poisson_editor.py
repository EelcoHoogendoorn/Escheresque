
"""
poisson editor with emerged boundary conditions




create toggle button; conform curve to mesh, or show actual value
also, toggle display of height info
when height is active, all elements, including


store radius relative to basis ( multiple or offset from curve)
edit in absolute sense

what is edge coord system? should we use a conformal coordsys instead?
like fields between pos-neg charge

"""

from escheresque import util
from escheresque.datamodel import DataModel, Constraints

from escheresque.interface.interface import *




class EdgeHandler(Handler):
    """
    example of a handler for smoothing over saving and shit
    edgehandler is wrong name, really
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
        return response == YES

        if not info.object.saved:
            response = confirm(info.ui.control,
                            "Value is not saved.  Are you sure you want to exit?")
            return response == YES
        else:
            return True






class Point(HasTraits):
    """
    plot elements for all occurances of a single point
    """
    def __init__(self, parent, point):
        super(Point, self).__init__()
        self.parent = parent        # parent ui model
        self.point = point          # datamodel handle

    @property
    def scene(self):
        return self.parent.scene
    @property
    def datamodel(self):
        return self.parent.datamodel


    def draw(self):
        def add_mirror(points):
            x,y,z = points.T
            dummy = self.scene.mlab.points3d(
                x,y,z,
                resolution=10,
                scale_factor = 0.03,
                reset_zoom = False,
                color = black,
                )
    ##        self.visible.actor.actors[0].pickable = False
            a = dummy.actor.actors[0]

            #lock actor in place
            lock_actor(a)
            a.dragable = 0
##            a.pickable = False

            return dummy

        self.mirrors = [add_mirror(mirror) for mirror in self.point.instantiate()]

        self.redraw()

    def set_visible(self, value):
        """recolor all actors"""
        for mirror in self.mirrors:
            mirror.actor.visible = value


    def redraw(self):
        for mirror, point in zip(self.mirrors, self.point.instantiate()):
            if self.parent.height_visible:
                from .. import brushes
                point *= brushes.Mapping(self.datamodel.hierarchy, point).sample(self.datamodel.heightfield)[:, None]
            x,y,z = point.T
            mirror.mlab_source.reset(x=x,y=y,z=z)

    def remove(self):
        for mirror in self.mirrors:
            mirror.remove()



class Edge(HasTraits):
    """
    plot elements for all occurances of a single edge
    use pipeline construction here too; only create rotated actors
    """
    def __init__(self, parent, edge):
        super(Edge, self).__init__()
        self.parent = parent
        self.edge = edge

    @property
    def scene(self):
        return self.parent.scene
    @property
    def group(self):
        return self.parent.group
    @property
    def datamodel(self):
        return self.parent.datamodel

    def draw(self):
        def add_mirror(curve):
            """
            would be better to construct only part of the pipeline that is required, and not need the invisible copy
            """
##            print curve.shape
            dummy = self.scene.mlab.plot3d(
                *curve.T,
                tube_radius = 0.005,
##                tube_radius = None,
    ##            representation='wireframe',
                reset_zoom = False,
                color = black)
            dummy.visible=False
            return dummy

        self.mirrors = [add_mirror(mirror[0]) for mirror in self.edge.instantiate()]


        def add_instance(mirror, R):
            surf = self.scene.mlab.pipeline.surface(
                mirror.module_manager,
                color=black,
                reset_zoom = False)
            a = surf.actor.actors[0]

            #set rotation matrix
            rotate_actor(a, R)
            #lock actor in place
            lock_actor(a)
            a.dragable = 0

            return surf

        self.curves = np.array( [[add_instance(mirror, R) for R in rotations]
                                        for mirror, rotations in zip(self.mirrors, self.group.rotations)])

        self.redraw()
        self.recolor()


    def redraw(self, reset=True):
        """only need to update the root datasource"""
        coords = self.edge.instantiate()
        for mirror, coord in zip(self.mirrors, coords):
            if not self.parent.height_visible:
                coord = util.normalize(coord)
            x,y,z = coord[0].T
            func = mirror.mlab_source.reset if reset else mirror.mlab_source.set
            func(x=x,y=y,z=z)

    def recolor(self):
        """recolor all actors"""
        for mcurves in self.curves:
            for curve in mcurves:
                curve.actor.actors[0]._vtk_obj.GetProperty().SetDiffuseColor(self.edge.color)

    def set_visible(self, value):
        """recolor all actors"""
        for mcurves in self.curves:
            for curve in mcurves:
                curve.actor.visible=value

    def remove(self):
        for curve in self.curves.flatten():
            curve.remove()
        for mirror in self.mirrors:
            mirror.remove()



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
            FV = self.complex.topology.FV[:,::int(np.sign(np.linalg.det(B)))]        #reverse vertex order depending on orientation
            source  = self.scene.mlab.pipeline.triangular_mesh_source(x,y,z, FV)

            #put these guys under a ui list
            l=lut_manager.tvtk.LookupTable()
            lut_manager.set_lut(l, lut_manager.pylab_luts['jet'])
##            lut_manager.set_lut(l, lut_manager.pylab_luts.values()[0])

            #add polydatamapper, to control color mapping and interpolation
##            mapper = tvtk.PolyDataMapper(input=source.outputs[0], lookup_table=l)
            mapper = tvtk.PolyDataMapper()
            from tvtk.common import configure_input
            configure_input(mapper, source.outputs[0])

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
            a.pickable = False
            a.property.backface_culling = True
##            a.property.representation = 'wireframe' if self.parent.wireframe else 'surface'
            a.property.representation = 'surface'

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
        reset height and related info (normals)
        """
        radius = self.datamodel.heightfield
        if not self.parent.height_visible:
            radius = np.ones_like(radius)

        #precompute normal information
        if self.parent.vertex_normal:
            normals = self.complex.normals(radius)

        self.recolor()

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

                #set normals
                if self.parent.vertex_normal:
                    M = self.group.transforms[mirror,0]
                    source.data.point_data.normals = np.dot(M, normals[index,:,:].T).T
                    source.data.cell_data.normals = None
                else:
                    source.data.point_data.normals = None
                    source.data.cell_data.normals = None

                # needed to force a redraw of these elements
                source.mlab_source.update()

##        for i in self.instances:
##            i.property.representation = 'wireframe' if self.parent.wireframe else 'surface'

    def recolor(self):
        color = self.complex.P0D2 * self.datamodel.forcefield
        m = np.abs(color).max() + 1e-12
        color = color / m / 2 + 0.5

        for index,T in enumerate( self.root):
            for mirror,M in enumerate(T):
                _, source, mapper = M

                if self.parent.curvature_visible:
                    col = color[:,index]
                else:
                    col = np.empty((self.complex.topology.P0, 3), np.uint8)
                    #set colors
                    if self.parent.group_visible and mirror==0:
                        col[:] = 200
                    else:
                        col[:] = 230
                source.mlab_source.set(scalars=col)
                source.mlab_source.update()


    def remove(self):
        for i in self.instances:
            self.scene.remove_actors(i)
##        for T in self.root:
##            for m in T:
##                mirror, source, mapper = m
##                mapper.remove()
##                source.remove()








class PoissonEditor(HasTraits):
    """
    editor for curve controlled heightfield

    basically a branch of curve editor
    merge these codebases at a later point
    """

    scene               = Instance(MlabSceneModel, args=())

    group_visible       = Bool(True)
    sketch_visible      = Bool(True)
    curvature_visible   = Bool(False)
    height_visible      = Bool(True)

    add_point_toggle    = Bool(False)
    add_edge_toggle     = Bool(False)
    add_cp_toggle       = Bool(False)


    mapping_height      = Bool(True)
    mapping_color       = Bool(False)
    vertex_normal       = Bool(True)

    selected_point      = Any #(point-object, transform)
    selected_edge       = Any #(edge-object, transform)
    selected_cp         = Any #(controlpoint, transform)

    delete_point        = Button()
    delete_edge         = Button()
    delete_cp           = Button()

    color_edge          = Color()
    boundary_edge       = Bool()
    driving_edge        = Bool()
    width_edge          = Range(0, 0.2, 0.02)

    constraints         = Enum(Constraints)

    tension_slider      = Range(0.0,1.0)
    radius_cp           = Float(1.)

    new                 = Button()
    save                = Button()
    save_as             = Button()
    load                = Button()
    export              = Button()
    update              = Button()

    datamodel           = Instance(DataModel)
    points              = List(Point)
    edges               = List(Edge)


    def __init__(self, datamodel):
        super(PoissonEditor, self).__init__()
        self.datamodel = datamodel
        #create plot items; but hold off on drawing them!
        self.points = [Point(self, p) for p in datamodel.points]
        self.edges  = [Edge (self, e) for e in datamodel.edges ]


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


    def add_point(self, pos):
        point = Point(self, self.datamodel.add_point(pos))
        self.points.append(point)
        with RenderLock(self.scene):
            point.draw()
    def remove_point(self, point):
        if any(point.point in e.edge.points for e in self.edges): return #only remove dangling points
        self.datamodel.points.remove(point.point)
        self.points.remove(point)
        point.remove()

    def add_edge(self, l, r):
        edge = Edge(self, self.datamodel.add_edge(l,r))
        self.edges.append(edge)
        with RenderLock(self.scene):
            edge.draw()
    def remove_edge(self, edge):
        self.datamodel.edges.remove(edge.edge)
        self.edges.remove(edge)
        edge.remove()

    def add_cp(self, idx):
        if not self.selected_edge: return
        edge, index = self.selected_edge
        edge.edge.curve.add_cp(idx, idx+1)
        with RenderLock(self.scene):
            edge.redraw()
            self.redraw_control()
    def remove_cp(self, idx):
        if not self.selected_edge: return
        edge, index = self.selected_edge
        edge.edge.curve.remove_cp(idx)
        with RenderLock(self.scene):
            edge.redraw()
            self.redraw_control()

    def _delete_point_fired(self):
        point, index = self.selected_point
        self.remove_point(point)
        self.selected_point = None
    def _delete_edge_fired(self):
        edge, index = self.selected_edge
        self.remove_edge(edge)
        self.selected_edge = None
    def _delete_cp_fired(self):
        self.remove_cp(self.selected_cp)
        self.selected_cp = None


    filedir  = Str('')
    filename = Str('')
    def _new_fired(self):
        #pop up file dialog
        from selectgroup import SelectGroup
        sg = SelectGroup()
        sg.configure_traits(kind='livemodal')
        if sg.selected_group:
            print 'generating new group; this may take a few seconds'
            def worker():
                """The threaded function"""
                kwargs = dict(N=sg.N) if sg.first=='Dihedral' else {}

                group = sg.selected_group(**kwargs)
                dm = DataModel(group)
                dm.generate(6)

                def dinges():
                    ee = PoissonEditor(dm)
                    ee.configure_traits()

                GUI.invoke_later(dinges)

            if MAYAVI_THREADING:
                self.thr = Thread(target=worker).start()
            else:
                worker()

    def _save_fired(self):
        try:
            import os.path
            self.datamodel.save(os.path.join(self.filedir, self.filename))
            print 'saved'
        except:
            self._save_as_fired()

    def _save_as_fired(self):
        file_wildcard = '*.sch'
        dialog = FileDialog(action="save as", wildcard=file_wildcard, default_directory = self.filedir)
        dialog.open()
        if dialog.return_code == OK:
            try:
#                print self.datamodel.edges
                self.datamodel.save(dialog.path)
#                print self.datamodel.edges
#                print DataModel.load(dialog.path).edges
                self.filedir = dialog.directory
                self.filename = dialog.filename
            except:
                import traceback
                traceback.print_exc()
                print 'error saving file'

    def _load_fired(self):
        file_wildcard = '*.sch'
        dialog = FileDialog(action="open", wildcard=file_wildcard, default_directory = self.filedir)
        dialog.open()
        if dialog.return_code == OK:
            try:
                ee = PoissonEditor(DataModel.load(dialog.path))
                ee.filedir = dialog.directory
                ee.filename = dialog.filename
                ee.configure_traits()
            except:
                import traceback
                traceback.print_exc()
                print 'error loading file'


    def _export_fired(self):
        """
        export domains to stl

        todo: create an export window, to override default bound curves
        """
        partitions = self.datamodel.partition()

        filename = r'C:\Users\Eelco\Dropbox\Git\Escheresque\data\part{0}.stl'

        if True:
            from mayavi import mlab
            from escheresque import stl, computational_geometry, brushes

##            if False:
##                x,y,z = curve_p.T
##                # Create the points
##                src = mlab.pipeline.scalar_scatter(x, y, z)
##                # Connect them
##                src.mlab_source.dataset.lines = curve_idx
##                # The stripper filter cleans up connected lines
##                lines = mlab.pipeline.stripper(src)
##                # Finally, display the set of lines
##                mlab.pipeline.surface(lines, colormap='Accent', line_width=10)

            for i,(p,t) in enumerate(partitions):
                p = p * self.datamodel.sample(p)[:,None]
                # p,t = computational_geometry.extrude(p*1, p*0.95, t)
                p,t = computational_geometry.swept_extrude(p, t, 0.1)


                stl.save_STL(filename.format(i), p[t])

            q = p.mean(axis=0)
            x,y,z= p.T + q[:,None]/4
            mlab.triangular_mesh(x,y,z, t)
            mlab.triangular_mesh(x,y,z, t, color = (0,0,0), representation='wireframe')

##            mlab.show()
##        if False:
##            radius = np.linspace(0.99, 1.01, len(partitions))
##            from mayavi import mlab
##            for i,(p,c) in enumerate(izip( partitions, [(1,0,0), (0,1,0),(0,0,1),(0,1,1),(1,0,1),(1,1,0)])):
##                solid_points, solid_triangles = extrude(allpoints, p, radius[i], 0.8)
##
##                x,y,z= solid_points.T
##
##                mlab.triangular_mesh(x,y,z, solid_triangles, color = c,       representation='surface')
##                mlab.triangular_mesh(x,y,z, solid_triangles, color = (0,0,0), representation='wireframe')



    def _update_fired(self):
        """update the heightfield in accordance with the guide curves, and redraw"""
        self.datamodel.update()
        with RenderLock(self.scene):
            self.domain.redraw()

    def redraw_point(self, point):
        """perform updates to plots after point move"""
        point.redraw()
        # FIXME: updating the edges causes a crash during point selection
        # for edge in self.edges:
        #     if point.point in edge.edge.points:
        #         edge.redraw(False)

    def redraw_control(self):
        """redraw control mesh"""
        edge, index = self.selected_edge

        cp = edge.edge.curve.controlpoints()[index]
        if not self.height_visible: cp = util.normalize(cp)
        self.control_points.mlab_source.reset(points=cp)

        cm = edge.edge.curve.controlmesh()[index]
        if not self.height_visible: cm = util.normalize(cm)
        self.control_mesh.mlab_source.reset(points=cm)

    def redraw_group(self):
        """redraw primal/mid/dual markers"""
        with RenderLock(self.scene):
            p = self.datamodel.primal()
            if self.height_visible: p = p * self.datamodel.sample(p)[:,None]
            self.primal.mlab_source.reset(points=p)

            p = self.datamodel.mid()
            if self.height_visible: p = p * self.datamodel.sample(p)[:,None]
            self.mid.mlab_source.reset(points=p)

            p = self.datamodel.dual()
            if self.height_visible: p = p * self.datamodel.sample(p)[:,None]
            self.dual.mlab_source.reset(points=p)


    def on_mouse_move(self, interactor, event):
        print interactor.GetEventPosition()

    def on_button_press(self, interactor, event):
        pos = interactor.GetEventPosition()
        self.scene.picker.worldpicker.pick(pos+(0,), self.scene.renderer)
        self.scene.picker.pointpicker.pick(pos+(0,), self.scene.renderer)

    def _group_visible_changed(self, new):
        with RenderLock(self.scene):
            self.primal.actor.visible = new
            self.mid.actor.visible = new
            self.dual.actor.visible = new
            self.domain.recolor()

    def _curvature_visible_changed(self, new):
        with RenderLock(self.scene):
            self.domain.recolor()

    def _sketch_visible_changed(self, new):
        with RenderLock(self.scene):
            for edge in self.edges:
                edge.set_visible(new)
            for point in self.points:
                point.set_visible(new)
            self.control_mesh.actor.visible = new
            self.cursor.actor.visible = new

    def _height_visible_changed(self, new):
        with RenderLock(self.scene):
            for edge in self.edges:
                edge.redraw()
            for point in self.points:
                point.redraw()
            self.domain.redraw()
        self.redraw_group()



    def _selected_point_changed(self, new):
        with RenderLock(self.scene):
            if new is not None:
                point, index = self.selected_point
                p = point.point.instantiate()[index]
                self.cursor.actor.actors[0].position = p*self.datamodel.sample(p)

                self.radius_cp = point.point.radius
                self.cursor.visible = True

            else:
                self.cursor.visible = False


    def _selected_edge_changed(self, new):
##            self.selected_cp = None
        with RenderLock(self.scene):
            if new is not None:
                self.redraw_control()

                self.control_points.visible = True
                self.control_mesh.visible = True

                edge = new[0].edge


                self.color_edge    = tuple([c*255 for c in edge.color])
                self.boundary_edge = edge.boundary
                self.driving_edge  = edge.driving
                self.width_edge    = edge.width

            else:
                self.control_points.visible = False
                self.control_mesh.visible = False

    def _selected_cp_changed(self, new):
        with RenderLock(self.scene):
            if new:
                edge, index = self.selected_edge
                x,y,z = edge.edge.curve.controlpoints()[index][new]

                self.cursor.actor.actors[0].position = (x,y,z)
                self.cursor.visible = True

                self.radius_cp = edge.edge.curve.radius[self.selected_cp]

            else:
                self.cursor.visible = False


    def _constraints_changed(self, new):
        if self.selected_point is None: return
        point, index  = self.selected_point
        newpos = point.point.set_constraint(new)

        with RenderLock(self.scene):
            self.redraw_point(point)
            self.cursor.actor.actors[0].position = newpos

    def _color_edge_changed(self, new):
        if self.selected_edge is None: return
        edge, index = self.selected_edge
        edge.edge.color = self.color_edge.redF(), self.color_edge.greenF(), self.color_edge.blueF()
        with RenderLock(self.scene):
            edge.recolor()

    def _boundary_edge_changed(self, new):
        if self.selected_edge is None: return
        edge, index = self.selected_edge
        edge.edge.boundary = self.boundary_edge

    def _driving_edge_changed(self, new):
        if self.selected_edge is None: return
        edge, index = self.selected_edge
        edge.edge.driving = self.driving_edge

    def _width_edge_changed(self, new):
        if self.selected_edge is None: return
        edge, index = self.selected_edge
        edge.edge.width = self.width_edge

    def _tension_slider_changed(self):
        if self.selected_cp is None: return
        edge, index = self.selected_edge
        edge.edge.curve.tension[self.selected_cp] = self.tension_slider
        with RenderLock(self.scene):
            edge.redraw(True)

    def _radius_cp_changed(self, new):
        if self.selected_cp is not None:
            edge, index = self.selected_edge
            edge.edge.curve.radius[self.selected_cp] = self.radius_cp
            # with RenderLock(self.scene):
            edge.redraw(True)
        if self.selected_point is not None:
            point, index = self.selected_point

            point.point.radius = self.radius_cp
            # with RenderLock(self.scene):
            self.redraw_point(point)
            #reset cursor
            p = self.cursor.actor.actors[0].position
            self.cursor.actor.actors[0].position = p / np.linalg.norm(p) * self.radius_cp



    def draw_scene(self):

        def trackball(obj, name, old, new):
            """
            trackball callback
            """
            #compute new position of cursor in world space
            mark = np.array( self.cursor.actor.actors[0].position)
            mark = mark / np.linalg.norm(mark)
            from escheresque.util import rotation_matrix
            newpos = np.dot( rotation_matrix(new), np.dot(rotation_matrix(old).T, mark))

            if self.selected_point:
                #move selected selected point accordingly
                point, index = self.selected_point
                newpos = point.point.match(index, newpos)
                with RenderLock(self.scene):
                    self.redraw_point(point)
                    self.cursor.actor.actors[0].position = (newpos * self.datamodel.sample(newpos))

            if self.selected_cp:
                #convert newpos into new spline coords
                with RenderLock(self.scene):
                    edge, index = self.selected_edge
                    edge.edge.curve.match(index, self.selected_cp, newpos)

                    edge.redraw(False)
                    self.redraw_control()
                    self.cursor.actor.actors[0].position = (newpos * self.datamodel.sample(newpos))



        with RenderLock(self.scene):
            #actors
            for e in self.edges:
                e.draw()
            for p in self.points:
                p.draw()

            #center sphere; only control mesh left
##                self.sphere = self.scene.mlab.points3d(
##                    *[0,0,0],
##                    resolution=40,
##                    scale_factor = 2,
##                    color = (1,1,1)
##                    )
##                self.sphere.actor.actors[0].pickable = False
##            self.sphere.visible=False

            self.ghost_sphere = self.scene.mlab.points3d(
                *[0,0,0],
                resolution=20,
                scale_factor = 2,
                color = (1,1,1),
                opacity = 1e-6
                )
            lock_actor(self.ghost_sphere.actor.actors[0], orientation=False)
            self.ghost_sphere.actor.actors[0].on_trait_change(trackball, 'orientation')

            self.wall_sphere = self.scene.mlab.points3d(
                *[0,0,0],
                resolution=20,
                scale_factor = 1.8,
                color = (1,1,1),
                opacity = 1
                )



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


            #enable optional symmetry plane/axis renderings. how to render point reflection though?




            #control mesh
            self.control_points = self.scene.mlab.points3d(
                *[0,0,0],
                resolution=10,
                scale_factor = 0.03,
                color = gray,
                colormap="gray")
            self.control_points.visible = False
            lock_actor(self.control_points.actor.actors[0])

            self.control_mesh = self.scene.mlab.plot3d(
                *[0,0,0],
                tube_radius = 0.003,
                color = gray)
            self.control_mesh.visible = False
            lock_actor(self.control_mesh.actor.actors[0])

            #cp cursor
            self.cursor = self.scene.mlab.points3d(
                *[0,0,0],
                resolution=10,
                scale_factor = 0.04,
                color = (1,0,1))
            self.cursor.visible = False
            self.cursor.actor.actor.pickable = False


            #dummy mesh
            self.domain = Domain(self)
            self.domain.draw()

        self.redraw_group()



    @on_trait_change('scene.activated')
    def post_activation(self):
        """stuff that needs to be run after mayavi init"""
        self.draw_scene()

        def point_picker_callback(picker):

            #pick points
            for point in self.points:
                for m,mirror in enumerate(point.mirrors):
                    if picker.actor in mirror.actor.actors:
                        factor = mirror.glyph.glyph_source.glyph_source.output.points.to_array().shape[0]
                        point_id = picker.point_id / factor
                        if point_id != -1:
                            newpoint = point, (m,point_id)
                            if self.add_edge_toggle and self.selected_point and self.selected_point!=newpoint:
                                self.add_edge(self.selected_point, newpoint)
                                self.add_edge_toggle = False

                            self.selected_cp = None
                            self.selected_edge = None

                            self.selected_point = None
                            self.constraints = point.point.constraint
                            self.selected_point = newpoint



            #pick edges
            for edge in self.edges:
                for m, mcurves in enumerate(edge.curves):
                    for r, curve in enumerate(mcurves):
                        if picker.actor in curve.actor.actors:

                            if self.add_cp_toggle:
                                #this might need updating! should be able to read out the vals
                                #perhaps also make adjustable, for performance reasons?
                                subdivisions = 5
                                resolution = 6
                                id = picker.point_id // (2**subdivisions) // resolution
                                self.add_cp(id)
                                self.add_cp_toggle = False

                            picked = edge, (m,r)
                            if picked == self.selected_edge: return

                            self.selected_point = None
                            self.selected_cp = None

                            self.selected_edge = None
                            self.color_edge = self.color_edge.fromRgbF(*edge.edge.color)   #this does not appear to trigger a UI update...
                            self.selected_edge = picked

            #pick control point
            if picker.actor in self.control_points.actor.actors:
                factor = self.control_points.glyph.glyph_source.glyph_source.output.points.to_array().shape[0]

                point_id = picker.point_id / factor
                # If the no points have been selected, we have '-1'
                if point_id != -1:
                    new_cp = point_id + 1

                    edge, mr = self.selected_edge

                    self.selected_cp = None
                    self.tension_slider = edge.edge.curve.tension[new_cp]
                    self.selected_cp = new_cp


            if self.add_point_toggle:
                pos = picker._get_pick_position()
                self.add_point(pos)
                self.add_point_toggle = False


        def worldpicker_callback(picker, event):
            worldpos = picker.GetPickPosition()
            if self.add_point_toggle:
                self.add_point(worldpos)
                self.add_point_toggle = False


        def on_point_pick(vtk_picker, event):
            """rather superfluous"""
            picker = tvtk.to_tvtk(vtk_picker)
            point_picker_callback(picker)



        #hook up picker
        picker = self.scene.picker.pointpicker
        picker.add_observer('EndPickEvent', on_point_pick)
        picker.tolerance = 3e-3

        # picker = self.scene.picker.worldpicker
        # picker.add_observer('EndPickEvent', worldpicker_callback)

        def mousewheel(*args, **kwargs):
            print args, kwargs


        """
        setup interactor matters here
        """
        if False:
##                self.scene.interactor.remove_all_observers()

            self.scene.interactor.add_observer('MouseWheelForwardEvent', mousewheel)
            #do a pick here
            self.scene.interactor.add_observer('LeftButtonPressEvent', mousewheel)
            self.scene.interactor.add_observer('LeftButtonReleaseEvent', mousewheel)
            self.scene.interactor.add_observer('RightButtonPressEvent', mousewheel)
            self.scene.interactor.add_observer('MouseMoveEvent', mousewheel)




##        self.scene.interactor.add_observer('MouseMoveEvent',
##                                self.on_mouse_move)
        self.scene.interactor.add_observer('LeftButtonPressEvent',
                                self.on_button_press)



    view = View(
                HSplit(
                    Group(
                        Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                             height=1000, width=1000, show_label=False)),
##                    Group(
                    Group(
                        Group(
                            Item('group_visible',  label='Group'),
                            Item('sketch_visible',  label='Sketch'),
                            Item('curvature_visible',  label='Curvature'),
                            Item('height_visible',  label='Relief'),
                            Item('vertex_normal', label='Normal interpolation'),

                            show_border = True,
                            label = 'Visibility'
                        ),
                        Group(
                            Item('add_point_toggle', label='Add Point'),
                            Item('add_edge_toggle', label='Add Edge'),
                            Item('radius_cp', label='Radius'),

                        ),
                        #point controls
                        Group(
                            Item('delete_point', label='Delete'),
                            Item(name='constraints', label='Constraint',
                                                  editor=EnumEditor(values={c: '{i}:{n}'.format(i=i,n=str(c)) for i,c in enumerate(Constraints)})),
                            enabled_when = 'selected_point != None',
                            label = 'Point',
                            show_border = True,
                            show_labels = True,
                        ),
                        #edge controls
                        Group(
                            Item('delete_edge', label=''),
                            Item('color_edge', style='custom', label='Color'),
                            Item('add_cp_toggle', label='Add Control Point'),
                            Item('boundary_edge', label='Use as tile boundary'),
                            Item('driving_edge', label='Use as driven boundary'),
                            Item('width_edge', label='Edge width'),
                            enabled_when = 'selected_edge != None',
                            label = 'Edge',
                            show_border = True,
                            show_labels = True,
                        ),
                        #control point controls
                        Group(
                            Item('tension_slider', label='Tension'),
                            Item('delete_cp', label='Delete'),
                            enabled_when = 'selected_cp != None',
                            label = 'Control Point',
                            show_border = True,
                            show_labels = False,
                        ),
                        HGroup(
                            Item('new'),
                            Item('save'),
                            Item('save_as'),
                            Item('load'),
                            Item('export'),
                            Item('update'),
                            show_border = True,
                            show_labels = False,
                        ),
##                    ),
##                    Group(
##                    ),
##                    layout='tabbed',
                    )
                ),
            resizable=True,
            handler=EdgeHandler(),
            )







##class MvtPicker(object):
##    mouse_mvt = False
##
##    def __init__(self, picker):
##        self.picker = picker
##
##    def on_button_press(self, obj, evt):
##        self.mouse_mvt = False
##
##    def on_mouse_move(self, obj, evt):
##        self.mouse_mvt = True
##
##    def on_button_release(self, obj, evt):
##        if not self.mouse_mvt:
##            x, y = obj.GetEventPosition()
##            self.picker.pick((x, y, 0), f.scene.renderer)
##        self.mouse_mvt = False
##
##
##
##mvt_picker = MvtPicker(picker.pointpicker)
##
##f.scene.interactor.add_observer('LeftButtonPressEvent',
##                                mvt_picker.on_button_press)
##f.scene.interactor.add_observer('MouseMoveEvent',
##                                mvt_picker.on_mouse_move)
##f.scene.interactor.add_observer('LeftButtonReleaseEvent',
##                                mvt_picker.on_button_release)
##





##        def _primal_visible_changed(self, new):
##            self.primal.actor.visible = new
##        def _mid_visible_changed(self, new):
##            self.mid.actor.visible = new
##        def _dual_visible_changed(self, new):
##            self.dual.actor.visible = new
