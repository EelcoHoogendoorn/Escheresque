

"""
dialog for selecting a symmetry group
"""

from collections import OrderedDict

from inspect import getmembers, isclass, getdoc

from escheresque.interface.interface import *
import escheresque.group

def build_mapping():
    """build mapping from polytopes to subgroups"""
    from escheresque.group.group import Group
    from escheresque.group import dihedral, tetrahedral, octahedral, icosahedral

    from inspect import getmembers, isclass
    #modules to be loaded

    geometries = ['dihedral', 'tetrahedral', 'octahedral', 'icosahedral']
    modules = OrderedDict( [(k,v) for k,v in locals().iteritems() if k in geometries])

    def getgroups(mod):
        """load all group subclasses from a module"""
        od = {k:g for k,g in getmembers(mod) if isclass(g) and issubclass(g, Group) and not g is Group}
        return OrderedDict(sorted(od.items(), key=lambda item: item[1].__name__))

    return OrderedDict([(name.capitalize(), getgroups(mod)) for name,mod in modules.iteritems()])


group_dict = build_mapping()





class SelectHandler(Handler):

    def close(self, info, isok):
        # Return True to indicate that it is OK to close the window.
        if not isok:
            info.object.second = ''     #reset selection
        return True


class SelectGroup(HasTraits):
    first_enum = List(group_dict.keys())
    second_enum = List

    first = Str
    second = Str

    select = Button
    description = Str

    N = Range(2, 10)


    order = Str
    index = Str
    chiral = Str
    rotations = Str



    traits_view = View(
        VSplit(
            HSplit(
                Group(
                    Label('Polyhedron'),
                    Item(name='first', show_label=False, height=0.5, style='custom', editor=EnumEditor(name='first_enum', mode='list')),
                    label       = 'Polyhedron',
                    show_border = True,
                    show_labels = True,
##                    dock        = 'tab',
##                    scrollable  = True,
                ),
                Group(
                    Label('SubGroup'),
                    Item(name='second', show_label=False, height=0.5, style='custom', editor=EnumEditor(name='second_enum', mode='list')),
                    Group(
                        Item('N'),
                        visible_when="first=='Dihedral'",
                    ),

                    label       = 'SubGroup',
                    show_border = True,
                    show_labels = True,
##                    dock        = 'tab',
##                    scrollable  = True,


                ),

            ),

            VGroup(
                Label('Description'),
                Item('description', style='readonly', show_label=False),
                Item('index', style='readonly'),
                Item('order', style='readonly'),
                Item('chiral', style='readonly'),
                Item('rotations', style='readonly'),

##                spring,
##                Item('select', show_label=False),
                show_labels = True,
                show_border = True,
            ),
        ),
        resizable=True,
        width = 800, height = 800,
        title = 'Select a symmetry group',
##        kind = 'modal',
        buttons = ['OK', 'Cancel'],
        handler = SelectHandler(),
    )

    def __init__(self):
        self.second_enum = []

    def _first_changed(self, new):
        self.second_enum = group_dict[self.first].keys()

    def _second_changed(self, new):
        sg = self.selected_group
        self.description    = getdoc( sg)
        self.order          = str(sg.order)
        self.index          = str(sg.index)
        self.chiral         = str(sg.mirrors==1)
        self.rotations      = str(sg.order / sg.mirrors)


    @property
    def selected_group(self):
        try:
            return group_dict[self.first][self.second]
        except:
            return None


if __name__=='__main__':
    example = SelectGroup()
    example.configure_traits()

