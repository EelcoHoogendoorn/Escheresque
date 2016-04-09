
import numpy as np

from datamodel import DataModel



if __name__=='__main__':


    if False:
        from tetrahedral import ChiralTetrahedral as Group
##        from tetrahedral import Null as Group
##        from octahedral import Tetrahedral as Group
##        from octahedral import Null as Group
        from octahedral import Pyritohedral as Group
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
        dm = DataModel.load(r'C:\Users\Eelco Hoogendoorn\Dropbox\Escheresque\code\v2\test.sch')

print dm.edges