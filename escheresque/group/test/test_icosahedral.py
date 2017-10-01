
# import matplotlib.pyplot as plt
from escheresque.group import icosahedral


def test_generate():
    # group = icosahedral.Origin()
    group = icosahedral.ChiralIcosahedral()
    domains = group.domains
    print (domains.shape)

    for e in group.edges:
        print (e.shape)
    for e in group.edge_transforms:
        print (e.shape)

    for v in group.vertices:
        print (v)
    for v in group.vertices_transforms:
        print (v)



test_generate()