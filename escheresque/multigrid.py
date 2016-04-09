
"""
multigrid operations
"""


def solve_mg(complex, operator, rhs, x0):
    """multigrid solution. do some walks over the complex, then turn to cg?"""


def mg_diffuse(hierarchy, x, time):
    """
    mg diffusion over a hierarchy of complexes
    1/ratio defines the multiplicative impact we will have on the longest waves, by doing one diffusion step with a normalized laplace
    first, walk down to find starting level
    does the act of waling down not also have a significant effect?
    do not aim to always entirely close the remaining gap; prefer to do 4 more steps on coarse level instead

    normal diffusion is conservative in D2; can we construct transfer such that this property is maintained?
    something to worry about for later id say

    """
    min_level = 3
    levels = len(hierarchy)
    level = levels-1
    while level > min_level:
        complex = hierarchy[level]
        ratio = complex.ratio
        if True:        #if less than two steps at this level are warranted
            break
        #transfer x to lower level. recursive function already looking good...
        x = complex.restrict(x)

        level -= 1

    #smooth on coarsest level

    #loop in the other direction. interpolate result
    while level < levels:
        x = complex.interpolate(x)
        level += 1

        if True:
            break




def mg_diffuse_fixed(hierarchy, x):
    """
    mg diffusion over a hierarchy of complexes
    simple fixed number of steps on each level
    dynamic stepping can be added later. note that minimum number of poststeps is important though; it is hard to do without

    normal diffusion is conservative in D2; can we construct transfer such that this property is maintained?
    lack of conservation appears quite profound.

    demand x as P0 input?

    """
    min_level = 4
    iterations = 5          #4 iterations are needed to avoid noticable interpolation error; 5 to be safe

    levels = len(hierarchy)
    level = levels-1
    complex = hierarchy[level]
    while level > min_level:
        for i in xrange(iterations):            #pre-smoothing produces more regular behavior; but are we just masking a more fundamental issue here?
            x = complex.diffuse_normalized(x)
        level -= 1
        complex = hierarchy[level]
        x = complex.restrict(x)         #restrict onto coarse complex; P0 to P0 mapping

    #smooth on coarsest level
    for i in xrange(iterations):
        x = complex.diffuse_normalized(x)

    #loop in the other direction. interpolate result
    while level+1 < levels:
        x = complex.interpolate(x)
        level += 1

        complex = hierarchy[level]
        for i in xrange(iterations):
            x = complex.diffuse_normalized(x)

    return x


