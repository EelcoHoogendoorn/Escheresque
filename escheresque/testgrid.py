
#rhs = s1,s2, 0,0, e1,e2,e3, v1,v2,v3,

#vec = v1e1,v1e2,v1e3, v2e1,v2e2,v2e3, v3e1,v3e2,v3e3

#slivers; off diagonals after elimination of interaction 2-term
s1 = [0,0,0, 0,0,0, 1,0,0]
s2 = [0,0,1, 0,0,0, 0,0,0]

#non-interacting terms have zero-constraint
c0 = [1,0,0, 0,0,0, 0,0,0]
c1 = [0,0,0, 0,0,0, 0,0,1]

#sliver plus another vert is edge-measurement
e1 = [1,0,0, 0,0,0, 1,0,0]
e2 = [0,1,0, 0,1,0, 0,1,0]
e3 = [0,0,1, 0,0,0, 0,0,1]

#sliver plus another edge is vert-measurement
v1 = [1,0,1, 0,0,0, 0,0,0]
v2 = [0,0,0, 1,1,1, 0,0,0]
v3 = [0,0,0, 0,0,0, 1,0,1]

import numpy as np
A = np.array([s1, s2, c0, c1, e1, e2, e3, v1, v2, v3])
I =  np.linalg.pinv(A)
I[I<1e-6] = 0
##print I*24
##quit()


#slivers; off diagonals after elimination of interaction 2-term
s1 = [0,0, 0,0,0, 1,0]
s3 = [0,1, 0,0,0, 0,0]
#sliver plus another vert is edge-measurement
e1 = [0,0, 1,0,0, 1,0]
e2 = [1,0, 0,1,0, 0,1]
e3 = [0,1, 0,0,1, 0,0]
#sliver plus another edge is vert-measurement
v1 = [1,1, 0,0,0, 0,0]
v2 = [0,0, 1,1,1, 0,0]
v3 = [0,0, 0,0,0, 1,1]



#v1e1 = 0
#v1e2 = v1 - s3
#v1e3 = s3

#v2e1 = e1 - s1
#v2e2 = v2 + s1+s3 -e1-e3
#v2e3 = e3 - s3

#v3e1 = s1
#v3e2 = v3 - s1
#v3e3 = 0

I = np.array([
[0,-1, 0,0,0, 1,0,0],
[0,+1, 0,0,0, 0,0,0],

[-1,0,  1,0, 0, 0,0,0],
[ 1,1, -1,0,-1, 0,1,0],
[0,-1,  0,0, 1, 0,0,0],

[+1,0, 0,0,0, 0,0,0],
[-1,0, 0,0,0, 0,0,1],
])


A = np.array([s1, s3, e1, e2, e3, v1, v2, v3])

quit()

I =  np.linalg.pinv(A.astype(np.float))
I[I<1e-6] = 0
print I*6
