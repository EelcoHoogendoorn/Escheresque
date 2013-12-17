
"""
quat class for custom camera control
"""

import numpy as np
from .util import normalize

class Quaternion(object):
    """quaternoin class as used by camera"""
    def __init__(self, x,y,z, w):
        self.x, self.y, self.z, self.w = x,y,z,w

    def vector_get(self):
        return np.array((self.x,self.y,self.z))
    def vector_set(self, vec):
        self.x, self.y, self.z = vec
    vector = property(vector_get, vector_set)

    def __mul__(self, other):
        sv, ov = self.vector, other.vector
        q = Quaternion(
            *np.cross(sv, ov) + sv * other.w + ov * self.w,
            w=self.w*other.w - np.dot(sv, ov))
        q.normalize()
        return q


    def to_spherical(self):
        """return """
        RADIANS = np.pi*2
        #RADIANS = 1

        ca = self.w
        sa = np.sqrt( 1.0 - ca * ca )
        print sa
        angle = np.arccos(ca) * 2# * RADIANS

        if np.abs(sa) < 0.0005:
            sa = 1

        t = self.vector / sa

        lattitude = np.arccos(t[2])

        if t[0]**2+t[1]**2 < 0.0005:
            longtitude = 0
        else:
            longtitude = np.arctan2(t[0], t[1])# * RADIANS

        if longtitude < 0:
            longtitude += RADIANS


        scale = 180.0/np.pi

        return longtitude*scale, lattitude*scale, angle*scale


##    def to_matrix(self):
##        qx,qy,qz,qw = self.x, self.y, self.z, self.w
##        q = np.array((
##            1.0 - 2.0*qy*qy - 2.0*qz*qz, 2.0*qx*qy - 2.0*qz*qw, 2.0*qx*qz + 2.0*qy*qw, 0.0,
##    	   2.0*qx*qy + 2.0*qz*qw, 1.0 - 2.0*qx*qx - 2.0*qz*qz, 2.0*qy*qz - 2.0*qx*qw, 0.0,
##	       2.0*qx*qz - 2.0*qy*qw, 2.0*qy*qz + 2.0*qx*qw, 1.0 - 2.0*qx*qx - 2.0*qy*qy, 0.0,
##	       0.0, 0.0, 0.0, 1.0))
##        q = q.reshape((4,4))
##        return q
    def to_matrix(self):
        qx,qy,qz,qw = self.x, self.y, self.z, self.w
        q = np.array((
            1.0 - 2.0*qy*qy - 2.0*qz*qz, 2.0*qx*qy - 2.0*qz*qw, 2.0*qx*qz + 2.0*qy*qw,
    	   2.0*qx*qy + 2.0*qz*qw, 1.0 - 2.0*qx*qx - 2.0*qz*qz, 2.0*qy*qz - 2.0*qx*qw,
	       2.0*qx*qz - 2.0*qy*qw, 2.0*qy*qz + 2.0*qx*qw, 1.0 - 2.0*qx*qx - 2.0*qy*qy,
            ))
        q = q.reshape((3,3))
        return q
##
##    def to_spherical(self):
##        """
##
##        Euler angles
##            Heading = rotation about y axis
##            Attitude = rotation about z axis
##            Bank = rotation about x axis
##        Euler angle order
##            Heading applied first
##            Attitude applied second
##            Bank applied last
##
##
##        http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToEuler/index.htm
##
##        public void set(Quat4d q1) {
##        	double test = q1.x*q1.y + q1.z*q1.w;
##        	if (test > 0.499) { // singularity at north pole
##        		heading = 2 * atan2(q1.x,q1.w);
##        		attitude = Math.PI/2;
##        		bank = 0;
##        		return;
##        	}
##        	if (test < -0.499) { // singularity at south pole
##        		heading = -2 * atan2(q1.x,q1.w);
##        		attitude = - Math.PI/2;
##        		bank = 0;
##        		return;
##        	}
##            double sqx = q1.x*q1.x;
##            double sqy = q1.y*q1.y;
##            double sqz = q1.z*q1.z;
##            heading = atan2(2*q1.y*q1.w-2*q1.x*q1.z , 1 - 2*sqy - 2*sqz);
##        	attitude = asin(2*test);
##        	bank = atan2(2*q1.x*q1.w-2*q1.y*q1.z , 1 - 2*sqx - 2*sqz)
##        }
##        """
##        scale = 180.0/np.pi
##        test = self.x*self.y + self.z*self.w
####        print test
##        if test > 0.499:
##            heading = 2 * np.arctan2(self.x, self.w)
##            attitude = np.pi/2
##            bank = 0
##        elif test < -0.499:
##            heading = -2 * np.arctan2(self.x, self.w)
##            attitude = -np.pi/2
##            bank = 0
##        else:
##            v = self.vector
##            v = v * v
##            heading  = np.arctan2(2*self.y*self.w-2*self.x*self.z , 1 - 2*v[1] - 2*v[2])
##            attitude = np.arccos (2*test)
##            bank     = np.arctan2(2*self.x*self.w-2*self.y*self.z , 1 - 2*v[0] - 2*v[2])
##        return heading*scale, attitude*scale, bank*scale

    @staticmethod
    def from_axis_angle(axis, angle):
        """
        axis as integer, angle in degrees
        """

        axis = np.eye(3)[axis]
        angle = angle * np.pi / 180
        return Quaternion._from_axis_angle(axis, angle)

    @staticmethod
    def from_pair(old, new):
        """
        find minimal rotation that maps one direction unto the other
        """
        axis = normalize( np.cross(old, new))
        angle = np.arccos( np.dot(old, new) / np.linalg.norm(old) / np.linalg.norm(new))

        return Quaternion._from_axis_angle(axis, angle)


    @staticmethod
    def _from_axis_angle(axis, angle):
        """make quat from nornalized axis and angle in radians"""
        sa = np.sin(angle/2)
        ca = np.cos(angle/2)

        q = Quaternion(*(axis*sa), w=ca)
        q.normalize()
        return q



    def __repr__(self):
        return str(( self.x,self.y,self.z,self.w))


    def length_squared(self):
        return self.x**2+self.y**2+self.z**2+self.w**2
    def normalize(self):
        l = np.sqrt(self.length_squared())
        self.x /= l
        self.y /= l
        self.z /= l
        self.w /= l

    def conj(self):
        return Quaternion(*-self.vector, w=self.w)
    def inv(self):
        q = self.conj()
        l = self.length_squared()
        q.x /= l
        q.y /= l
        q.z /= l
        q.w /= l
        return q

