
"""
gray-scott RD module
"""

import numpy as np


class ReactionDiffusion(object):
    """
    rd object. run this in thread, with periodic updates
    add some UI elements to control it
    select coefs and do start stop, and so on
    """

    #Diffusion constants for u and v. Probably best not to touch these
##    ru = 0.2*1.5
##    rv = 0.1*1.5
    ru = 1
    rv = 0.5
    #Timestep size; if set too big, the simulation becomes unstable

    """
    Various interesting parameter combinations, governing the rate of synthesis
    of u and rate of breakdown of v, respectively.
    Different rate parameters yield very different results, but the majority of
    parameters combination do not yield interesting results at all.
    So if you feel like trying something yourself, start by permuting some existing settings
    """
    params = dict(
        divide_and_conquer  = (0.035, 0.099),   #dividing blobs
        aniogenesis         = (0.040, 0.099),   #a vascular-like structure
        fingerprints        = (0.032, 0.091),   #a fingerprint-like pattern
        holes               = (0.042, 0.101),
        labyrinth           = (0.045, 0.108),   #somehow this configuration does not like closed loops
        chemotaxis          = (0.051, 0.115),   #growing roots?

        #lower parameter values tend to destabilize the patterns
        unstable_blobs      = (0.024, 0.084),
        unstable_labyrinth  = (0.024, 0.079),
        unstable_holes      = (0.022, 0.072),

        #even lower parameters lead to wave-like phenomena
        swimming_medusae    = (0.011, 0.061),
        traveling_waves     = (0.019, 0.069),
        standing_waves      = (0.015, 0.055),
        trippy_chaos        = (0.025, 0.075),
    )


    def __init__(self, complex):
        self.complex = complex

        self.dt = 0.15

        #this is important for initialization! right initial conditions matter a lot
        self.state   = np.zeros((2,complex.size), np.float)
        self.state[0] = 1
        #add seeds
        self.state[1,np.random.randint(self.complex.size, size=10)] = 1
        #fix boundaries
        self.state[1] = self.complex.wrap(self.complex.boundify)(self.state[1])

        key = 'swimming_medusae'
        self.coeff = self.params[key]

        self.laplace_normalized_p0 = self.complex.wrap(self.complex.laplace_normalized_p0)

    def diffuse(self, state, mu):
##        return self.complex.diffuse(state) * (mu / -self.complex.largest * 3)
        return self.laplace_normalized_p0(state) * (-mu * 2)


    def gray_scott_derivatives(self, u, v):
        """the gray-scott equation; calculate the time derivatives, given a state (u,v)"""
        f, g = self.coeff
        reaction = u * v * v                        #reaction rate of u into v; note that the production of v is autocatalytic
        source   = f * (1 - u)                      #replenishment of u is proportional to its deviation from one
        sink     = g * v                            #decay of v is proportional to its concentration
        udt = self.diffuse(u, self.ru) - reaction + source  #time derivative of u
        vdt = self.diffuse(v, self.rv) + reaction - sink    #time derivative of v
        return udt, vdt                             #return both rates of change

    def integrate(self, derivative):
        """
        forward euler integration of the equations, giveen their state and time derivative at this point
        the state after a small timestep dt is taken to be the current state plus the time derivative times dt
        this approximation to the differential equations works well as long as dt is 'sufficiently small'
        """
        for s,d in zip(self.state, derivative):
            s += d * self.dt

    def simulate(self, iterations):
        """
        generator function to do the time integration, to be used inside the animation
        rather than computing all the image frames before starting the animation,
        the frames are computed 'on demand' by this function, returning/yielding
        the image frames one by one
        """
        #repeat 'frames' times
        for i in range(iterations):

            #make 20 timesteps per frame; we dont need to show every one of them,
            #since the change from the one to the next is barely perceptible
            for r in range(20):
                #update the chemical concentrations
                self.integrate(self.gray_scott_derivatives(*self.state))

            #every 20 iterations, yield output
            #the v field is what we yield to be plotted
            #we might as well plot u, as it visualizes the dynamics just as well
##            yield v
