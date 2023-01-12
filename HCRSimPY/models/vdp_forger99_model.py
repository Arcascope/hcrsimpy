"""
This file defines the Forger 1999 model

Forger, D. B., Jewett, M. E., & Kronauer, R. E. (1999).
 A Simpler Model of the Human Circadian Pacemaker. Journal of Biological Rhythms,
 14(6), 533–538. https://doi.org/10.1177/074873099129000867

 However, it uses the parameters taken from

Serkh K, Forger DB. Optimal schedules of
light exposure for rapidly correcting circadian misalignment.
PLoS Comput Biol. 2014;10(4):e1003523. Published 2014 Apr 10. doi:10.1371/journal.pcbi.1003523

Rather than the parameters from the original paper.


"""

from HCRSimPY.light import *


class vdp_forger99_model(object):
    """Implemetation of the VDP based model Simpler Model """

    def __init__(self, LightFun):
        """
        Create a single population model by passing in a Light Function as a function of time.

        This will create a model with the default parameter values

        This class can be used to simulate and plot the results of the given light schedule on the circadian phase
        and amplitude.
        """
        self.setParameters()
        self.Light = LightFun

    def setParameters(self):
        """
        Specify the dafult model parameters

        setParameters()

        Return: None
        """

        # Set the parameters
        self.taux = 24.2
        self.mu = 0.23
        self.G = 33.75
        self.alpha_0 = 0.05
        self.delta = 0.0075
        self.p = 0.50
        self.I0 = 9500.0
        self.kparam = 0.55

    def updateParameters(self, paramDict):
        """
        Update the model parameters using a passed in parameter dictionary. Any parameters not included
        in the dictionary will be set to the default values.

        updateParameters(paramDict)

        Returns null, changes the parameters stored in the class instance
        """

        params = ['taux', 'mu', 'G', 'alpha_0', 'delta', 'p', 'I0', 'kparam']

        # Now set the parameters
        for k in paramDict.keys():
            mycode = 'self.' + k + "=paramDict[\'" + k + "\']"
            exec(mycode)

    def getParameters(self):
        """Get a dictionary of the current parameters being used by the model object.

        getParameters()

        returns a dict of parameters
        """

        current_params = {
            'taux': self.taux,
            'mu': self.mu,
            'G': self.G,
            'alpha_0': self.alpha_0,
            'delta': self.delta,
            'p': self.p,
            'I0': self.I0,
            'kparam': self.kparam}

        return (current_params)

    def alpha0(self, t):
        """A helper function for modeling the light input processing"""
        return (self.alpha_0 * pow((self.Light(t) / self.I0), self.p))

    def derv(self, t, y):
        """
        This defines the ode system for the single population model.

        derv(self,t,y)


        returns dydt numpy array.

        """
        x = y[0]
        xc = y[1]
        n = y[2]

        Bhat = self.G * (1.0 - n) * self.alpha0(t) * \
            (1 - 0.4 * x) * (1 - 0.4 * xc)

        dydt = np.zeros(3)

        dydt[0] = sp.pi / 12.0 * (xc + Bhat)
        dydt[1] = sp.pi / 12.0 * (self.mu * (xc - 4.0 / 3.0 * pow(xc, 3.0)) - x * (
            pow(24.0 / (0.99669 * self.taux), 2.0) + self.kparam * Bhat))
        dydt[2] = 60.0 * (self.alpha0(t) * (1.0 - n) - self.delta * n)

        return (dydt)

    def integrateModel(self, tend, initial=[1.0, 1.0, 0.0]):
        """ Integrate the model forward in time.

        integrateModel(tend, initial=[1.0,0.0, 0.0])

        tend: float giving the final time to integrate to.
        initial: initial dynamical state
        The parameters are tend= the end time to stop the simulation and initial=[x,xc, n]

        Writes the integration results into the scipy array self.results.

        Returns the circadian phase (in hours) at the ending time for the system.

        """

        dt = 0.1
        self.ts = np.arange(0.0, tend + dt, dt)

        r = sp.integrate.solve_ivp(
            self.derv, (0, tend), initial, t_eval=self.ts, method='Radau')  # uses RK45
        self.results = np.transpose(r.y)

        # times negative one because VDP runs clockwise versus counterclockwise
        ent_angle = 1.0 * atan2(self.results[-1, 1], self.results[-1, 0])
        if (ent_angle < 0.0):
            ent_angle += 2 * sp.pi

        ent_angle = ent_angle * 24.0 / (2.0 * sp.pi)
        return (ent_angle)

    def integrateModelData(self, timespan, initial):
        """
        Integrate the model using a light function defined by data

        integrateModelData(timespan, initial, dt=0.1)

        The timespan is a tuple of the start and end times (0.0,10.0)
        The initial are the initial conditions for the dynamical system
        The dt tells scipy how often to save the dynamical state of the system.

        Writes the results into the numpy array self.results.

        """
        dt = 0.01
        self.ts = np.arange(timespan[0], timespan[1], dt)
        r = sp.integrate.solve_ivp(
            self.derv, (timespan[0], timespan[-1]), initial, t_eval=self.ts, method='Radau')
        self.results = np.transpose(r.y)

    def integrateTransients(self, numdays=500):
        """
        Integrate the model for numdays days to get rid of any transients,
        returns the endpoint to be used as initial conditions.

        integrateTransients(numdays=50)

        Returns a numpy array giving the end state for the model
        """
        tend = numdays * 24.0

        r = sp.integrate.solve_ivp(
            self.derv, (0, tend), [
                0.7, 0.0, 0.0], t_eval=[tend], method='Radau')
        results_trans = np.transpose(r.y)

        return (results_trans[-1, :])

    def getTS(self):
        """
        Return a time series data frame for the system.

        getTS()

        The amplitude for this vdp model is defined as euclidean distance from
        the origin to (x,x_c) coordinates in phase space.

        The unwrapped phase estimate is taken as the -1*arctan of the phase
        plane coordinates. This is transformed so that it is comparible with the
        Hannay models.

        returns a pandas data frame with the Time, Light_Level in lux, Phase (radians), R (amplitude),
        n (light activation variable) as columns
        """

        light_ts = list(map(self.Light, self.ts))
        # define the amplitude as the sqrt of each coordinate squared
        Amplitude = np.sqrt(self.results[:, 0] ** 2 + self.results[:, 1] ** 2)

        # Need to extract a phase in radians
        wrappedPhase = -1.0 * \
            np.arctan2(self.results[:, 1], self.results[:, 0])

        # Make it between 0 and 2pi
        for i in range(len(wrappedPhase)):
            if wrappedPhase[i] < 0.0:
                wrappedPhase[i] += 2 * sp.pi

        Phase = np.unwrap(wrappedPhase, discont=0.0)

        ts = pd.DataFrame({'Time': self.ts,
                           'Light_Level': light_ts,
                           'Phase': Phase,
                           'R': Amplitude,
                           'n': self.results[:,
                                             2]})
        return (ts)


def guessICDataVDP(LightFunc, time_zero, length=50):
    """Guess the Initial conditions for the model using the persons light schedule"""

    a = vdp_model(LightFunc)
    # make a rough guess as to the initial phase
    init = np.array([1.0, 1.0, 0.0])

    a.integrateModel(int(length) * 24.0, initial=init)
    init = a.results[-1, :]
    a.integrateModel(48.0, initial=init)

    limit_cycle = a.results
    def timeDay(x): return fmod(x, 48.0)
    lc_ts = np.array(list(map(timeDay, a.ts)))

    idx = np.searchsorted(lc_ts, time_zero) - 1
    initial = limit_cycle[idx, :]
    # print time_zero, initial
    return (initial)


if __name__ == '__main__':
    duration = 16.0  # gets 8 hours of sleep
    intensity = 150.0
    wake = 6.0
    def LightFunReg(t): return RegularLightSimple(t, intensity, wake, duration)
# potential error with vdp_model?
# Unresolved reference 'vdp_model'
    a = vdp_model(LightFunReg)
    a.integrateModel(24 * 40)
    tsdf = a.getTS()
