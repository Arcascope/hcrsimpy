"""
This file defines the two population model from the paper:

Hannay, K. M., Booth, V., & Forger, D. B. (2019). Macroscopic Models for Human Circadian Rhythms.
Journal of Biological Rhythms, 34(6), 658–671. https://doi.org/10.1177/0748730419878298

Here are some marker states for that model (all equals should be read as approx)

CBT=DLMO+7hrs
CBT=DLMO_mid+2hrs
CBT=circadian phase pi in the model
DLMO=circadian phase 5pi/12=1.309 in the model
MelatoninOffset=DLMO+10hrs

Note, this model rotates counterclockwise in accordance with the usual convention
in mathematics. The VDP family of models rotate clockwise. This can be confusing when trying
to compare phase plane plots between the models, but is fixable with a simple rotation.

An effort will be made to have the core methods align between all of the models implemented
in this package.

"""

from HCRSimPY.light import *


class TwoPopModel(object):
    """ A simple python implementation of the two population human model from Hannay et al 2019"""

    def __init__(self, LightFun):
        """
        Create a two population model by passing in a Light Function as a function of time.

        This will create a model with the default parameter values as given in Hannay et al 2019.

        This class can be used to simulate and plot the results of the given light schedule on the circadian phase
        and amplitude.
        """

        # Set the parameters to the published values by default
        self.setParameters()
        self.Light = LightFun

    def setParameters(self):
        """
        Sets the model parameters to the default published values. These are stored as
        attributes of the class.

        set Parameters()

        No return value
        """

        self.tauV = 24.25
        self.tauD = 24.0
        self.Kvv = 0.05
        self.Kdd = 0.04
        self.Kvd = 0.05
        self.Kdv = 0.01
        self.gamma = 0.024
        self.A1 = 0.440068
        self.A2 = 0.159136
        self.BetaL = 0.06452
        self.BetaL2 = -1.38935
        self.sigma = 0.0477375
        self.G = 33.75
        self.alpha_0 = 0.05
        self.delta = 0.0075
        self.p = 1.5
        self.I0 = 9325.0

    def updateParameters(self, paramDict):
        """
        Update the model parameters using a passed in parameter dictionary. Any parameters not included
        in the dictionary will be set to the default values.

        updateParameters(paramDict)

        Returns null, changes the parameters stored in the class instance

        """
        # unused list "params"
        params = [
            'tauV',
            'tauD',
            'Kvv',
            'Kdd',
            "Kvd",
            "Kdv",
            'gamma',
            'A1',
            'A2',
            'BetaL',
            'BetaL2',
            'sigma',
            'G',
            'alpha_0',
            'delta',
            'p',
            'I0']

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
            'tauV': self.w0,
            'tauD': self.tauD,
            'Kvv': self.Kvv,
            'Kdd': self.Kdd,
            'Kdv': self.Kdv,
            'Kvd': self.Kdv,
            'gamma': self.gamma,
            'A1': self.A1,
            'A2': self.A2,
            'BetaL': self.BetaL,
            'BetaL2': self.BetaL2,
            'sigma': self.sigma,
            'G': self.G,
            'alpha_0': self.alpha_0,
            'delta': self.delta,
            'p': self.p,
            'I0': self.I0}

        return (current_params)

    def alpha0(self, t):
        """A helper function for modeling the light input processing"""
        return (self.alpha_0 * pow(self.Light(t), self.p) /
                (pow(self.Light(t), self.p) + self.I0))

    def derv(self, t, y):
        """
        This defines the ode system for the single population model.

        derv(self,t,y)


        returns dydt numpy array.

        """

        Rv = y[0]
        Rd = y[1]
        Psiv = y[2]
        Psid = y[3]
        n = y[4]

        Bhat = self.G * (1.0 - n) * self.alpha0(t)

        LightAmp = self.A1 * 0.5 * Bhat * (1.0 - pow(Rv, 4.0)) * cos(Psiv + self.BetaL) + self.A2 * 0.5 * Bhat * Rv * (
            1.0 - pow(Rv, 8.0)) * cos(2.0 * Psiv + self.BetaL2)
        LightPhase = self.sigma * Bhat - self.A1 * Bhat * 0.5 * (pow(Rv, 3.0) + 1.0 / Rv) * sp.sin(
            Psiv + self.BetaL) - self.A2 * Bhat * 0.5 * (1.0 + pow(Rv, 8.0)) * sp.sin(2.0 * Psiv + self.BetaL2)

        dydt = np.zeros(5)

        dydt[0] = -self.gamma * Rv + self.Kvv / 2.0 * Rv * (1 - pow(Rv, 4.0)) + self.Kdv / 2.0 * Rd * (
            1 - pow(Rv, 4.0)) * cos(Psid - Psiv) + LightAmp
        dydt[1] = -self.gamma * Rd + self.Kdd / 2.0 * Rd * \
            (1 - pow(Rd, 4.0)) + self.Kvd / 2.0 * Rv * (1.0 - pow(Rd, 4.0)) * cos(Psid - Psiv)
        dydt[2] = 2.0 * sp.pi / self.tauV + self.Kdv / 2.0 * Rd * \
            (pow(Rv, 3.0) + 1.0 / Rv) * sp.sin(Psid - Psiv) + LightPhase
        dydt[3] = 2.0 * sp.pi / self.tauD - self.Kvd / 2.0 * \
            Rv * (pow(Rd, 3.0) + 1.0 / Rd) * sp.sin(Psid - Psiv)
        dydt[4] = 60.0 * (self.alpha0(t) * (1.0 - n) - self.delta * n)
        return (dydt)

    def integrateModel(self, tend, initial=[1.0, 1.0, 0.0, 0.0, 0.0]):
        """ Integrate the model forward in time.

        integrateModel(tend, initial=[1.0,0.0, 0.0])

        tend: float giving the final time to integrate to.
        initial: initial dynamical state Rv,Rd, PsiV, PsiD,n

        Writes the integration results into the scipy array self.results.

        Returns the circadian phas of the ventral SCN (in hours) at the ending time for the system.

        """
        dt = 0.1
        self.ts = np.arange(0.0, tend + dt, dt)
        # start the initial phase between 0 and 2pi
        initial[2] = fmod(initial[2], 2 * sp.pi)
        # start the initial phase between 0 and 2pi
        initial[3] = fmod(initial[3], 2 * sp.pi)

        r = sp.integrate.solve_ivp(
            self.derv, (0, tend), initial, t_eval=self.ts, method='Radau')
        self.results = np.transpose(r.y)

        # angle at the lights on period
        ent_angle = fmod(self.results[-1, 2], 2 * sp.pi) * 24.0 / (2.0 * sp.pi)
        return (ent_angle)

    def integrateModelData(self, timespan, initial, dt=0.1):
        """ Integrate the model using a light function defined by data
        integrateModelData(timespan, initial, dt=0.1)

        The timespan is a tuple of the start and end times (0.0,10.0)
        The initial are the initial conditions for the dynamical system
        The dt tells scipy how often to save the dynamical state of the system.

        Writes the results into the numpy array self.results.

        """
        dt = 0.01
        self.ts = np.arange(timespan[0], timespan[1], dt)
        # start the initial phase between 0 and 2pi
        initial[2] = fmod(initial[2], 2 * sp.pi)
        # start the initial phase between 0 and 2pi
        initial[3] = fmod(initial[3], 2 * sp.pi)
        r = sp.integrate.solve_ivp(
            self.derv, (timespan[0], timespan[-1]), initial, t_eval=self.ts, method='Radau')

        self.results = np.transpose(r.y)

    def integrateTransients(self, numdays=50):
        """
        Integrate the model for numdays days to get rid of any transients,
        returns the endpoint to be used as initial conditions.

        integrateTransients(numdays=50)

        Returns a numpy array giving the end state for the model


        """

        tend = numdays * 24.0  # need to change this back to 500
        r = sp.integrate.solve_ivp(
            self.derv, (0, tend), [
                0.7, 0.7, 0.0, 0.0, 0.0], t_eval=[tend], method='Radau')
        results_trans = np.transpose(r.y)
        return (results_trans[-1, :])

    def getTS(self):
        """
        Return a time series data frame for the system. Has a very, very simple melatonin state prediction which is off 
        by default

        getTS()

        returns a pandas data frame with the Time, Light_Level in lux, Phase
        ventral (radians), Rv (amplitude), n (light activation variable) and phase
        difference theta θ=ψv-ψd
        as columns
        """

        light_ts = list(map(self.Light, self.ts))
        theta = self.results[:, 2] - self.results[:, 1]
        Rv = self.results[:, 0]
        Rd = self.results[:, 1]
        ts = pd.DataFrame({'Time': self.ts,
                           'Light_Level': light_ts,
                           'Phase': self.results[:,
                                                 2],
                           'R': self.results[:,
                                             0],
                           'n': self.results[:,
                                             4],
                           'theta': self.results[:,
                                                 2] - self.results[:,
                                                                   3]})
        return (ts)


def guessICDataTwoPop(LightFunc, time_zero, length=150):
    """Guess the Initial conditions for the model using the persons light schedule
    Need to add a check to see if the system is entrained at all
    """

    a = TwoPopModel(LightFunc)
    # make a rough guess as to the initial phase
    init = [0.7, 0.7, fmod(time_zero / 24.0 * 2 * sp.pi + sp.pi, 2 * sp.pi),
            fmod(time_zero / 24.0 * 2 * sp.pi + sp.pi, 2 * sp.pi), 0.01]

    a.integrateModel(int(length) * 24.0, initial=init)
    init = a.results[-1, :]
    a.integrateModel(48.0, initial=init)

    limit_cycle = a.results
    def timeDay(x): return fmod(x, 48.0)
    lc_ts = np.array(list(map(timeDay, a.ts)))

    idx = np.searchsorted(lc_ts, time_zero) - 1
    initial = limit_cycle[idx, :]
    initial[2] = fmod(initial[2], 2 * sp.pi)
    initial[3] = fmod(initial[3], 2 * sp.pi)
    # print time_zero, initial
    return (initial)
