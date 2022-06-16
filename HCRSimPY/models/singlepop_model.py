"""
This file defines the single population model from the paper:

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
from __future__ import print_function

from builtins import map
from builtins import object
from builtins import range
from math import *

import numpy as np
import pandas as pd
import scipy as sp


# from plots import *
# from HCRSimPY.light_schedules import *


# Make a class to store methods related to simulating the circadian model


class SinglePopModel(object):
    """A simple python program to integrate the human circadian rhythms model (Hannay et al 2019) for a given light
    schedule"""

    def __init__(self, LightFun):
        """
        Create a single population model by passing in a Light Function as a function of time.

        This will create a model with the default parameter values as given in Hannay et al 2019.

        This class can be used to simulate and plot the results of the given light schedule on the circadian phase
        and amplitude.
        """
        # Read the parameters from a file
        self.setParameters()
        self.Light = LightFun

    def setParameters(self):
        """
        Load the model parameters, if useFile is False this will search the local directory for a optimalParams.dat file
        .

        setParameters()

        No return value

        """
        self.w0, self.K, self.gamma, self.Beta1, self.A1, self.A2, self.BetaL1, self.BetaL2, self.sigma, self.G, \
            self.alpha_0, self.delta, self.p, self.I0 = [
                0.263524, 0.06358, 0.024, -0.09318, 0.3855, 0.1977, -0.0026, -0.957756, 0.0400692, 33.75, 0.05, 0.0075,
                1.5,
                9325.0]

    def updateParameters(self, paramDict):
        """
        Update the model parameters using a passed in parameter dictionary. Any parameters not included
        in the dictionary will be set to the default values.

        updateParameters(paramDict)

        Returns null, changes the parameters stored in the class instance

        """

        params = [
            'w0',
            'K',
            'gamma',
            'Beta1',
            'A1',
            'A2',
            'BetaL1',
            'BetaL2',
            'sigma',
            'G',
            'alpha_0',
            'delta',
            'p',
            'I0']

        if 'tau' in paramDict.keys():
            paramDict['w0'] = 2 * sp.pi / paramDict['tau']

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
            'w0': self.w0,
            'K': self.K,
            'gamma': self.gamma,
            'Beta1': self.Beta1,
            'A1': self.A1,
            'A2': self.A2,
            'BetaL1': self.BetaL1,
            'BetaL2': self.BetaL2,
            'sigma': self.sigma,
            'G': self.G,
            'alpha_0': self.alpha_0,
            'delta': self.delta,
            'p': self.p,
            'I0': self.I0}

        return (current_params)

    def updatePeriod(self, newVal):
        """
        Change the period of the circadian clock, should be put in as hours

        updatePeriod(newVal)

        changes the self.w0 parameter for the model.

        """
        if (newVal >= 10.0 and newVal <= 35.0):
            self.w0 = 2 * sp.pi / newVal
        else:
            print(
                "The new circadian period should be in hours, it looks like you forgot this so the period was not "
                "updated.")

    def alpha0(self, t):
        """A helper function for modeling the light input processing"""
        assert self.Light(t) >= 0.0
        return (self.alpha_0 * pow(self.Light(t), self.p) /
                (pow(self.Light(t), self.p) + self.I0))

    def derv(self, t, y):
        """
        This defines the ode system for the single population model.

        derv(self,t,y)


        returns dydt numpy array.

        """
        R = y[0]
        Psi = y[1]
        n = y[2]

        Bhat = self.G * (1.0 - n) * self.alpha0(t)
        LightAmp = self.A1 * 0.5 * Bhat * (1.0 - pow(R, 4.0)) * sp.cos(Psi + self.BetaL1) + self.A2 * 0.5 * Bhat * R * (
            1.0 - pow(R, 8.0)) * sp.cos(2.0 * Psi + self.BetaL2)
        LightPhase = self.sigma * Bhat - self.A1 * Bhat * 0.5 * (pow(R, 3.0) + 1.0 / R) * sp.sin(
            Psi + self.BetaL1) - self.A2 * Bhat * 0.5 * (1.0 + pow(R, 8.0)) * sp.sin(2.0 * Psi + self.BetaL2)

        dydt = np.zeros(3)

        dydt[0] = -1.0 * self.gamma * R + self.K * \
            sp.cos(self.Beta1) / 2.0 * R * (1.0 - pow(R, 4.0)) + LightAmp
        dydt[1] = self.w0 + self.K / 2.0 * \
            sp.sin(self.Beta1) * (1 + pow(R, 4.0)) + LightPhase
        dydt[2] = 60.0 * (self.alpha0(t) * (1.0 - n) - self.delta * n)

        return (dydt)

    def integrateModel(self, tend, initial=[1.0, 0.0, 0.0]):
        """ Integrate the model forward in time.

        integrateModel(tend, initial=[1.0,0.0, 0.0])

        tend: float giving the final time to integrate to.
        initial: initial dynamical state
        The parameters are tend= the end time to stop the simulation and initial=[R, Psi, n]

        Writes the integration results into the scipy array self.results.

        Returns the circadian phase (in hours) at the ending time for the system.

        """
        dt = 0.1
        self.ts = np.arange(0.0, tend, dt)
        # start the initial phase between 0 and 2pi
        initial[1] = fmod(initial[1], 2 * sp.pi)

        r = sp.integrate.solve_ivp(
            self.derv, (0, tend), initial, t_eval=self.ts, method='Radau')
        self.results = np.transpose(r.y)

        # angle at the lights on period
        ent_angle = fmod(self.results[-1, 1], 2 * sp.pi) * 24.0 / (2.0 * sp.pi)
        return (ent_angle)

    def integrateModelData(self, timespan, initial, dt=0.1):
        """
        Integrate the model using a light function defined by data

        integrateModelData(timespan, initial, dt=0.1)

        The timespan is a tuple of the start and end times (0.0,10.0)
        The initial are the initial conditions for the dynamical system
        The dt tells scipy how often to save the dynamical state of the system.

        Writes the results into the numpy array self.results.

        """
        self.ts = np.arange(timespan[0], timespan[1], dt)
        # start the initial phase between 0 and 2pi
        initial[1] = fmod(initial[1], 2 * sp.pi)
        r = sp.integrate.solve_ivp(
            self.derv,
            (timespan[0],
             timespan[1]),
            initial,
            t_eval=self.ts,
            method='Radau')

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
                0.7, 0.0, 0.01], t_eval=[tend], method='Radau')
        results_trans = np.transpose(r.y)
        return (results_trans[-1, :])

    def getTS(self, addMelatonin=True):
        """
        Return a time series data frame for the system. Has a very, very simple melatonin state prediction which is off
        by default

        getTS(self, addMelatonin=True)

        returns a pandas data frame with the Time, Light_Level in lux, Phase (radians), R (amplitude), n 
        (light activation variable) 
        as columns
        """

        light_ts = list(map(self.Light, self.ts))
        ts = pd.DataFrame({'Time': self.ts,
                           'Light_Level': light_ts,
                           'Phase': self.results[:,
                                                 1],
                           'R': self.results[:,
                                             0],
                           'n': self.results[:,
                                             2]})

        if (addMelatonin):
            melatonin = []
            light_threshold = 100.0  # half max in melatonin suppression

            for i in range(self.results.shape[0]):
                phase = fmod(self.results[i, 1], 2 * sp.pi)

                if ((phase >= 1.309) and (phase <= 3.92)
                        and (light_ts[i] <= light_threshold)):
                    melatonin.append(1.0)
                else:
                    melatonin.append(0.0)

            ts['Melatonin'] = np.array(melatonin)

        return (ts)

    def findKeyTimes(self):
        """Find the mean circadian phases at different times in the data set as well as the variation.

        findKeyTimes()

        Returns a pandas data frame with the circular mean of the phase coherence at each time point

        """
        wrapped_time = np.round([fmod(x, 24.0) for x in self.ts], 2)
        df = pd.DataFrame({'Time': wrapped_time, 'Phase': self.results[:, 1]})

        # Find the circular statistics for the circadian phase data at each
        # time point
        df2 = df.groupby('Time')['Phase'].agg(
            {'Circular_Mean': circular_mean, 'Phase_Coherence': phase_coherence, 'Samples': np.size})

        mean_func = sp.interpolate.interp1d(
            np.array(
                df2['Circular_Mean']), np.array(
                df2.index))
        return ((mean_func(sp.pi), mean_func(1.309)))

    def findAveragePhase(self):
        """Find the average circadian phase for each clock time in the simulation.

        findAveragePhase()

        Returns a pandas data frame with index given by the wrapped time, the mean phase across the simulation, 
        phase coherence and number of samples

        """

        wrapped_time = np.round([fmod(x, 24.0) for x in self.ts], 2)
        df = pd.DataFrame({'Time': wrapped_time, 'Phase': self.results[:, 1]})

        df2 = df.groupby('Time')['Phase'].agg(
            {'Circular_Mean': circular_mean, 'Phase_Coherence': phase_coherence, 'Samples': np.size})

        return (df2)


def guessICData(LightFunc, time_zero, length=150, show=False):
    """
    Guess the Initial conditions for the model using the persons light schedule
    Need to add a check to see if the system is entrained at all....

    guessICData(LightFunc, time_zero, length=150, show=False)
    """

    a = SinglePopModel(LightFunc)
    # make a rough guess as to the initial phase
    init = [0.7, fmod(time_zero / 24.0 * 2 * sp.pi + sp.pi, 2 * sp.pi), 0.1]

    a.integrateModel(int(length) * 24.0, initial=init)
    init = a.results[-1, :]
    a.integrateModel(48.0, initial=init)

    limit_cycle = a.results
    def timeDay(x): return fmod(x, 48.0)
    lc_ts = np.array(list(map(timeDay, a.ts)))

    idx = np.searchsorted(lc_ts, time_zero) - 1
    initial = limit_cycle[idx, :]
    initial[1] = fmod(initial[1], 2 * sp.pi)

    if (show):
        print(("Time zero, initial ", time_zero, initial))
    return (initial)


