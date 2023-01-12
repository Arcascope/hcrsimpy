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
from .circadian_model import CircadianModel
from ctypes import c_void_p, c_double, c_int, cdll
import scipy as sp
from abc import ABC, abstractmethod
import numpy as np
import pylab as plt
from pathlib import Path
import sys
import fastclock 

# hack to not have to write an interface, just load the library 
fastclock = cdll.LoadLibrary(fastclock.__file__)

class SinglePopModel(CircadianModel):
    """
        A simple python program to integrate the human circadian rhythms model 
        (Hannay et al 2019) for a given light schedule
    """

    def __init__(self, params: dict = None):
        """
            Create a single population model by passing in a Light Function as a function of time.

            This will create a model with the default parameter values as given in Hannay et al 2019.

            This class can be used to simulate and plot the results of the given light schedule on the circadian phase
            and amplitude.
        """
        self._default_params()
        if params:
            self.set_parameters(params)

    def _default_params(self):
        """
            Use the default parameters as defined in Hannay et al 2019
        """
        default_params = {'tau': 23.84, 'K': 0.06358, 'gamma': 0.024,
                          'Beta1': -0.09318, 'A1': 0.3855, 'A2': 0.1977,
                          'BetaL1': -0.0026, 'BetaL2': -0.957756, 'sigma': 0.0400692,
                          'G': 33.75, 'alpha_0': 0.05, 'delta': 0.0075,
                          'p': 1.5, 'I0': 9325.0}

        self.set_parameters(default_params)

    def set_parameters(self, param_dict: dict):
        """
            Update the model parameters using a passed in parameter dictionary. Any parameters not included
            in the dictionary will be set to the default values.

            updateParameters(param_dict)

            Returns null, changes the parameters stored in the class instance
        """

        params = ['tau', 'K', 'gamma', 'Beta1', 'A1', 'A2', 'BetaL1',
                  'BetaL2', 'sigma', 'G', 'alpha_0', 'delta', 'p', 'I0']

        for key, value in param_dict.items():
            setattr(self, key, value)

    def get_parameters(self):
        """
            Get a dictionary of the current parameters being used by the model object.

                get_parameters()

            returns a dict of parameters
        """

        current_params = {'tau': self.tau, 'K': self.K, 'gamma': self.gamma,
                          'Beta1': self.Beta1, 'A1': self.A1, 'A2': self.A2,
                          'BetaL1': self.BetaL1,
                          'BetaL2': self.BetaL2, 'sigma': self.sigma,
                          'G': self.G, 'alpha_0': self.alpha_0,
                          'delta': self.delta, 'p': self.p, 'I0': self.I0}

        return(current_params)

    def get_parameters_array(self):
        """
            Return a numpy array of the models current parameters
        """
        return np.array([self.tau, self.K, self.gamma, self.Beta1, self.A1, self.A2, self.BetaL1, self.BetaL2, self.sigma, self.G, self.alpha_0, self.delta, self.I0, self.p])

    def integrate_model(self,
                        ts: np.ndarray,
                        light_est: np.ndarray,
                        state: np.ndarray = None):
        """
            Integrate the spmodel forward in time using the given light estimate vector
        """
        params = self.get_parameters_array()
        n = len(ts)
        sol = np.zeros((n, 3))
        fastclock.integrate_spmodel_save(
            c_void_p(sol.ctypes.data),
            c_void_p(state.ctypes.data),
            c_void_p(ts.ctypes.data),
            c_void_p(light_est.ctypes.data),
            c_void_p(params.ctypes.data),
            c_int(n))
        return(np.transpose(sol))
        

    def integrate_observer(self, ts: np.ndarray, light_est: np.ndarray, u0: np.ndarray = None, observer=None):
        """
            Integrate the spmodel forward in time using the given light estimate vector
        """
        if observer is None:
            observer = SinglePopModel.DLMOObs
        sol = self.integrate_model(ts, light_est, u0)
        zero_crossings = np.where(np.diff(np.sign(observer(0.0, sol))))[0]
        return ts[zero_crossings]

    def DLMOObs(t, state):
        return np.sin(0.5*(state[1]-5*np.pi/12.0))

    def CBTObs(t, state):
        return np.sin(0.5*(state[1]-np.pi))

    def amplitude(state):
        return(state[0])

    def phase(state):
        return(state[1])




