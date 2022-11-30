
import scipy as sp
from abc import ABC, abstractmethod
import numpy as np
import pylab as plt
from pathlib import Path


class CircadianModel(ABC):

    def __init__(self, params: dict = None):
        pass

    @abstractmethod
    def _default_params(self):
        """
            Defines the default parameters for the model
        """
        pass

    @abstractmethod
    def integrate_model(self, 
                        ts: np.ndarray, 
                        steps: np.ndarray, 
                        u0: np.ndarray):
        """
        Integrate the model using RK4 method
        """
        pass

    """ @staticmethod
    def step_rk4(self, state: np.ndarray, light_val: float, dt=0.10):
        
            Return the state of the model assuming a constant light value
            for one time step and using rk4 to perform the step
    
        pass """
        
    @abstractmethod 
    def derv(self, y: np.ndarray, t: float):
        """  
            The RHS of the equation for the model
        """
        pass

    @staticmethod
    def DLMOObs(t: float, 
                state: np.ndarray):
        """
            Function which is zero at the DLMO marker for the model 
        """
        pass

    @staticmethod
    def CBTObs(t: float, state: np.ndarray):
        """
            Function which returns zero at the CBT minimum of the model 
        """
        pass

    @staticmethod
    def amplitude(state: np.ndarray):
        """
            Gives the amplitude of the model at a given state
        """
        pass

    @staticmethod
    def phase(state: np.ndarray):
        """
            Gives the phase of the model at a given state
        """
        pass

    def __call__(self, ts: np.array, steps: np.array,
                 initial: np.ndarray, dt=0.10, *args, **kwargs):
        self.integrate_model(ts, steps, initial, *args, **kwargs)


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
            Load the model parameters, if useFile is False this will search the local directory for a optimalParams.dat file.

            setParameters()

            No return value
        """
        default_params = {'tau': 23.84, 'K': 0.06358, 'gamma': 0.024,
                          'Beta1': -0.09318, 'A1': 0.3855, 'A2': 0.1977,
                          'BetaL1': -0.0026, 'BetaL2': -0.957756,
                          'sigma': 0.0400692,
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

        for key, value in param_dict.items():
            setattr(self, key, value)

    def get_parameters(self):
        """
            Get a dictionary of the current parameters being used 
            by the model object.

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
        return np.array([self.tau, self.K, self.gamma, self.Beta1,
                         self.A1, self.A2, self.BetaL1, self.BetaL2,
                         self.sigma, self.G, self.alpha_0, self.delta,
                         self.I0, self.p])
        
        
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

    def integrate_model(self,
                        ts: np.ndarray,
                        light_est: np.ndarray,
                        state: np.ndarray = None):
        """
            Integrate the spmodel forward in time using the given 
            light estimate vector
        """
        params = self.get_parameters_array()
   
        sol = sp.integrate.solve_ivp(
            self.derv,
            (ts[0],
             ts[-1]),
            state,
            t_eval=self.ts,
            method='Radau')

        return(np.transpose(sol))
        

    def integrate_observer(self, ts: np.ndarray, 
                           light_est: np.ndarray, 
                           u0: np.ndarray = None, 
                           observer=None):
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
