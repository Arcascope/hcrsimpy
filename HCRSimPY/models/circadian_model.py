

import scipy as sp
from abc import ABC, abstractmethod
import numpy as np


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
    def integrate_model(self, ts: np.ndarray, steps: np.ndarray, u0: np.ndarray):
        """
        Integrate the model using RK4 method
        """
        pass

    @staticmethod
    def step_rk4(self, state: np.ndarray, light_val: float, dt=0.10):
        """
            Return the state of the model assuming a constant light value
            for one time step and using rk4 to perform the step
        """
        pass

    @staticmethod
    def DLMOObs(t: float, state: np.ndarray):
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

