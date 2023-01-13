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



from hcrsimpy.models import CircadianModel
import numpy as np
from scipy.signal import find_peaks

class Forger99Model(CircadianModel):
    """ A simple python implementation of the two population human model from Hannay et al 2019"""

    def __init__(self, params: dict= None):
        """
        Create a Forger VDP model 

        This will create a model with the default parameter values as given in Hannay et al 2019.

        This class can be used to simulate and plot the results of the given light schedule on the circadian phase
        and amplitude.
        """

        # Set the parameters to the published values by default
        self._default_params()
        
        if params:
            self.set_parameters(params)
        
        
    def _default_params(self):
        """
            Use the default parameters as defined in Hannay et al 2019
        """
        
        default_params = {'taux': 24.2,
                          'mu': 0.23,
                          'G': 33.75, 
                          'alpha_0': 0.05, 
                          'delta': 0.0075,
                          'p': 0.50, 
                          'I0': 9500.0, 
                          'kparam': 0.55}

        self.set_parameters(default_params)
        
    def set_parameters(self, param_dict: dict):
        """
            Update the model parameters using a passed in parameter dictionary. Any parameters not included
            in the dictionary will be set to the default values.

            updateParameters(param_dict)

            Returns null, changes the parameters stored in the class instance
        """
        

        params = [
            'taux',
            'mu'
            'G',
            'alpha_0',
            'delta',
            'p',
            'I0', 
            'kparam']

        for key, value in param_dict.items():
            setattr(self, key, value)
            
    def get_parameters_array(self):
        """
            Return a numpy array of the models current parameters
        """
        return np.array([self.taux, 
                         self.mu, 
                         self.G, 
                         self.alpha_0, 
                         self.delta, 
                         self.p, 
                         self.I0, 
                         self.kparam ])


    def get_parameters(self):
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

    def alpha0(self, light: float):
        """A helper function for modeling the light input processing"""
        return (self.alpha_0 * pow((light / self.I0), self.p))

    def derv(self, 
             y: np.ndarray, 
             light: float):
        """
        This defines the ode system for the forger 99 model
        returns dydt numpy array.
        """

        x = y[0]
        xc = y[1]
        n = y[2]

        Bhat = self.G * (1.0 - n) * self.alpha0(light=light) * \
            (1 - 0.4 * x) * (1 - 0.4 * xc)

        dydt = np.zeros(3)
        dydt[0] = np.pi / 12.0 * (xc + Bhat)
        dydt[1] = np.pi / 12.0 * (self.mu * (xc - 4.0 / 3.0 * pow(xc, 3.0)) - x * (
            pow(24.0 / (0.99669 * self.taux), 2.0) + self.kparam * Bhat))
        dydt[2] = 60.0 * (self.alpha0(light=light) * (1.0 - n) - self.delta * n)

        return (dydt)
    
    def step_rk4(self, state: np.ndarray, 
                 light_val: float, 
                 dt: float):
        """
            Return the state of the model assuming a constant light value
            for one time step and using rk4 to perform the step
        """
        k1 = self.derv(state, light=light_val)
        k2 = self.derv(state + k1 * dt / 2.0, light=light_val)
        k3 = self.derv(state + k2 * dt / 2.0, light=light_val)
        k4 = self.derv(state + k3 * dt, light=light_val)
        state = state + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        return state

    def integrate_model(self,
                        ts: np.ndarray,
                        light_est: np.ndarray,
                        state: np.ndarray):
        """
            Integrate the two population model forward in 
            time using the given light estimate vector using RK4
        """
        n = len(ts)
        sol = np.zeros((3,n))
        sol[:,0] = state
        for idx in range(1,n):
            state = self.step_rk4(state = state, 
                                        light_val=light_est[idx], 
                                        dt = ts[idx]-ts[idx-1])
            sol[:,idx] = state
        return sol 
    
    def integrate_observer(self, ts: np.ndarray, 
                           light_est: np.ndarray, 
                           u0: np.ndarray = None, 
                           observer= -7.0):
        """
            Integrate the two population model forward in time using the given light estimate vector
            Returns the times that the observer crosses zero, defaults to DLMO
        """
        sol = self.integrate_model(ts, light_est, u0)
        cbt_mins = find_peaks(-1*sol[0,:])[0] # min of x is the CBTmin
        #zero_crossings = np.where(np.diff(np.sign(observer(0.0, sol))))[0]
        return ts[cbt_mins] + observer
        
    def DLMOObs(t, state):
        phi = Forger99Model.phase(state)
        return np.sin(0.5*(phi-5*np.pi/12.0))

    def CBTObs(t, state):
        return np.sin(0.5*(state[2]-np.pi))

    def amplitude(state):
        # Make this joint amplitude at some point 
        return np.sqrt(state[0]**2+state[1]**2)

    def phase(state):
        x= state[0] 
        y = state[1]*-1.0
        return np.angle(x + complex(0,1)*y)
        return np.arctan(-1*state[1],state[0])
        #return np.arctan2(-1*state[1],state[0])
        
    
    @staticmethod
    def guessICDataForger99(time_zero, length=150):
        """Guess the Initial conditions for the model using the persons light schedule
        Need to add a check to see if the system is entrained at all
        """
        pass


if __name__ == "__main__":
    model = Forger99Model()
    


