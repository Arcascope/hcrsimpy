""" 
    Some functions to help in processing sleep data 
"""
# %%
import numpy as np
import pylab as plt
from scipy.integrate._ivp.ivp import solve_ivp
from scipy.optimize import minimize
from .utils import convert_binary
#import pyomo.environ as pyo


class TwoProcessModel(object):

    def __init__(self, TimeTotal: np.ndarray, R: np.ndarray, Psi: np.ndarray, Steps: np.ndarray):

        self.StepsFunc = lambda t: np.interp(t, TimeTotal, Steps)
        self.PhaseFunc = lambda t:  np.interp(t, TimeTotal, Psi)
        self.AmplitudeFunc =  lambda t: np.interp(t, TimeTotal, R)
        self.Steps = Steps
        self.TimeTotal = TimeTotal
        self.steps_wake_threshold = 10.0
        self.awake = True

    def check_wake_status(self, awake, h, c):

        H_minus = 0.17
        H_plus = 0.6
        homeostat_a = 0.10

        upper = (H_plus + homeostat_a * c)
        lower = (H_minus + homeostat_a * c)
        above_threshold = h > upper
        below_threshold = h <= lower

        if above_threshold:
            return False
        else:
            if below_threshold:
                return True
            else:
                return awake

    def dhomeostat(self, t, u):

        h = u[0]
        mu_s, tau_1, tau_s = (1.0, 18.2, 4.2)
        self.awake = self.check_wake_status(
            self.awake, h, 1.0*np.cos(self.PhaseFunc(t)))
        steps_wake = self.StepsFunc(t) > self.steps_wake_threshold or self.awake

        dh = np.zeros(1)
        if steps_wake:
            dh[0] = (mu_s - h) / tau_1
        else:
            dh[0] = -h / tau_s
        return dh

    def __call__(self, initial_value: float = 0.50):

        sol = solve_ivp(self.dhomeostat,
                        (self.TimeTotal[0], self.TimeTotal[-1]), 
                        [initial_value],
                        t_eval = self.TimeTotal)
        return(sol.y[0,:])


def sleep_midpoint(TimeTotal: np.ndarray, Wake: np.ndarray, durations=True):
    """
        Given a wearable data frame with a Wake column which takes 
        the values 0, 1, missing this routine will create a sleep phase 
        column which is based on constant phase accumulation between sleep 
        midpoints. 

        The sleep midpoints are found using the criteria than they the median 
        time where 

    """

    sleep_start = []
    sleep_end = []
    awake = Wake[0] > 0.50

    if not awake:
        sleep_start.append(TimeTotal[1])

    for k in range(1, len(Wake)):
        if (Wake[k] > 0.50 and not awake):
            awake = True
            sleep_end.append(TimeTotal[k])

        if (Wake[k] <= 0.50 and awake):
            awake = False
            sleep_start.append(TimeTotal[k])

    if Wake[-1] <= 0.50:
        sleep_end.append(TimeTotal[-1])

    assert len(sleep_start) == len(sleep_end)
    sleep_midpoints = []
    sleep_durations = []
    for (s1, s2) in zip(sleep_start, sleep_end):
        sleep_midpoints += [(s2-s1)/2+s1]
        sleep_durations += [s2-s1]

    if durations:
        return np.array(sleep_midpoints), np.array(sleep_durations)
    else:
        return np.array(sleep_midpoints)


def cluster_sleep_periods_scipy(wake_data: np.ndarray, 
                                epsilon: float,
                                makeplot: bool = False,
                                max_sleep_clusters=None, 
                                min_sleep_clusters=None):
    """
        Given a binary vector wake_data which gives a prediction for the sleep/wake  
        status and a regularization penalty ε this function will create smoothed 
        sleep-wake periods. This helps as preprocessing to remove erroneous short sleep 
        periods (and wake) which may mess up calculations like the sleep midpoint for 
        the day

        cluster_sleep_periods(wake_data : np.ndarray, epsilon: float, makeplot: bool=False):
    """

    np.nan_to_num(wake_data, 0.50)

    def objective(w):
        return sum(w * (1 - wake_data)) + sum((1 - w)*wake_data) + epsilon*sum((w[1:]-w[0:-1])**2)

    max_clusters = max_sleep_clusters or len(wake_data)

    def constraint1(x):
        return max_clusters-sum((x[1:]-x[0:-1])**2)  # geq 0

    min_clusters = min_sleep_clusters or 0

    print(
        f"The max clusters are {max_clusters} and the min clusters are {min_clusters}")

    def constraint2(x):
        return sum((x[1:]-x[0:-1])**2)-min_clusters  # geq 0

    bnds = (0.0, 1.0)
    all_bnds = [bnds for b in range(len(wake_data))]

    constraint1d = {'type': 'ineq', 'fun': constraint1}
    constraint2d = {'type': 'ineq', 'fun': constraint2}
    all_cons = [constraint1d, constraint2d]

    x0 = wake_data
    sol = minimize(objective, x0, method='SLSQP', bounds=all_bnds)

    if makeplot:
        pl = plt.scatter(range(len(wake_data)), wake_data + 0.1 *
                         np.random.randn(len(wake_data)), label="", color="blue")
        plt.plot(range(len(wake_data)), convert_binary(
            sol.x), lw=2.0, label="", color="red")
        plt.show()

    print(
        f"The max clusters are {max_clusters} takes value {constraint1(sol.x)}>=0.0")
    print(
        f"The min clusters are {min_clusters} and takes the value {constraint2(sol.x)}>=0.0")
    return(convert_binary(sol.x))


def cluster_sleep_periods(wake_data: np.ndarray,
                          epsilon: float = 30.0,
                          makeplot: bool = False,
                          max_sleep_clusters=None,
                          min_sleep_clusters=None):
    """
        Given a binary vector wake_data which gives a prediction for the sleep/wake  
        status and a regularization penalty ε this function will create smoothed 
        sleep-wake periods. This helps as preprocessing to remove erroneous short sleep 
        periods (and wake) which may mess up calculations like the sleep midpoint for 
        the day

        cluster_sleep_periods(wake_data : np.ndarray, epsilon: float, makeplot: bool=False):
    """

    N = len(wake_data)
    #wake_data = convert_binary(np.array(wake_data))

    # Model
    opt_model = pyo.ConcreteModel()
    wake_idx = range(N)
    opt_model.sleep_var = pyo.Var(wake_idx, initialize=0.50, bounds=(0, 1))

    # Constraints....
    max_clusters = max_sleep_clusters or len(wake_data)
    min_clusters = min_sleep_clusters or 0

    opt_model.sw = pyo.Constraint(rule=(min_clusters,
                                  sum([abs((opt_model.sleep_var[j+1]-opt_model.sleep_var[j]))
                                       for j in range(N-1)]),
                                  max_clusters))

    def obj_func(model):
        match_part = sum(model.sleep_var[j]*(1.0 - wake_data[j]) +
                         wake_data[j]*(1.0 - model.sleep_var[j]) for j in wake_idx)
        reg_part = epsilon * \
            sum([(model.sleep_var[j+1]-model.sleep_var[j])**2 for j in range(N-1)])
        return match_part+reg_part

    opt_model.obj = pyo.Objective(expr=obj_func(opt_model))

    opt = pyo.SolverFactory('ipopt')
    opt.solve(opt_model, tee=True)
    sv = np.array([pyo.value(opt_model.sleep_var[j]) for j in range(N)])

    if makeplot:
        pl = plt.scatter(range(N), wake_data + 0.1 *
                         np.random.randn(N), label="", color="blue")
        plt.plot(range(len(wake_data)), convert_binary(
            sv), lw=2.0, label="", color="red")
        plt.show()

    return convert_binary(sv)



