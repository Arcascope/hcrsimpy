
import numpy as np 
from ..models import SinglePopModel
from ..utils import phase_ic_guess


def compactness(awObj, gamma: float = 0.0, 
                multiplier: float = 1.0, 
                num_days: float = 4.5):
    
    spmodel = SinglePopModel({'K': 0.0, 'gamma': gamma})
    psi0 = phase_ic_guess(awObj.time_total[0])

    idx = awObj.time_total < awObj.time_total[0]+24*num_days
    sol = spmodel.integrate_model(awObj.time_total[idx], multiplier*awObj.steps[idx], np.array([0.10, psi0, 0.0]))
    return awObj.time_total[idx], sol

def compactness_trajectory(awObj, gamma: float = 0.0, multiplier: float = 1.0, num_days: float = 4.5):
    
    spmodel = SinglePopModel({'K': 0.0, 'gamma': gamma})
    compactness_trajectory = []
    time_trajectory = []
    timeStart = awObj.time_total[0]
    while timeStart < awObj.time_total[-1] - 24*num_days:
        try:
            psi0 = phase_ic_guess(timeStart)
            idxStart = ( awObj.time_total > timeStart) 
            tsFilter = awObj.time_total[idxStart] 
            idx = tsFilter < np.array(tsFilter[0])+24*num_days
            stepsFilter = awObj.steps[idxStart] 
            sol = spmodel.integrate_model(tsFilter[idx], multiplier*stepsFilter[idx], np.array([0.10, psi0, 0.0]))
            if sol[0,-1] > 0.0:
                compactness_trajectory.append(sol[0,-1])
                time_trajectory.append(timeStart)
        except:
            print("Error in trajectory")
        timeStart += 1.0
    return time_trajectory, compactness_trajectory
    
if __name__ == "__main__":
    pass