
import numpy as np
import scipy as sp
import pylab as plt

def threeMelPulse(t, timePulse=96.0):
    """
    Make a single melatonon pulse at the given time, uses the same pulse as the fitting program
    """
    value=sp.cos(sp.pi/24.0*t-sp.pi*timePulse/24.0)**30
    envelope=0.5*sp.tanh(10*(t-timePulse+5))-0.5*sp.tanh(10*(t-timePulse-48-5))
    return(value*envelope)



if __name__=='__main__':

    x=np.linspace(0,110,1000)
    myMel=lambda t: threeMelPulse(t, timePulse=20.0)
    y=map(myMel,x)
    plt.plot(x,y)
    plt.show()
