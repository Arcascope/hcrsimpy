
import numpy as np
import scipy as sp
import pylab as plt

def threeMelPulse(t, timePulse=96.0):
    """
    Make a single melatonon pulse at the given time, uses a normal/gaussian distribution check this
    at some point to make sure it matches fitting pulse
    """
    value=1.0/np.sqrt(2*sp.pi*0.20**2)*np.exp(-1*(t-timePulse)**2/(2.0*0.2**2))
    value+=1.0/np.sqrt(2*sp.pi*0.20**2)*np.exp(-1*(t-timePulse-24.0)**2/(2.0*0.2**2))
    value+=1.0/np.sqrt(2*sp.pi*0.20**2)*np.exp(-1*(t-timePulse-48.0)**2/(2.0*0.2**2))
    return(0.90770*value)




if __name__=='__main__':

    x=np.linspace(0,100,1000)
    myMel=lambda t: singleMelPulse(t, timePulse=50.0)
    y=map(myMel,x)
    plt.plot(x,y)
    plt.show()
    
