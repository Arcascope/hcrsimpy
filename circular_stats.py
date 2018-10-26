#Define some functions needed for circular statistics

import numpy as np
import scipy as sp
from math import *


def circular_mean(series):
    Z=complex(0,0)
    series=np.array(series)
    for i in range(len(series)):
        Z+=np.exp(series[i]*complex(0,1))

    Z=Z/float(len(series))

    ans=np.angle(Z)
    if (ans<0.0):
        ans+=2*sp.pi
    return(ans)

def phase_coherence(series):
    Z=complex(0,0)
    series=np.array(series)
    for i in range(len(series)):
        Z+=np.exp(series[i]*complex(0,1))

    Z=Z/float(len(series))

    ans=np.absolute(Z)
    return(ans)


def angle_difference(c1, c2):
    """Find the angle between two angles given in radians
    angle_difference(c1, c2)
    c1=c2
    """

    return(np.angle(np.exp(complex(0,1)*(c1-c2))))


def subtract_clock_times(c1, c2):
    """Find the hour differences between two clock times"""
    return(fmod(c1-c2, 24.0))
    

