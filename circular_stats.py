#Define some functions needed for circular statistics

import numpy as np
import scipy as sp
from math import *
import pylab as plt


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
    c1-c2
    """

    return(np.angle(np.exp(complex(0,1)*(c1-c2))))


def subtract_clock_times(c1, c2):
    """Find the hour differences between two clock times new"""
    a1=sp.pi/12.0*c1
    a2=sp.pi/12.0*c2
    adiff=angle_difference(a1, a2)
    return(12.0/sp.pi*adiff)


def circular_av_clock(series):
    """Find the average time given a list of clock times"""
    angles=sp.pi/12.0*series
    ans_angle=circular_mean(angles)
    #back to clock time
    return(ans_angle*12.0/sp.pi)
    

def circular_scatter(ax, angles, clock_times=False, radius=1.0, color='blue'):
    """Adds a polar scatter plot of clock times to an axes with polar axis i.e.
        ax = plt.subplot(111, polar=True)
        Will also plot the circular mean angle and the phase coherence
    """
    
    angles=np.array(angles)
    radii=radius*np.ones(len(angles))


    if clock_times:
        angles=angles*sp.pi/12.0

    ax.scatter(angles, radii, color=color)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids(range(0,360,45), range(0,24,3))
    ax.set_rmax(1.1)
    ax.annotate("", xytext=(0.0,0.0), xy=(circular_mean(angles),phase_coherence(angles)),arrowprops=dict(facecolor=color))
    
    
