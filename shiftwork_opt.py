"""
CBT=DLMO+7hrs
CBT=DLMO_mid+2hrs
CBT=circadian phase pi in the model
DLMO=circadian phase 5pi/12=1.309 in the model

MelatoninOffset=DLMO+10hrs
"""
from __future__ import print_function



from builtins import map
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
from scipy.integrate import *
import pylab as plt
from math import *
import sys
import pandas as pd
from scipy import interpolate
import seaborn as sbn
from actogram import *
from circular_stats import *
from LightSchedule import *
from singlepop_model import *



#------------------------------------------------------------

def measureSWBadness(LightFunc, num_days=150, light_threshold=10.0):
    """
    This function finds a measurement for the badness of a given shift work
    schedule. The returned value gives the fraction of minutes where the light schedule
    gives light values above a threshold that occur during the typical circadian sleep
    time (DLMO+2 to DLMO+10). Burgess et al 2003 Behavioral Sleep Med

    The system is first run to equilbrium and then the model is simulated for some integer number of
    days. (num_days). 

    measureSWBadness(LightFunc, num_days=150, light_threshold=10.0)

    """

    a=SinglePopModel(LightFunc)
    new_init=a.integrateTransients()
    a.integrateModel(num_days*24.0, initial=new_init, dt=1.0)
    res=a.getTS()

    circadian_sleep_start=5.0*sp.pi/12.0+2*sp.pi/12.0 #DLMO+2hr
    circadian_sleep_end=circadian_sleep_start+8.0*sp.pi/12.0 # assume 8 hour window for circadian sleep

    res['Phase']=res['Phase'].apply(lambda x: fmod(x,2*sp.pi))

    resBad=res[(res.Light_Level>light_threshold) & (res.Phase>circadian_sleep_start) & (res.Phase<=circadian_sleep_end)]

    fraction_bad=resBad.shape[0]/res.shape[0]

    #print("Fraction of minutes of circ sleep lost: ", fraction_bad, resBad.shape[0])
    
    return(fraction_bad)
    


if __name__=='__main__':

    #Design some shift work schedules, which are as close as possible
    # to the classic 5/2 work day/weekend schedule
    bad_score=[]

    daysOnS=range(1,21)
    
    for dayson in daysOnS:
        daysoff=np.floor(2*dayson/5)
        if daysoff==0:
            daysoff+=1 #everybody gets some time off....
        LLight=lambda t: ShiftWorkLight(t,dayson=dayson, daysoff=daysoff)
        cycle=dayson+daysoff #how many days to complete a whole cycle
        bad_score.append((dayson, measureSWBadness(LLight, num_days=40*cycle)))
        
    x,y=zip(*bad_score)
    plt.bar(x,y, tick_label=daysOnS, color="darkgreen")
    plt.xlabel("Days worked in a row")
    plt.ylabel("Fraction minutes of sleep lost")
    plt.title("Basic Shift Work Rotations")
    plt.show()
