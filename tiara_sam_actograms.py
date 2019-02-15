import numpy as np
import scipy as sp
from scipy.integrate import *
import pylab as plt
from math import *
import sys
from sets import Set
import pandas as pd
from scipy import interpolate
from actogram import *
from circular_stats import *
from LightSchedule import *
from singlepop_model import *
from stroboscopic import *
from MelatoninSchedule import *
from test_light_schedules import makeActogram


def ProgressMeasure(shift, pulse):
    """Put in a light and melatonin schedule and get out reentrainment time"""
    duration=16.0 #gets 8 hours of sleep
    intensity=150.0
    wake=6.0
    LightFunReg=lambda t: RegularLightSimple(t,intensity,wake,duration)
    LightFunTest=lambda t: OneDayShift(t,shift=shift, pulse=2.0, wakeUp=wake)

    #Create SP Model
    a=SinglePopModel(LightFunReg)
    init=a.integrateTransients()
    b=SinglePopModel(LightFunTest)
    ent_angle=b.integrateModel(24*40, initial=init);
    tsdf=b.getTS()

    dlmo_func=sp.interpolate.interp1d(np.array(tsdf['Phase']), np.array(tsdf['Time']), bounds_error=False)

    real_days=tsdf['Time'].iloc[-1]/24.0
    num_days=ceil(tsdf['Time'].iloc[-1]/24.0)
        
    if (tsdf.Phase.iloc[0]<1.309):
        dlmo_phases=np.arange(1.309, real_days*2.0*sp.pi, 2*sp.pi) #all the dlmo phases using 1.309 as the cicadian phase of the DLMO
        dlmo_times=np.array(map(lambda x: fmod(x,24.0), np.array(map(dlmo_func, dlmo_phases))))
        dlmo_times= dlmo_times[np.isfinite(dlmo_times)]
        dayYvalsDLMO=num_days-np.arange(0.5, len(dlmo_times)+0.5, 1.0)
    else:
        dlmo_phases=np.arange(1.309+2*sp.pi, real_days*2.0*sp.pi, 2*sp.pi) #all the dlmo phases using 1.309 as the cicadian phase of the DLMO
        dlmo_times=np.array(map(lambda x: fmod(x,24.0), np.array(map(dlmo_func, dlmo_phases))))
        dlmo_times= dlmo_times[np.isfinite(dlmo_times)]
        dayYvalsDLMO=num_days-np.arange(1.5, len(dlmo_times)+1.5, 1.0)

    if (tsdf.Phase.iloc[0]<sp.pi):
        cbt_phases=np.arange(sp.pi, real_days*2.0*sp.pi, 2*sp.pi)
        cbt_times=np.array(map(lambda x: fmod(x,24.0), np.array(map(dlmo_func, cbt_phases))))
        cbt_times=cbt_times[np.isfinite(cbt_times)]
        dayYvalsCBT=num_days-(np.arange(0.5, len(cbt_times)+0.5, 1.0))
    else:
        cbt_phases=np.arange(sp.pi+2*sp.pi, real_days*2.0*sp.pi, 2*sp.pi)
        cbt_times=np.array(map(lambda x: fmod(x,24.0), np.array(map(dlmo_func, cbt_phases))))
        cbt_times=cbt_times[np.isfinite(cbt_times)]
        dayYvalsCBT=num_days-np.arange(1.5, len(cbt_times)+1.5, 1.0)

    time_before=dlmo_times[10]
    time_last=dlmo_times[-1]
    value=subtract_clock_times(time_before, time_last)
    return(value)
    
        
        

    
    



    

if __name__=='__main__':
    #OD=lambda t: OneDayShift(t,pulse=2.0)
    #makeActogram(OD)
    print ProgressMeasure(shift=8.0, pulse=2.0)

    """
Work on finding the pulse time [0.24.0] that minimimizes the absolute value/square of the ProgressMeasure function. Use a a for loop over the pulse times
    """
    results=[]
    for x in range(0,24):
        Value=ProgressMeasure(shift=8.0, pulse=x)**2
        results.append(Value)
    print results
    

    
    

