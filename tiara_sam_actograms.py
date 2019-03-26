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
import multiprocessing


def ProgressMeasure(shift, pulse, plot=False):
    """Put in a light and get how many hours they are out of phase on the first day of the shift"""
    duration=16.0 #gets 8 hours of sleep
    intensity=150.0
    wake=8.0
    LightFunReg=lambda t: RegularLightSimple(t,intensity,wake,duration)
    LightFunTest=lambda t: OneDayShift(t,shift=shift, pulse=pulse, wakeUp=wake)
    JetLag=lambda t: SlamShift(t, shift)

    
    #Create SP Model
    a=SinglePopModel(LightFunReg)
    init=a.integrateTransients()
    b=SinglePopModel(LightFunTest)
    ent_angle=b.integrateModel(24*40, initial=init);
    tsdf=b.getTS()

    #Create model with no extra light pulse
    c=SinglePopModel(LightFunReg)
    init=c.integrateTransients()
    c.Light=JetLag
    ent_angle=c.integrateModel(24*40, initial=init);
    tsdfNo=c.getTS()

    if plot:
        plt.figure()
        ax=plt.gca()
        acto=actogram(ax, tsdf) #add an actogram to those axes
        acto.addMarker2(8*24.0+pulse)
        acto.addCircadianPhases(tsdfNo, col='red')
        ax.set_title("Optimal Shifts East 8.0 Hrs")
        plt.show()

    timePulse=24.0*10.0+pulse

    PT=sp.interpolate.interp1d(np.array(tsdf['Time']), np.array(tsdf['Phase']), bounds_error=False)

    phase1=fmod(PT(timePulse+24.0), 2*sp.pi) #phase one day after the end of the pulses
    phase2=fmod(PT(timePulse+20*24.0), 2*sp.pi) #phase at same time of day 10 days later

    outOfPhase=angle_difference(phase1, phase2)
    return(outOfPhase*24.0/(2.0*sp.pi))

        
        

    
def findMin(shift):
    xvalues=np.arange(0,24.0,1.0)
    myfunc=lambda x: ProgressMeasure(shift=shift, pulse=x)
    y=map(myfunc, xvalues)
    print zip(xvalues,y)
    y=np.absolute(y)
    idxmin=np.argmin(y)
    xval=xvalues[idxmin]
    return(xval, y[idxmin])





    

if __name__=='__main__':

    ProgressMeasure(8.0, 23.0, True)
    sys.exit(0)

    #findMin(6.0)

    #print "This many hours out of phase ",  ProgressMeasure(shift=6, pulse=10.0, plot=True)
    #sys.exit(0)

    shifts=np.arange(-12.0,12.0,1.0)
    f=open("results_opt_mil_more.txt", "w")
    for s in shifts:
        bestTime, hoursOff=findMin(s)
        output=str(s)+"\t"+str(bestTime)+"\t"+str(hoursOff)+"\n"
        f.write(output)
    f.close()
    
    

    
    

