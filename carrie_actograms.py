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
from singlepop_model_melatonin_carrie import *
from stroboscopic import *
from MelatoninSchedule import * 

def actogramRegularLightMel(MelatoninTime=96.0):
    """Show the effect of a regular light schedule on the circadian clock"""

    duration=16.0 #gets 8 hours of sleep
    intensity=150.0
    wake=6.0
    LightFunReg=lambda t: RegularLightSimple(t,intensity,wake,duration)

    Mel=lambda t: threeMelPulse(t, timePulse=MelatoninTime)
    
    #Create SP Model
    a=SinglePopModel(LightFunReg, Mel)
    init=a.integrateTransients()
    ent_angle=a.integrateModel(24*40, initial=init);
    tsdf=a.getTS()
    plt.figure()
    ax=plt.gca()
    acto=actogram(ax, tsdf) #add an actogram to those axes
    acto.addMarker(MelatoninTime)

    plt.title('Entrainment under Regular Light Conditions')
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    actogramRegularLightMel(110.0)
