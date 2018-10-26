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
from vdp_model import *
from twopop_model import *
from stroboscopic import *


def compareRegularLight():

    duration=16.0 #gets 8 hours of sleep
    intensity=150.0
    wake=6.0
    LightFunReg=lambda t: RegularLightSimple(t,intensity,wake,duration)
    
    #Create SP Model
    a=SinglePopModel(LightFunReg)
    init=a.integrateTransients()
    ent_angle=a.integrateModel(24*40, initial=init);
    tsdf=a.getTS()
    plt.figure()
    ax=plt.gca()
    acto=actogram(ax, tsdf) #add an actogram to those axes

    #Add vdp 
    v=vdp_model(LightFunReg)
    init=v.integrateTransients()
    ent_angle=v.integrateModel(24*40, initial=init);
    tsdf2=v.getTS()
    acto.addCircadianPhases(tsdf2, col='darkgreen')

    #add tp model
    t=TwoPopModel(LightFunReg)
    init=t.integrateTransients()
    ent_angle=t.integrateModel(24*40, initial=init);
    tsdf3=t.getTS()
    acto.addCircadianPhases(tsdf3, col='red')
    plt.title('Entrainment under Regular Light Conditions')
    plt.show()


def compareShiftWork(dayson=5, daysoff=2):

    LightFun=lambda t: ShiftWorkLight(t, dayson, daysoff)

    #Create SP Model
    a=SinglePopModel(LightFun)
    init=a.integrateTransients()
    ent_angle=a.integrateModel(24*40, initial=init);
    tsdf=a.getTS()
    plt.figure()
    ax=plt.gca()
    acto=actogram(ax, tsdf) #add an actogram to those axes

    #Add vdp 
    v=vdp_model(LightFun)
    init=v.integrateTransients()
    ent_angle=v.integrateModel(24*40, initial=init);
    tsdf2=v.getTS()
    acto.addCircadianPhases(tsdf2, col='darkgreen')

    #add tp model
    t=TwoPopModel(LightFun)
    init=t.integrateTransients()
    ent_angle=t.integrateModel(24*40, initial=init);
    tsdf3=t.getTS()
    acto.addCircadianPhases(tsdf3, col='red')
    plt.title('Entrainment under Shift Work Conditions')
    plt.show()





def compareActogram(Light):

    #Create SP Model
    a=SinglePopModel(Light)
    init=a.integrateTransients()
    ent_angle=a.integrateModel(24*40, initial=init);
    tsdf=a.getTS()
    plt.figure()
    ax=plt.gca()
    acto=actogram(ax, tsdf) #add an actogram to those axes

    #Add vdp 

    v=vdp_model(Light)
    init=v.integrateTransients()
    ent_angle=v.integrateModel(24*40, initial=init);
    tsdf2=v.getTS()

    acto.addCircadianPhases(tsdf2)
    plt.show()


def JetLagActogram(shift):
    """Simulate the circadian rhythms of a slam shift in the light schedule. By default this will for a fully entrained subject and the shift will occur on the 11th day"""

    if (shift<0.0):
        print "Simulating westbound travel by ", abs(shift), " time zones"
    else:
        print "Simulating eastbound travel by ", abs(shift), " time zones"


        
    LightFunReg=lambda t: RegularLightSimple(t,150.0, 8.0,16.0)
    JetLag=lambda t: SlamShift(t, shift)

    
    #Create SP Model
    a=SinglePopModel(LightFunReg)
    init=a.integrateTransients()
    a.Light=JetLag
    ent_angle=a.integrateModel(24*40, initial=init);
    tsdf=a.getTS()
    plt.figure()
    ax=plt.gca()
    acto=actogram(ax, tsdf) #add an actogram to those axes

    
    #Add vdp 

    v=vdp_model(LightFunReg)
    init=v.integrateTransients()
    v.Light=JetLag
    ent_angle=v.integrateModel(24*40, initial=init);
    tsdf2=v.getTS()

    acto.addCircadianPhases(tsdf2, col='darkgreen')
    

    plt.figure()
    ax=plt.gca()
    strobo=stroboscopic(ax, tsdf[tsdf['Time']>=10*24.0])
    strobo.addStroboPlot(tsdf2[tsdf2['Time']>=10*24.0])
    
    plt.show()



if __name__=='__main__':

    #JetLagActogram(-11.0)
    compareRegularLight()
    #compareShiftWork(15,10)
    


