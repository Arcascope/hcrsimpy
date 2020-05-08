"""
This file can be used to examine the predicted effects of given light schedules on the single population human model.

It contains a few basic function to make some actograms for basic light schedules including regular, jetlag and shiftwork
scenarios.


"""



from __future__ import print_function

import numpy as np
import scipy as sp
import pylab as plt


from HCRSimPY.plots import *
from HCRSimPY.light_schedules import *
from HCRSimPY.models import *




def actogramRegularLight(save=False):
    """Show the effect of a regular light schedule on the circadian clock"""

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

    plt.title('Entrainment under Regular Light Conditions')
    plt.tight_layout()
    if (save):
        plt.savefig('Regular_Light_actogram.eps')
    plt.show()


def actogramShiftWork(dayson=5, daysoff=2):
    """
    Simulate shift work schedule using the SP model:
    compareShiftWork(dayson=5, daysoff=2)
    """

    LightFun=lambda t: ShiftWorkLight(t, dayson, daysoff)

    #Create SP Model
    a=SinglePopModel(LightFun)
    init=a.integrateTransients()
    ent_angle=a.integrateModel(24*40, initial=init);
    tsdf=a.getTS()
    plt.figure()
    ax=plt.gca()
    acto=actogram(ax, tsdf) #add an actogram to those axes

    plt.title('Entrainment under Shift Work Conditions')
    plt.show()





def makeActogram(Light):
    """A generate function to create an actogram where you provide a light function
    makeActogram(Light)
    """

    #Create SP Model
    a=SinglePopModel(Light)
    init=a.integrateTransients()
    ent_angle=a.integrateModel(24*40, initial=init);
    tsdf=a.getTS()
    plt.figure()
    ax=plt.gca()
    acto=actogram(ax, tsdf) #add an actogram to those axes

    plt.show()


def JetLagActogram(shift):
    """
    Simulate the circadian rhythms of a slam shift in the light schedule. By default this will for a fully entrained subject and the shift will occur on the 11th day
    JetLagActogram(shift)

    """

    if (shift<0.0):
        print(("Simulating westbound travel by ", abs(shift), " time zones"))
    else:
        print(("Simulating eastbound travel by ", abs(shift), " time zones"))



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
    ax.set_title("8 Hour Shift with no Intervention")
    plt.figure()
    ax=plt.gca()
    strobo=stroboscopic(ax, tsdf[tsdf['Time']>=10*24.0])

    plt.show()



if __name__=='__main__':

    JetLagActogram(8.0)
    actogramRegularLight()
    actogramShiftWork(5,2)
