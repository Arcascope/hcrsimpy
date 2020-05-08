"""
This file allows for the comparision of the predictions for the three models and shows the results on an actogram
"""



from __future__ import print_function
import numpy as np
import scipy as sp
from scipy.integrate import *
import pylab as plt
from math import *
import sys
import pandas as pd
from scipy import interpolate


from HCRSimPY.plots import *
from HCRSimPY.light_schedules import *
from HCRSimPY.models import *
from HCRSimPY.utils import circular_stats





def findKeyTimes(tsdf):
    """Find the DLMO and CBT times for a given time series prediction"""

    wrapped_time=np.round([fmod(x, 24.0) for x in list(tsdf.Time)],2)
    df=pd.DataFrame({'Time': wrapped_time, 'Phase': tsdf.Phase})
    df2=df.groupby('Time')['Phase'].agg({'Circular_Mean':circular_mean, 'Phase_Coherence': phase_coherence, 'Samples':np.size})

    mean_func=sp.interpolate.interp1d(np.array(df2['Circular_Mean']), np.array(df2.index))
    return((mean_func(sp.pi), mean_func(1.309)))








def regularRoutineStats():
    """ Find how the DLMO and CBT phases depend on the light intensity provided to the model """

    Intensities=[50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 750.0, 1000.0, 2500.0, 5000.0, 10000.0]
    resultsSP=[]
    resultsTP=[]
    resultsVDP=[]
    for i in Intensities:
        duration=16.0 #gets 8 hours of sleep
        intensity=i
        wake=6.0
        LightFunReg=lambda t: RegularLightSimple(t,intensity,wake,duration)

        #Create SP Model
        a=SinglePopModel(LightFunReg)
        init=a.integrateTransients()
        ent_angle=a.integrateModel(24*40, initial=init);
        tsdf=a.getTS()
        CBT, DLMO=findKeyTimes(tsdf)
        resultsSP.append((i, DLMO, CBT))

        #Add vdp
        v=vdp_model(LightFunReg)
        init=v.integrateTransients()
        ent_angle=v.integrateModel(24*40, initial=init);
        tsdf2=v.getTS()
        CBT, DLMO=findKeyTimes(tsdf2)
        resultsVDP.append((i, DLMO, CBT))

        #add tp model
        t=TwoPopModel(LightFunReg)
        init=t.integrateTransients()
        ent_angle=t.integrateModel(24*40, initial=init);
        tsdf3=t.getTS()
        CBT, DLMO=findKeyTimes(tsdf3)
        resultsTP.append((i, DLMO, CBT))

    resultsSP=np.array(resultsSP)
    resultsVDP=np.array(resultsVDP)
    resultsTP=np.array(resultsTP)


    plt.plot(np.log10(resultsSP[:,0]), 22.0-resultsSP[:,1], color='blue', lw=2.0)
    plt.plot(np.log10(resultsVDP[:,0]), 22.0-resultsVDP[:,1], color='darkgreen', lw=2.0)
    plt.plot(np.log10(resultsTP[:,0]), 22.0-resultsTP[:,1], color='red', lw=2.0)

    plt.scatter(np.log10(resultsSP[:,0]), 22.0-resultsSP[:,1], color='blue', marker="o")
    plt.scatter(np.log10(resultsVDP[:,0]), 22.0-resultsVDP[:,1], color='darkgreen', marker="^")
    plt.scatter(np.log10(resultsTP[:,0]), 22.0-resultsTP[:,1], color='red', marker="s")

    plt.xlabel(r'$\log_{10}(Lux)$')
    plt.ylabel('Lights Off-DLMO')
    plt.title('Light Intensity Effects on DLMO Timing')
    plt.xlim(1.5,4.0)
    plt.ylim(1.0,3.0)
    plt.tight_layout()
    plt.show()







def compareRegularLight(Intensity=150.0):

    duration=16.0 #gets 8 hours of sleep
    wake=6.0
    LightFunReg=lambda t: RegularLightSimple(t,Intensity,wake,duration)

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


    #Add vdp
    v=vdp_model(LightFunReg)
    init=v.integrateTransients()
    v.Light=JetLag
    ent_angle=v.integrateModel(24*40, initial=init);
    tsdf2=v.getTS()
    acto.addCircadianPhases(tsdf2, col='darkgreen')

    #add two population model
    t=TwoPopModel(LightFunReg)
    init=t.integrateTransients()
    t.Light=JetLag
    ent_angle=t.integrateModel(24*40, initial=init);
    tsdf3=t.getTS()
    acto.addCircadianPhases(tsdf3, col='red')

    plt.figure()
    ax=plt.gca()
    strobo=stroboscopic(ax, tsdf[tsdf['Time']>=10*24.0])
    strobo.addStroboPlot(tsdf2[tsdf2['Time']>=10*24.0], col='darkgreen')
    strobo.addStroboPlot(tsdf3[tsdf2['Time']>=10*24.0], col='red')

    plt.show()



if __name__=='__main__':

    JetLagActogram(-10.0)
    #compareRegularLight()
    #compareShiftWork(15,10)
