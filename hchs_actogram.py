import numpy as np
import scipy as sp
from scipy.integrate import *
import pylab as plt
from math import *
import sys
from sets import Set
import pandas as pd
from scipy import interpolate
from circular_stats import *
from LightSchedule import *
from singlepop_model import *
from vdp_model import *
from twopop_model import *
from stroboscopic import *

from latexify import *
latexify()

from actogram import *
from joblib import Parallel, delayed
import os
import random

def findKeyDLMOTimes(tsdf):
    """Find the DLMO and CBT times for a given time series prediction"""

    wrapped_time=np.round(map(lambda x: fmod(x, 24.0), list(tsdf.Time)),2)
    df=pd.DataFrame({'Time': wrapped_time, 'Phase': tsdf.Phase})
    df2=df.groupby('Time')['Phase'].agg({'Circular_Mean':circular_mean, 'Phase_Coherence': phase_coherence, 'Samples':np.size})
    mean_func=sp.interpolate.interp1d(np.array(df2['Circular_Mean']), np.array(df2.index))
    coherence_func=sp.interpolate.interp1d(np.array(df2['Circular_Mean']),np.array(df2['Phase_Coherence']))
    return((mean_func(1.309), coherence_func(1.309)))



def record_diff(tsdfS, tsdfV, tsdfT):
     """Find the differences in the DLMO timing of the three models for that given light schedule"""

     d1, r1 =findKeyDLMOTimes(tsdfS)
     d2, r2=findKeyDLMOTimes(tsdfV)
     d3, r3=findKeyDLMOTimes(tsdfT)

     return((d1,d2,d3,r1,r2,r3))


def get_diff(f):
     """Used to find the average DLMO times of all data in the hchs data set, in a parallel fashion"""
     fnum=f.split('-')[-1].split('.')[0]

     #try:
     trans_days=50
     hc=hchs_light(f)
     sm=hc.findMidSleep()
     init=guessICData(hc.LightFunctionInitial, 0.0, length=trans_days)
     initVDP=guessICDataVDP(hc.LightFunctionInitial, 0.0, length=trans_days)
     initTwo=guessICDataTwoPop(hc.LightFunctionInitial, 0.0, length=trans_days)

     a=SinglePopModel(hc.LightFunctionInitial)
     b=vdp_model(hc.LightFunctionInitial)
     c=TwoPopModel(hc.LightFunctionInitial)
     ent_angle=a.integrateModelData((0.0, 40.0*24.0), initial=init);
     ent_angle_vdp=b.integrateModelData((0.0, 40.0*24.0), initial=initVDP);
     ent_angle_two=c.integrateModelData((0.0, 40.0*24.0), initial=initTwo);
     tsdf=a.getTS()
     tsdf_vdp=b.getTS()
     tsdf_two=c.getTS()
     fnum=f.split('-')[-1].split('.')[0]
     d1, d2, d3,r1,r2,r3=record_diff(tsdf, tsdf_vdp, tsdf_two)
     return(fnum+", "+str(d1)+", "+str(d2)+", "+str(d3)+", "+str(r1)+ ", "+ str(r2)+ ", "+ str(r3)+", "+ str(sm)+"\n")

     #except:
          #print "Error with: ", f
          #return(fnum+", "+str(-1.0)+", "+str(-1.0)+", "+str(-1.0)+"\n")


     
     


def get_all_hchs_files():
    """ Get a list of every hchs file """

    file_list=[]
     
    for file in os.listdir("../../HumanData/HCHS/"):
        if file.endswith(".csv"):
            file_list.append(str(os.path.join("../../HumanData/HCHS/", file)))

    return(file_list)


def runParticularData(filenumber, trans_days=50):
     """Given a id number run that system"""
     fileName="../../HumanData/HCHS/hchs-sol-sueno-"+str(filenumber)+".csv"

     hc=hchs_light(fileName)

     init=guessICData(hc.LightFunctionInitial, 0.0, length=trans_days)
     initVDP=guessICDataVDP(hc.LightFunctionInitial, 0.0, length=trans_days)
     initTwo=guessICDataTwoPop(hc.LightFunctionInitial, 0.0, length=trans_days)
     
     a=SinglePopModel(hc.LightFunctionInitial)
     b=vdp_model(hc.LightFunctionInitial)
     c=TwoPopModel(hc.LightFunctionInitial)
     ent_angle=a.integrateModelData((0.0, 40.0*24.0), initial=init);
     ent_angle_vdp=b.integrateModelData((0.0, 40.0*24.0), initial=initVDP);
     ent_angle_two=c.integrateModelData((0.0, 40.0*24.0), initial=initTwo);
     tsdf=a.getTS()
     tsdf_vdp=b.getTS()
     tsdf_two=c.getTS()
     plt.figure()
     ax=plt.gca()
     acto=actogram(ax, tsdf) #add an actogram to those axes
     acto.addCircadianPhases(tsdf_vdp, col='darkgreen')
     acto.addCircadianPhases(tsdf_two, col='red')
     ax.set_title('HCHS Actogram')
     plt.show()

     print filenumber


def runAllFilesDLMO():
     """
        Run through all the files and measure the DLMO predicted differences for the three models
     """
     fl=get_all_hchs_files()
     print "Total Files: ", len(fl)
     allOutputs= Parallel(n_jobs=32)(delayed(get_diff)(f) for f in fl)

     
     outfile=open('hchs_model_diff_new.csv', 'w')
     outfile.write('Filename, SP_DLMO, VDP_DLMO, TP_DLMO, R_SP, R_VDP, R_TP, Est_Sleep_MP\n')

     for o in allOutputs:
          outfile.write(o)




def plotRandomSP(trans_days=100):

    fl=get_all_hchs_files()
    fileName=random.choice(fl)
    hc=hchs_light(fileName)

    init=guessICData(hc.LightFunctionInitial, 0.0, length=trans_days)
     
    a=SinglePopModel(hc.LightFunctionInitial)
     
    ent_angle=a.integrateModelData((0.0, 10.0*24.0), initial=init);
    tsdf=a.getTS()
    plt.figure()
    ax=plt.gca()
    acto=actogram(ax, tsdf) #add an actogram to those axes
    
    ax.set_title('HCHS Actogram')
    plt.tight_layout()
    plt.savefig('Modern_Light_actogram.eps')
    plt.show()

    print fileName

    
def chooseRandomData(trans_days=50):

     fl=get_all_hchs_files()
     fileName=random.choice(fl)
     hc=hchs_light(fileName)

     init=guessICData(hc.LightFunctionInitial, 0.0, length=trans_days)
     initVDP=guessICDataVDP(hc.LightFunctionInitial, 0.0, length=trans_days)
     initTwo=guessICDataTwoPop(hc.LightFunctionInitial, 0.0, length=trans_days)
     
     a=SinglePopModel(hc.LightFunctionInitial)
     b=vdp_model(hc.LightFunctionInitial)
     c=TwoPopModel(hc.LightFunctionInitial)
     ent_angle=a.integrateModelData((0.0, 40.0*24.0), initial=init);
     ent_angle_vdp=b.integrateModelData((0.0, 40.0*24.0), initial=initVDP);
     ent_angle_two=c.integrateModelData((0.0, 40.0*24.0), initial=initTwo);
     tsdf=a.getTS()
     tsdf_vdp=b.getTS()
     tsdf_two=c.getTS()
     d1, d2, d3=record_diff(tsdf, tsdf_vdp, tsdf_two)
     plt.figure()
     ax=plt.gca()
     acto=actogram(ax, tsdf) #add an actogram to those axes
     acto.addCircadianPhases(tsdf_vdp, col='darkgreen')
     acto.addCircadianPhases(tsdf_two, col='red')
     ax.set_title('HCHS Actogram')
     plt.show()

     print fileName




def runAllShiftWorkers():
    """
        Run through all the sw and make an actogram for their recorded light schedules
    """
    swlist=open('HCHS_ShiftWorkers_PID.csv').readlines()[1:]

    swlist=[s.strip().strip("\"") for s in swlist]

    
    allOutputs= Parallel(n_jobs=32)(delayed(chooseShiftWorker)(f) for f in swlist)

    print "Total Number of files generated: ", sum(allOutputs)
     
    



     

def chooseShiftWorker(sw, trans_days=50.0):
    """Choose a shift worker and make an actogram"""

    try:
        fileName="../../HumanData/HCHS/hchs-sol-sueno-"+sw+".csv"

        hc=hchs_light(fileName)

        init=guessICData(hc.LightFunctionInitial, 0.0, length=trans_days)
     
        a=SinglePopModel(hc.LightFunctionInitial)
     
        ent_angle=a.integrateModelData((0.0, 10.0*24.0), initial=init);
        tsdf=a.getTS()
        plt.figure()
        ax=plt.gca()
        acto=actogram(ax, tsdf) #add an actogram to those axes
        dlmo=acto.getDLMOtimes()
        #print "Phase Coherence DLMO: ", phase_coherence_clock(dlmo)
        saveString=sw+"\t"+str(phase_coherence_clock(dlmo))
        print saveString 
    
        mytitle='HCHS Actogram '+sw
        ax.set_title(mytitle)
        plt.tight_layout()
        figname="ShiftWorkerPlots/sw_actogram_"+sw+".eps"
        plt.savefig(figname)
        return(1)
    except:
        return(0)

    

if __name__=='__main__':


    #runAllShiftWorkers()

    
    
     runAllFilesDLMO()

     sys.exit(0)
     
     if len(sys.argv)<2:
          plotRandomSP()
     else:
          if len(sys.argv)==2:
               runParticularData(sys.argv[1])
          else:
               runParticularData(sys.argv[1], int(sys.argv[2]))



