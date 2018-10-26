import numpy as np
import scipy as sp
from scipy.integrate import *
import pylab as plt
from math import *
import sys
from sets import Set
import pandas as pd
from scipy import interpolate
import seaborn as sbn
from LightSchedule import *


class TwoPopModel:

    def __init__(self, LightFun):

        #Read the parameters from a file
        self.getParameters()
        self.Light=LightFun



    def getParameters(self):

        self.tauV=24.25
        self.tauD=24.0
        self.Kvv=0.05
        self.Kdd=0.04
        self.Kvd=0.05
        self.Kdv=0.01
        self.gamma=0.024
        self.A1=0.43
        self.A2=0.28
        self.BetaL=0.09
        self.BetaL2=-1.49
        self.sigma=0.07
        self.G=33.75
        self.alpha_0=0.05
        self.delta=0.0075
        self.p=1.5
        self.I0=9985.0
        
        


    def alpha0(self,t):

        return(self.alpha_0*pow(self.Light(t), self.p)/(pow(self.Light(t), self.p)+self.I0));


    def derv(self,t,y):

        Rv=y[0];
	Rd=y[1];
	Psiv=y[2];
	Psid=y[3];
	n=y[4];

	Bhat=self.G*(1.0-n)*self.alpha0(t);

	LightAmp=self.A1*0.5*Bhat*(1.0-pow(Rv,4.0))*cos(Psiv+self.BetaL)+self.A2*0.5*Bhat*Rv*(1.0-pow(Rv,8.0))*cos(2.0*Psiv+self.BetaL2);
	LightPhase=self.sigma*Bhat-self.A1*Bhat*0.5*(pow(Rv,3.0)+1.0/Rv)*sp.sin(Psiv+self.BetaL)-self.A2*Bhat*0.5*(1.0+pow(Rv,8.0))*sp.sin(2.0*Psiv+self.BetaL2);

        dydt=np.zeros(5)
       
	dydt[0]=-self.gamma*Rv+self.Kvv/2.0*Rv*(1-pow(Rv,4.0))+self.Kdv/2.0*Rd*(1-pow(Rv,4.0))*cos(Psid-Psiv)+LightAmp;
	dydt[1]=-self.gamma*Rd+self.Kdd/2.0*Rd*(1-pow(Rd,4.0))+self.Kvd/2.0*Rv*(1.0-pow(Rd,4.0))*cos(Psid-Psiv);
	dydt[2]=2.0*sp.pi/self.tauV+self.Kdv/2.0*Rd*(pow(Rv,3.0)+1.0/Rv)*sp.sin(Psid-Psiv)+LightPhase;
	dydt[3]=2.0*sp.pi/self.tauD-self.Kvd/2.0*Rv*(pow(Rd,3.0)+1.0/Rd)*sp.sin(Psid-Psiv);
	dydt[4]=60.0*(self.alpha0(t)*(1.0-n)-self.delta*n);
        return(dydt)


    def integrateModel(self, tend, initial=[1.0,1.0, 0.0, 0.0, 0.0]):

        dt=0.1
        self.ts=np.arange(0.0,tend+dt,dt)
        initial[2]=fmod(initial[2], 2*sp.pi) #start the initial phase between 0 and 2pi
        initial[3]=fmod(initial[3], 2*sp.pi) #start the initial phase between 0 and 2pi

        r=sp.integrate.solve_ivp(self.derv,(0,tend), initial, t_eval=self.ts, method='Radau') 
        self.results=np.transpose(r.y)
        
        ent_angle=fmod(self.results[-1,2], 2*sp.pi)*24.0/(2.0*sp.pi) #angle at the lights on period
        return(ent_angle)

    def integrateModelData(self, timespan, initial):
        """ Integrate the model using a light function defined by data 
        integrateModelData(self, timespan, initial)
        """
        dt=0.01
        self.ts=np.arange(timespan[0], timespan[1], dt)
        initial[2]=fmod(initial[2], 2*sp.pi) #start the initial phase between 0 and 2pi
        initial[3]=fmod(initial[3], 2*sp.pi) #start the initial phase between 0 and 2pi
        r=sp.integrate.solve_ivp(self.derv,(timespan[0],timespan[-1]), initial, t_eval=self.ts, method='Radau')
        
        self.results=np.transpose(r.y)


    def integrateTransients(self, numdays=50):
        """Integrate the model for 500 days to get rid of any transients, returns the endpoint to be used as initial conditions"""

        tend=numdays*24.0 #need to change this back to 500
        r=sp.integrate.solve_ivp(self.derv,(0,tend), [0.7, 0.7, 0.0, 0.0, 0.0], t_eval=[tend], method='Radau')
        results_trans=np.transpose(r.y)
        return(results_trans[-1,:])

    def getTS(self):
        """Return a time series data frame for the system"""

        light_ts=map(self.Light, self.ts)
        theta=self.results[:,2]-self.results[:,1]
        Rv=self.results[:,0]
        Rd=self.results[:,1]
        ts=pd.DataFrame({'Time': self.ts, 'Light_Level':light_ts, 'Phase': self.results[:,2], 'R': self.results[:,0], 'n': self.results[:,4], 'theta':self.results[:,2]-self.results[:,3]})
        return(ts)
       

def guessICDataTwoPop(LightFunc, time_zero, length=150):
    """Guess the Initial conditions for the model using the persons light schedule
    Need to add a check to see if the system is entrained at all
    """
    
    a=TwoPopModel(LightFunc)
    #make a rough guess as to the initial phase
    init=[0.7, 0.7, fmod(time_zero/24.0*2*sp.pi+sp.pi, 2*sp.pi), fmod(time_zero/24.0*2*sp.pi+sp.pi, 2*sp.pi), 0.01]
    
    a.integrateModel(int(length)*24.0, initial=init)
    init=a.results[-1,:]
    a.integrateModel(48.0, initial=init)

    limit_cycle=a.results
    timeDay=lambda x: fmod(x,48.0)
    lc_ts=np.array(map(timeDay, a.ts))

    idx=np.searchsorted(lc_ts,time_zero)-1
    initial=limit_cycle[idx,:]
    initial[2]=fmod(initial[2], 2*sp.pi)
    initial[3]=fmod(initial[3], 2*sp.pi)
    print time_zero, initial
    return(initial)




        
        



if __name__=='__main__':
    pass
   
