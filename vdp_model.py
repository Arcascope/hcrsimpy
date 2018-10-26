
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







class vdp_model:

    def __init__(self, LightFun):

        #Set the parameters
        self.taux=24.2
        self.mu=0.23
        self.G=33.75
        self.alpha_0=0.05
        self.delta=0.0075
        self.p=0.50
        self.I0=9500.0
        self.kparam=0.55
        
        self.Light=LightFun


    
    def alpha0(self,t):

        return(self.alpha_0*pow((self.Light(t)/self.I0), self.p));


    def derv(self,t,y):
        
        x=y[0];
	xc=y[1];
	n=y[2];

	Bhat=self.G*(1.0-n)*self.alpha0(t)*(1-0.4*x)*(1-0.4*xc);

        dydt=np.zeros(3)
	       
	dydt[0]=sp.pi/12.0*(xc+Bhat);
	dydt[1]=sp.pi/12.0*(self.mu*(xc-4.0/3.0*pow(xc,3.0))-x*(pow(24.0/(0.99669*self.taux),2.0)+self.kparam*Bhat));
	dydt[2]=60.0*(self.alpha0(t)*(1.0-n)-self.delta*n); 

        return(dydt)


    def integrateModel(self, tend, initial=[1.0,1.0,0.0]):

        
        dt=0.1
        self.ts=np.arange(0.0,tend+dt,dt) 

        r=sp.integrate.solve_ivp(self.derv,(0,tend), initial, t_eval=self.ts, method='Radau') #uses RK45
        self.results=np.transpose(r.y)

        ent_angle=1.0*atan2(self.results[-1,1],self.results[-1,0]); #times negative one because VDP runs clockwise versus counterclockwise
	if (ent_angle < 0.0):
	    ent_angle+=2*sp.pi;
	
	ent_angle=ent_angle*24.0/(2.0*sp.pi);
        return(ent_angle)

    def integrateModelData(self, timespan, initial):
        """ Integrate the model using a light function defined by data 
        integrateModelData(self, timespan, initial)
        """
        dt=0.01
        self.ts=np.arange(timespan[0], timespan[1], dt)
        r=sp.integrate.solve_ivp(self.derv,(timespan[0],timespan[-1]), initial, t_eval=self.ts, method='Radau')
        self.results=np.transpose(r.y)
          
        

    def integrateTransients(self, numdays=500):
        """Integrate the model for numdays to get rid of any transients, returns the endpoint to be used as initial conditions"""

        tend=numdays*24.0

        r=sp.integrate.solve_ivp(self.derv,(0,tend), [0.7, 0.0, 0.0], t_eval=[tend], method='Radau') 
        results_trans=np.transpose(r.y)
        
        return(results_trans[-1,:])


    def getTS(self):
        """Return a time series data frame for the system"""

        light_ts=map(self.Light, self.ts)
        Amplitude=np.sqrt(self.results[:,0]**2+self.results[:,1]**2) #define the amplitude as the sqrt of each coordinate squared

        #Need to extract a phase in radians
        wrappedPhase=-1.0*np.arctan2(self.results[:,1],self.results[:,0])

        
        #Make it between 0 and 2pi
        for i in range(len(wrappedPhase)):
            if wrappedPhase[i]<0.0:
                wrappedPhase[i]+=2*sp.pi
        
        
        Phase=np.unwrap(wrappedPhase, discont=0.0)
        
        ts=pd.DataFrame({'Time': self.ts, 'Light_Level':light_ts, 'Phase': Phase, 'R': Amplitude, 'n': self.results[:,2]})
        return(ts)

        
def guessICDataVDP(LightFunc, time_zero, length=50):
    """Guess the Initial conditions for the model using the persons light schedule"""
    
    a=vdp_model(LightFunc)
    #make a rough guess as to the initial phase
    init=np.array([1.0, 1.0, 0.0])
    
    a.integrateModel(int(length)*24.0, initial=init)
    init=a.results[-1,:]
    a.integrateModel(48.0, initial=init)

    limit_cycle=a.results
    timeDay=lambda x: fmod(x,48.0)
    lc_ts=np.array(map(timeDay, a.ts))

    idx=np.searchsorted(lc_ts,time_zero)-1
    initial=limit_cycle[idx,:]
    print time_zero, initial
    return(initial)    




        
        



if __name__=='__main__':

    duration=16.0 #gets 8 hours of sleep
    intensity=150.0
    wake=6.0
    LightFunReg=lambda t: RegularLightSimple(t,intensity,wake,duration)

    a=vdp_model(LightFunReg)
    a.integrateModel(24*40)
    tsdf=a.getTS()
    
