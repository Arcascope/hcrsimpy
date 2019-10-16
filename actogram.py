"""

This class can be used to make some actogram plots of circadian rhythms


CBT=DLMO+7hrs
CBT=DLMO_mid+2hrs
CBT=circadian phase pi in the model
DLMO=circadian phase 5pi/12=1.309 in the model

"""



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


class actogram(object):

    def __init__(self, ax, tsdf, threshold=10.0):
        """Create an actogram object from time series data given as a pandas dataframe with the columns Time, Phase, Light_Level, it can have more columns but must have those at least
        Time should be measured in hours and Phase should be unwrapped phase in radians, light level should be given as lux
        """
        self.tsdf=tsdf
        self.num_days=ceil(tsdf['Time'].iloc[-1]/24.0)
        self.threshold=threshold
        self.tsdf=tsdf
        self.ax=ax

        #Set graphical parameters
        self.ax.set_ylim(0, self.num_days)
        self.ax.set_xlim(0,48)
        self.ax.set_yticks(np.arange(self.num_days))
        ylabels_list=list(range(1,int(self.num_days)+1))
        ylabels_list.reverse()
        self.ax.set_yticklabels(ylabels_list)
        self.ax.set_xticks(np.arange(0,48+6,6))
        xlabels_list=list(range(0,30,6))+list(range(6,30,6))
        self.ax.set_xticklabels(xlabels_list)
        self.ax.set_xticks(np.arange(0,48,1), minor=True)


        self.ax.yaxis.grid(True, linewidth=1.0, color='k')
        self.ax.xaxis.grid(True)
        self.ax.plot(24.0*np.ones(100), np.linspace(0, self.num_days,100), ls='--', lw=2.0, color='black', zorder=9)

        self.addLightSchedule()
        self.dlmo=self.addCircadianPhases()


    def getRectangles(self, timeon, timeoff, colorIn='yellow'):
        bottom_x=fmod(timeon, 24.0)
        bottom_y=self.num_days-int(timeon/24.0)-1
        r1 = plt.Rectangle((bottom_x, bottom_y), timeoff-timeon,1, fc=colorIn, alpha=0.5)
        r2 = plt.Rectangle((bottom_x+24.0, bottom_y), timeoff-timeon,1, fc=colorIn, alpha=0.5)
        return((r1,r2))

    def addRect(self, timeon, timeoff, colorIn='yellow'):
        """Used to add a rectangle to the axes"""
        r=self.getRectangles(timeon, timeoff, colorIn)
        self.ax.add_patch(r[0])
        self.ax.add_patch(r[1])

    def addLightSchedule(self):
        """Add the light schedule as colored rectangles to the axes"""

        lightdata=np.array(self.tsdf.Light_Level)
        timedata=np.array(self.tsdf.Time)
        lightsOn=False
        if (lightdata[0]>self.threshold):
            lightsOn=True
            lightStart=timedata[0]
        else:
            darkOn=True
            darkStart=timedata[0]

        dayCounter=int(timedata[0]/24.0) #count the days in the data set
        for i in range(1, len(lightdata)):
            currentDay=int(timedata[i]/24.0)
            if (currentDay!=dayCounter):
                dayCounter=currentDay
                if (lightsOn==True):
                    self.addRect(lightStart, timedata[i])
                    if (i+1<len(timedata)):
                        lightStart=timedata[i+1] #reset the light counter to start over the next day
                else:
                    self.addRect(darkStart, timedata[i], colorIn='black')
                    if (i+1< len(timedata)):
                        darkStart=timedata[i+1]

            if (lightdata[i]<self.threshold and lightsOn==True):
                self.addRect(lightStart, timedata[i-1])
                lightsOn=False
                darkOn=True
                darkStart=timedata[i]
            if (lightsOn==False and lightdata[i]>=self.threshold):
                lightsOn=True
                lightStart=timedata[i]
                darkOn=False
                self.addRect(darkStart, timedata[i-1], colorIn='black')


    def addCircadianPhases(self, tsdf2=None, col='blue'):
        """
        This method can be used to add a set of circadian phases onto an axis. You should pass in a time series data frame. It is assumed the light data will be identical and we
        are just adding a seperate set of circadian phase markers for a comparison model.
        addCircadian(self, tsdf2=None, col='blue')
        """

        if (tsdf2 is None):
            tsdf2=self.tsdf


        #Find a function to give the estimated dlmo times
        dlmo_func=sp.interpolate.interp1d(np.array(tsdf2['Phase']), np.array(tsdf2['Time']), bounds_error=False)

        real_days=self.tsdf['Time'].iloc[-1]/24.0


        if (tsdf2.Phase.iloc[0]<1.309):
            dlmo_phases=np.arange(1.309, real_days*2.0*sp.pi, 2*sp.pi) #all the dlmo phases using 1.309 as the cicadian phase of the DLMO
            dlmo_times=np.array(list(([fmod(x,24.0) for x in list(map(dlmo_func, dlmo_phases))])))
            dlmo_times= dlmo_times[np.isfinite(dlmo_times)]
            dayYvalsDLMO=self.num_days-np.arange(0.5, len(dlmo_times)+0.5, 1.0)
        else:
            dlmo_phases=np.arange(1.309+2*sp.pi, real_days*2.0*sp.pi, 2*sp.pi) #all the dlmo phases using 1.309 as the cicadian phase of the DLMO
            dlmo_times=np.array(list([fmod(x,24.0) for x in list(map(dlmo_func, dlmo_phases))]))
            dlmo_times= dlmo_times[np.isfinite(dlmo_times)]
            dayYvalsDLMO=self.num_days-np.arange(1.5, len(dlmo_times)+1.5, 1.0)

        if (tsdf2.Phase.iloc[0]<sp.pi):
            cbt_phases=np.arange(sp.pi, real_days*2.0*sp.pi, 2*sp.pi)
            cbt_times=np.array(list([fmod(x,24.0) for x in list(map(dlmo_func, cbt_phases))]))
            cbt_times=cbt_times[np.isfinite(cbt_times)]
            dayYvalsCBT=self.num_days-(np.arange(0.5, len(cbt_times)+0.5, 1.0))
        else:
            cbt_phases=np.arange(sp.pi+2*sp.pi, real_days*2.0*sp.pi, 2*sp.pi)
            cbt_times=np.array(list([fmod(x,24.0) for x in list(map(dlmo_func, cbt_phases))]))
            cbt_times=cbt_times[np.isfinite(cbt_times)]
            dayYvalsCBT=self.num_days-np.arange(1.5, len(cbt_times)+1.5, 1.0)


        self.ax.scatter(dlmo_times, dayYvalsDLMO, color=col, zorder=10, marker="s")
        self.ax.scatter(dlmo_times+24.0, dayYvalsDLMO, color=col, zorder=10, marker="s")

        self.ax.scatter(cbt_times, dayYvalsCBT, color=col, marker='x', zorder=10)
        self.ax.scatter(cbt_times+24.0, dayYvalsCBT, color=col, marker='x', zorder=10)

        return dlmo_times

    def addMarker(self, time, col='red'):
        """Add another marker to the actogram at a particular time, i.e. Carrie melatonin dosage time"""
        time2=[fmod(time, 24.0), fmod(time,24.0), fmod(time, 24.0)]
        yval=[self.num_days-int(time/24.0)+0.5, self.num_days-int(time/24.0)+0.5-1, self.num_days-int(time/24.0)+0.5-2]

        self.ax.scatter(time2, yval, color='red', marker="v")
        self.ax.scatter(np.array(time2)+24.0, yval, color='red', marker="v")

    def addMarker2(self, time, col='red'):
        """Add another marker to the actogram at a particular time, i.e. Carrie melatonin dosage time"""
        time2=[fmod(time, 24.0), fmod(time,24.0), fmod(time, 24.0)]
        yval=[self.num_days-int(time/24.0)+0.5, self.num_days-int(time/24.0)+0.5-1, self.num_days-int(time/24.0)+0.5-2]

        self.ax.scatter(time2, yval, color='red', marker="P")
        self.ax.scatter(np.array(time2)+24.0, yval, color='red', marker="P")


    def getDLMOtimes(self):
        return(self.dlmo)
