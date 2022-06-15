"""
This python module contains functions for creating light schedules giving the light levels in lux as a function of time
in hours.

Used as input to simulate the circadian models.

Contains some common schedules as well as some custom data driven schedules

"""
from __future__ import division
from __future__ import print_function

from builtins import map
from builtins import object
from builtins import range
from builtins import zip
from math import fmod

import numpy as np
import pandas as pd
import pylab as plt
import scipy as sp
from numba import jit
from past.utils import old_div
from scipy import interpolate

pd.options.mode.chained_assignment = None  # default='warn'


def cal_days_diff(a, b):
    """Get the calander days between two time dates"""
    A = a.replace(hour=0, minute=0, second=0, microsecond=0)
    B = b.replace(hour=0, minute=0, second=0, microsecond=0)
    return (A - B).days


@jit(nopython=True)
def interpolateLinear(t, xvals, yvals):
    """Implement a faster method to get linear interprolations of the light functions"""

    if (t >= xvals[-1]):
        return (0.0)
    if (t <= xvals[0]):
        t += 24.0

    i = np.searchsorted(xvals, t) - 1
    ans = (yvals[i + 1] - yvals[i]) / \
          ((xvals[i + 1] - xvals[i]) * (t - xvals[i])) + yvals[i]
    return (ans)


@jit(nopython=True)
def interpolateLinearExt(t, xvals, yvals):
    """Implement a faster method to get linear interprolations of the light functions, exclude non-full days"""
    i = np.searchsorted(xvals, t) - 1
    ans = (yvals[i + 1] - yvals[i]) / \
          ((xvals[i + 1] - xvals[i]) * (t - xvals[i])) + yvals[i]
    return (ans)


def parse_dt(date, time):
    strDate = date + ' ' + time
    return pd.to_datetime(strDate, format='%m/%d/%Y %I:%M %p')


@jit(nopython=True)
def LightLog(lightlevel, threshold=1.0):
    """Take the log10 of the light levels, but map 0 to zero and not negative numbers"""
    if (lightlevel < threshold):
        return (0.0)
    if (lightlevel >= threshold):
        return (np.log10(lightlevel))


def RegularLight(t, Intensity, PP, period):
    """Define a basic light schedule"""
    val = 0.5 * sp.tanh(100 * (fmod(t, period))) - 0.5 * \
        sp.tanh(100 * (fmod(t, period) - PP))
    return (Intensity * val)


def RegularLightSimple(t, Intensity=150.0, wakeUp=8.0, workday=16.0):
    """Define a basic light schedule with a given intensity of the light, wakeup time and length of the active period
    (non-sleeping)
    This schedule will automatically repeat on a daily basis, so each day will be the same.....
    """

    s = fmod(t, 24.0) - wakeUp

    if (s < 0):
        s += 24.0

    val = 0.5 * sp.tanh(100 * (s)) - 0.5 * sp.tanh(100 * (s - workday))
    return (Intensity * val)


def ShiftWorkLight(t, dayson=5, daysoff=2):
    """Simulate a night shift worker. Assume they are working a night shift for dayson number of days followed by
     daysoff normal days where they revert to a normal schedule
    ShiftWorkLight(t, dayson=5, daysoff=2)
    """

    t = fmod(t, (dayson + daysoff) * 24)  # make it repeat
    if (t <= 24 * dayson):

        return (
            RegularLightSimple(
                t,
                Intensity=150.0,
                wakeUp=16.0,
                workday=16.0))

    else:
        return (
            RegularLightSimple(
                t,
                Intensity=250.0,
                wakeUp=9.0,
                workday=16.0))


def SocialJetLag(
        t,
        weekdayWake=7.0,
        weekdayBed=24.0,
        weekendWake=11.0,
        weekendBed=2.0):
    """Simulate a social jetlag schedule. """

    t = fmod(t, (7) * 24)  # make it repeat each week

    if (t <= 24 * 5):
        # Monday through thursday
        duration = fmod(weekdayBed - weekdayWake, 24.0)
        if duration < 0.0:
            duration += 24.0
        return (
            RegularLightSimple(
                t,
                Intensity=150.0,
                wakeUp=weekdayWake,
                workday=duration))
    if (t > 24 * 5 and t <= 24 * 7):
        # Friday, stay up late
        duration = fmod(weekendBed - weekendWake, 24.0)
        if duration < 0.0:
            duration += 24.0
        return (
            RegularLightSimple(
                t,
                Intensity=250.0,
                wakeUp=weekendWake,
                workday=duration))


def SlamShift(t, shift=8.0, intensity=150.0, beforeDays=10):
    """Simulate a sudden shift in the light schedule"""

    t = fmod(t, 40 * 24)  # make it repeat every 40 days

    if (t <= 24 * beforeDays):
        return (RegularLightSimple(t, intensity, wakeUp=8.0, workday=16.0))
    else:
        newVal = fmod(8.0 + shift, 24.0)
        if (newVal < 0):
            newVal += 24.0
        return (RegularLightSimple(t, intensity, wakeUp=newVal, workday=16.0))


def OneDayShift(t, pulse=23.0, wakeUp=8.0, workday=16.0, shift=8.0):
    t = fmod(t, 40 * 24)

    beforeDays = 10.0

    if (t < 24 * beforeDays - 3 * 24.0):
        return (RegularLightSimple(t, 150.0, wakeUp=wakeUp, workday=workday))

    if ((t >= 24 * beforeDays - 3 * 24.0) and (t <= 24 * beforeDays)):
        # adjustment day get to add one hour of bright light
        pulse += 24 * beforeDays
        Light = RegularLightSimple(t, 150.0, wakeUp=wakeUp, workday=workday)
        Light += 10000.0 * (0.5 * sp.tanh(100 * (t - pulse + 72.0)) -
                            0.5 * sp.tanh(100 * (t - 1.0 - pulse + 72.0)))
        Light += 10000.0 * (0.5 * sp.tanh(100 * (t - pulse + 48.0)) -
                            0.5 * sp.tanh(100 * (t - 1.0 - pulse + 48.0)))
        Light += 10000.0 * (0.5 * sp.tanh(100 * (t - pulse + 24.0)) -
                            0.5 * sp.tanh(100 * (t - 1.0 - pulse + 24.0)))
        return (Light)

    if (t > 24 * (beforeDays)):
        newVal = fmod(8.0 + shift, 24.0)
        if (newVal < 0):
            newVal += 24.0
        return (RegularLightSimple(t, 150.0, wakeUp=newVal, workday=16.0))


class WrightLightData(object):

    def __init__(self):

        self.art_data = np.array(sorted(map(tuple, list(np.loadtxt(
            '../ExpData/Wright/Wright_Artificial.csv', delimiter=','))), key=lambda x: x[0]))
        self.nat_data = np.array(sorted(map(tuple, list(np.loadtxt(
            '../ExpData/Wright/Wright_Natural.csv', delimiter=','))), key=lambda x: x[0]))

        # Make sure the data covers [0,24] only
        self.nat_data = self.nat_data[self.nat_data[:, 0] <= 24.0]
        self.art_data = self.art_data[self.art_data[:, 0] <= 24.0]

        self.nat_data = np.vstack(([0, 1], self.nat_data))

        self.art_data = np.vstack((self.art_data, [24.0, 11.4]))
        self.art_data = np.vstack(([0.0, 11.5589392], self.art_data))

        self.LightFunNat = interpolate.interp1d(
            self.nat_data[:, 0], self.nat_data[:, 1])
        self.LightFunArt = interpolate.interp1d(
            self.art_data[:, 0], self.art_data[:, 1])

        # self.plotLight()

    def LightFunction(self, t, art=False):

        s = fmod(t, 24.0)

        if art:
            return (self.LightFunArt(s))
        else:
            return (self.LightFunNat(s))

    def plotLight(self):
        plt.plot(self.art_data[:, 0], np.log10(
            self.art_data[:, 1]), color='blue')
        plt.plot(self.nat_data[:, 0], np.log10(
            self.nat_data[:, 1]), color='red')
        plt.show()


class NoisyLightInput(object):

    def __init__(self):
        # Length of active period
        self.ap_sd = 2.0
        self.ap_loc = 16.0

        # Light variability
        self.nl_sd = 2.0  # Natural light standard deviation set at 10^ to this number
        self.al_sd = 50.0  # artificial light standard deviation

        # Background light
        # the light level doesn't go less that this during active period
        self.background_ll = 150.0

        self.generateTS()

    def generateTS(self):

        dt = 1.0 / 60.0  # minute long bins
        self.time = np.arange(0.0, 24.0 * 1000 + dt, dt)

        self.lightVals = []

        period = 24.0  # period of the days
        AP = 16.0  # activity period (this is allowed to vary day to day)
        # photoperiod (hours where you can get outdoor sunlight, this is fixed)
        PP = 12.0
        IntLast = 0.0
        # This controls when the AP starts (defaults to dawn). Can't do wake up
        # earlier than dawn as of yet
        LightOnset = 0.0

        for t in self.time:

            day = t / 24.0

            if (day.is_integer()):
                AP = np.random.normal(loc=self.ap_loc, scale=self.ap_sd)
                if (AP > 22.0):
                    AP = 22.0
                if (PP < 12.0):
                    AP = 12.0

                # LightOnset=np.random.uniform(0.0,2.0)

            val = 0.5 * sp.tanh(10 * (fmod(t, period) - LightOnset)) - 0.5 * sp.tanh(
                10 * (fmod(t, period) - (AP + LightOnset)))  # Sleep modulation of light input
            if (fmod(t, period) <= PP):
                # Random light Intensity, allowing for natural light exposures
                Int = self.background_ll + \
                    10 ** np.random.normal(0.0, self.nl_sd)
            else:
                # Random light intensity only indoor ranges
                Int = self.background_ll + \
                    np.random.normal(loc=0.0, scale=self.al_sd)

            if (Int > 10000):
                Int = 10000.0
            if (Int < 0.0):
                Int = 0.0
            self.lightVals.append(Int * val)

            IntLast = Int

        self.lightVals = np.array(self.lightVals)

        # Do some smoothing of the data
        LightTS = pd.Series(self.lightVals, self.time)
        b = LightTS.rolling(window=10, center=False).mean()
        self.lightVals[10:] = b.values[10:]

        self.LightFunc = interpolate.interp1d(self.time, self.lightVals)

    def plotNoisyLight(self, numdays=10):

        plt.plot(self.time[0:numdays * 24 * 60] / 24.0,
                 np.log10(self.lightVals[0:numdays * 24 * 60] + 1.0))
        plt.show()

    def plotAverageLight(self):

        dt = 1.0 / 60.0  # minute long bins
        avgLightLevels = dict()
        possibleKeys = Set()
        for tt in self.time:
            possibleKeys.add(np.round(fmod(tt, 24.0), 3))

        for h in possibleKeys:
            avgLightLevels[h] = list()

        for j in range(len(self.time)):
            s = np.round(fmod(self.time[j], 24.0), 3)
            avgLightLevels[s].append(self.lightVals[j])

        averageVal = []
        for k in list(avgLightLevels.keys()):
            averageVal.append((k, np.mean(avgLightLevels[k])))

        averageVal = sorted(averageVal, key=lambda x: x[0])

        x, y = list(zip(*averageVal))
        y = np.array(y)
        plt.plot(x, y + 1.0)
        plt.show()


class hchs_light(object):

    def __init__(self, filename, smoothWindow=20):

        fileData = pd.read_csv(filename, delimiter=',', skiprows=0)
        # fileData=fileData.set_index('Date_Time')
        fileData['Lux'] = fileData.whitelight + fileData.bluelight + \
            fileData.redlight + fileData.greenlight
        fileData.time = pd.to_datetime(fileData.time, format='%H:%M:%S')
        self.data = fileData

        startingDay = self.data.dayofweek.iloc[0]
        dow = list(self.data.dayofweek)
        count = 0
        day_list = [0]
        for i in range(1, len(dow)):
            if dow[i - 1] == dow[i]:
                day_list.append(count)
            else:
                count += 1
                day_list.append(count)

        self.data.day = day_list

        # Create an index to count the hours of the data (UNITS of hours)

        self.data['TimeTotal'] = (self.data.day) * 24.0 + pd.DatetimeIndex(self.data.time).hour + pd.DatetimeIndex(
            self.data.time).minute / 60.0 + pd.DatetimeIndex(self.data.time).second / 3600.0
        self.data['TimeCount'] = pd.DatetimeIndex(self.data.time).hour + pd.DatetimeIndex(
            self.data.time).minute / 60.0 + pd.DatetimeIndex(self.data.time).second / 3600.0
        self.data['Lux'] = self.data.Lux.rolling(
            smoothWindow, center=True).max()
        # Smooth the wake scores as well
        self.data['wake'] = self.data.wake.rolling(
            smoothWindow, center=True).mean()

        # Drop the missing values introduced
        self.data = self.data.dropna()

        self.startTime = self.data.TimeTotal.iloc[0]
        self.endTime = self.data.TimeTotal.iloc[-1]

        # Make a function which gives the light level
        self.LightFunction = lambda t: interpolateLinear(
            fmod(
                t, self.endTime), np.array(
                self.data['TimeTotal']), np.array(
                self.data['Lux']))

        # Make a extended function for lomg simulations
        xvals = np.array(self.data['TimeTotal'])
        yvals = np.array(self.data.Lux)
        yvals = yvals[xvals >= 24.0]
        xvals = xvals[xvals >= 24.0]

        num_days = int(xvals[-1] / 24.0)
        final_time = num_days * 24.0
        # Exclude the final day
        yvals = yvals[xvals <= final_time]
        xvals = xvals[xvals <= final_time]

        # start at zero
        xvals = xvals - 24.0

        self.LightFunctionInitial = lambda t: np.absolute(
            interpolateLinearExt(fmod(t, xvals[-1]), xvals, yvals))
        # Trim off the non-full days within the data set and make a light
        # function using that data

    def findMidSleep(self, cutoff=0.5, show=False):
        """Estimate the midpoint of the sleep. Actually the angular mean of the sleep times"""

        av_data = self.data.groupby(by=['TimeCount']).mean()

        # Now round the sleep scores:
        av_data.loc[av_data['wake'] >= cutoff, 'wake'] = 1.0
        av_data.loc[av_data['wake'] < cutoff, 'wake'] = 0.0

        sleepTimes = pd.Series(av_data.loc[av_data['wake'] == 0.0, :].index)

        def toangle(x): return np.exp(complex(0, 1) * (2 * sp.pi * x / 24.0))

        all_angles = np.angle(list(map(toangle, sleepTimes)))
        Z = np.mean(list(map(toangle, sleepTimes)))
        angle = np.fmod(np.angle(Z) + 2.0 * sp.pi, 2.0 * sp.pi)
        amp = abs(Z)

        timeEquivalent = old_div(angle * 24.0, (2.0 * sp.pi))

        if (show):
            plt.scatter(sp.cos(all_angles), sp.sin(all_angles))
            plt.scatter(sp.cos(angle), sp.sin(angle), color='green')
            plt.xlim(-1, 1)
            plt.ylim(-1, 1)
            plt.show()

            plt.plot(av_data.index, av_data.wake, color='green')
            plt.plot(av_data.index, list(
                map(LightLog, list(av_data.Lux))), color='blue')
            plt.show()

        return (timeEquivalent)

    def plot_light(self):
        """Make a plot of the light schedule with each day on top of the last"""
        ts = np.arange(
            self.data.TimeTotal.iloc[0], self.data.TimeTotal.iloc[-1], 0.01)
        y = list(map(self.LightFunction, ts))
        plt.plot(ts, list(map(LightLog, y)))
        # plt.scatter(np.array(self.data.TimeTotal), map(LightLog, self.data.Lux), color='red', s=0.1)
        plt.plot(np.array(self.data.TimeTotal), self.data.wake)
        plt.show()


class JennyDataReader(object):
    """BL is the school year, P1 is the summer months"""

    def __init__(self, filename):

        if (('BL' in filename) or ('B0' in filename)):
            self.label = 'school'
        if ('P1' in filename):
            self.label = 'summer'

        if not ('HumanData' in filename):
            Fullfilename = "~/work/Research/gpu/Human/HumanData/DATA/" + filename
        else:
            Fullfilename = filename

        fileData = pd.read_csv(
            Fullfilename,
            delimiter=',',
            skiprows=3,
            parse_dates={
                'Date_Time': [
                    'Date',
                    'Time']},
            date_parser=parse_dt)
        fileData = fileData.set_index('Date_Time')
        # Make the Sleep column categorical
        fileData['Sleep'] = fileData['Sleep or Awake?'].astype('category')
        fileData = fileData.drop(columns=['Sleep or Awake?'])

        self.data = fileData[['Lux', 'Sleep']]

        # Add a numerical sleep score column
        sleep_score_dict = {'S': 1.0, 'W': 0.0}

        def sleepscoref(x):
            return sleep_score_dict[x]

        self.data.loc[:, 'Sleep_Score'] = pd.Series(data=list(
            map(sleepscoref, self.data['Sleep'])), index=self.data.index, dtype=np.float64)

        self.removeNaps()

        self.findFirstLight()

        self.data = self.data.first('8D')  # take only the first 8 days of data
        # Define the light level as a max over the last 5 minutes
        self.data.loc[:, 'Lux'] = self.data['Lux'].rolling(
            window=10, center=True).max()
        self.data = self.data.dropna()

        days = np.array([cal_days_diff(d, pd.DatetimeIndex(self.data.index)[
            0]) for d in pd.DatetimeIndex(self.data.index)])

        self.data['TimeCount'] = pd.DatetimeIndex(
            self.data.index).hour * 60 + pd.DatetimeIndex(self.data.index).minute
        self.data['TimeCount'] = self.data['TimeCount'] / 60.0

        # Find the light time in hours
        self.data['TimeTotal'] = (days * 24.0 * 60 + pd.DatetimeIndex(
            self.data.index).hour * 60 + pd.DatetimeIndex(self.data.index).minute) / 60.0

        # Store time bounds for ease of use
        self.startTime = self.data.TimeTotal.iloc[0]
        self.endTime = self.data.TimeTotal.iloc[-1]

        self.LightFunctionAvg = lambda t: interpolateLinear(
            t, np.array(
                self.avg_data.index), np.array(
                self.avg_data['Lux']), period=24.0)
        self.LightFunction = lambda t: interpolateLinear(
            t, np.array(
                self.data['TimeTotal']), np.array(
                self.data['Lux']))

        # Make a extended function for lomg simulations
        xvals = np.array(self.data['TimeTotal'])
        yvals = np.array(self.data.Lux)
        yvals = yvals[xvals >= 24.0]
        xvals = xvals[xvals >= 24.0]

        # Need to chop off the hangover on the end
        num_days = int(xvals[-1] / 24.0)
        final_time = num_days * 24.0
        # Exclude the final day
        yvals = yvals[xvals <= final_time]
        xvals = xvals[xvals <= final_time]

        ind_included_day = []

        for i in range(0, self.data.shape[0]):
            ttVal = self.data['TimeTotal'].iloc[i]
            if ((ttVal >= xvals[0]) and (ttVal <= xvals[-1])):
                ind_included_day.append(1)
            else:
                ind_included_day.append(0)

        self.data['Included_Day'] = pd.Series(
            ind_included_day, index=self.data.index)

        # start at zero
        xvals = xvals - 24.0

        self.total_days = num_days

        # Trim off the non-full days within the data set and make a light
        # function using that data
        spInterp = sp.interpolate.UnivariateSpline(xvals, yvals, ext=3)
        self.LightFunctionInitial = lambda t: np.absolute(
            spInterp(fmod(t, xvals[-1])))

        # Extract the dates to be used in comparing with sleep data for
        # included_days only

        self.data['Date_Only'] = self.data.index.date

    def findFirstLight(self):
        """Find the first time light is recorded in the data set"""

        threshold = 10.0
        firstLight = 0
        for j in range(0, self.data.shape[0]):
            if self.data.iloc[j, 0] > threshold:
                firstLight = j
                break

        lastLight = 0
        for j in range(1, self.data.shape[0]):
            if self.data.iloc[-j, 0] > threshold:
                lastLight = self.data.shape[0] - j
                break

        self.data = self.data.iloc[firstLight:lastLight, :]

    def plotLightSleep(self):
        """Make a plot of the light function"""

        x = np.arange(0, 10 * 24.0, 0.1)
        y = np.array(list(map(self.LightFunctionInitial, x)))

        plt.plot(x / 24.0, list(map(LightLog, y)))
        plt.show()

    def removeNaps(self):
        """Try and remove nap times from the sleep column"""

        hours = pd.DatetimeIndex(self.data.index).hour
        selection_criteria = (hours < 18.0) & (hours > 11.0)

        self.data.loc[selection_criteria, 'Sleep_Score'] = 0.0

    def estimateAvgSleepMidpoint(self, cutoff=0.50, show=False):

        av_data = self.data.groupby(by=['TimeCount']).mean()

        # Now round the sleep scores:
        av_data.loc[av_data['Sleep_Score'] >= cutoff, 'Sleep_Score'] = 1.0
        av_data.loc[av_data['Sleep_Score'] < cutoff, 'Sleep_Score'] = 0.0

        sleepTimes = pd.Series(
            av_data.loc[av_data['Sleep_Score'] == 1.0, :].index)

        def toangle(x): return np.exp(complex(0, 1) * (2 * sp.pi * x / 24.0))

        all_angles = np.angle(list(map(toangle, sleepTimes)))
        Z = np.mean(list(map(toangle, sleepTimes)))
        angle = np.fmod(np.angle(Z) + 2.0 * sp.pi, 2.0 * sp.pi)
        amp = abs(Z)

        timeEquivalent = old_div(angle * 24.0, (2.0 * sp.pi))

        if (show):
            plt.scatter(sp.cos(all_angles), sp.sin(all_angles))
            plt.scatter(sp.cos(angle), sp.sin(angle), color='green')
            plt.xlim(-1, 1)
            plt.ylim(-1, 1)
            plt.show()

        return ((timeEquivalent, amp))

    def estimateAvgSleepOnset(self):
        pass

    def estimateAvgSleepOffset(self):
        pass


class WrightLightData(object):

    def __init__(self):

        self.art_data = np.array(sorted(map(tuple, list(np.loadtxt(
            '../../ExpData/Wright/Wright_Artificial.csv', delimiter=','))), key=lambda x: x[0]))
        self.nat_data = np.array(sorted(map(tuple, list(np.loadtxt(
            '../../ExpData/Wright/Wright_Natural.csv', delimiter=','))), key=lambda x: x[0]))

        # Make sure the data covers [0,24] only
        self.nat_data = self.nat_data[self.nat_data[:, 0] <= 24.0]
        self.art_data = self.art_data[self.art_data[:, 0] <= 24.0]

        self.nat_data = np.vstack(([0, 1], self.nat_data))

        self.art_data = np.vstack((self.art_data, [24.0, 11.4]))
        self.art_data = np.vstack(([0.0, 11.5589392], self.art_data))

        self.LightFunNat = interpolate.interp1d(
            self.nat_data[:, 0], self.nat_data[:, 1])
        self.LightFunArt = interpolate.interp1d(
            self.art_data[:, 0], self.art_data[:, 1])

        # self.plotLight()

    def LightFunction(self, t, art=False):

        s = fmod(t, 24.0)

        if art:
            return (self.LightFunArt(s))
        else:
            return (self.LightFunNat(s))

    def plotLight(self):
        plt.plot(self.art_data[:, 0], np.log10(
            self.art_data[:, 1]), color='blue')
        plt.plot(self.nat_data[:, 0], np.log10(
            self.nat_data[:, 1]), color='red')
        plt.show()


if __name__ == '__main__':
    hc = hchs_light('../../HumanData/HCHS/hchs-sol-sueno-00579338.csv')
    hc.plot_light()
    print(hc.findMidSleep(show=True))
