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
from scipy import interpolate
import torch 


pd.options.mode.chained_assignment = None  # default='warn'

def simple_norm_stepshr_sleep_classifier(t):
        t[0,:] = torch.tanh((t[0,:] - 10.0)/ 100.0)
        t[1, torch.nonzero(t[1,:])] = torch.tanh((t[1,torch.nonzero(t[1,:])] - 60.0) / 30.0)
        return t 

def phase_ic_guess(time_of_day: float):
    time_of_day = np.fmod(time_of_day, 24.0)

    # Wake at 8 am after 8 hours of sleep
    # State at 00:00
    psi = 1.65238233

    # Convert to radians, add to phase
    psi += time_of_day * np.pi / 12
    return psi

def abs_hour_diff(x, y):
    """
    function abs_hour_diff(x,y)

    Find the difference in hours between
    two clock times (wrapped)
    """
    a1 = min(x, y)
    a2 = max(x, y)
    s1 = a2-a1
    s2 = 24.0+a1-a2
    return(min(s1, s2))


def cut_phases_12(p):
    """
    Function to make the branch cut for the DLMO times be at 12 instead of 24.
    This is better because lots of DLMOs are near midnight, but many fewer are near
    noon.

        cut_phases_12(timept)
    """

    while (p < 0.0):
        p += 24.0

    p = np.fmod(p, 24.0)

    if p > 12:
        return p-24.0
    else:
        return p


def convert_binary(x, breakpoint: float = 0.50):
    x[x <= breakpoint] = 0.0
    x[x > breakpoint] = 1.0
    return x


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
