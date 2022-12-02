# %%

from scipy.ndimage import gaussian_filter1d
from copy import deepcopy
import pylab as plt
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))


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


class Actogram:

    def __init__(self,
                 time_total: np.ndarray,
                 light_vals: np.ndarray,
                 second_zeit: np.ndarray = None,
                 ax=None,
                 threshold=10.0,
                 threshold2=None,
                 opacity: float = 0.0,
                 color: str = 'black',
                 smooth=True,
                 sigma=[2.0,2.0]
                 ):
        """
            Create an actogram of the given marker..... 

            (self, time_total: np.ndarray, light_vals: np.ndarray, ax=None, threshold=10.0) 
        """

        self.time_total = time_total
        self.light_vals = light_vals
        self.num_days = np.ceil((time_total[-1] - time_total[0])/24.0)

        self.second_zeit = second_zeit if second_zeit is not None else light_vals

        if smooth:
            self.light_vals = gaussian_filter1d(self.light_vals, sigma=sigma[0])
            self.second_zeit = gaussian_filter1d(self.second_zeit, sigma=sigma[1])

        if threshold2 is None:
            threshold2 = threshold

        if ax is not None:
            self.ax = ax
        else:
            #plt.figure(figsize=(18, 12))
            plt.figure()
            ax = plt.gca()
            self.ax = ax

        # Set graphical parameters
        label_scale = int(np.floor(self.num_days/30))
        if label_scale < 1:
            label_scale = 1

        self.opacity = opacity
        self.darkColor = color
        start_day = np.floor(self.time_total[0]/24.0)
        self.ax.set_ylim(start_day, self.num_days+start_day)
        self.ax.set_xlim(0, 48)
        self.ax.set_yticks(
            np.arange(int(start_day), self.num_days+1+start_day, label_scale))
        ylabels_list = list(range(int(start_day), int(
            self.num_days+start_day)+1, label_scale))
        # ylabels_list.reverse()

        self.ax.set_yticklabels(ylabels_list)
        self.ax.set_xticks(np.arange(0, 48+3, 3))
        xlabels_list = list(range(0, 27, 3))+list(range(3, 27, 3))
        self.ax.set_xticklabels(xlabels_list)
        self.ax.set_xticks(np.arange(0, 48, 1), minor=True)

        #self.ax.yaxis.grid(False, linewidth=1.0, color='k')
        # self.ax.xaxis.grid(False)

        self.ax.plot(24.0*np.ones(100), np.linspace(0, self.num_days,
                     100), ls='--', lw=2.0, color='black', zorder=9)
        self.ax.set_xlabel("ZT")
        self.ax.set_ylabel("Days")

        self.addLightSchedule(self.light_vals, threshold, plt_option='left', color=self.darkColor)
        self.addLightSchedule(self.second_zeit, threshold2, plt_option='right', color=self.darkColor)
        self.ax.invert_yaxis()

    def getRectangles(self, timeon, timeoff, colorIn='white'):
        bottom_x = np.fmod(timeon, 24.0)
        bottom_y = int(timeon/24.0)  # -1
        alpha = self.opacity if colorIn != 'white' else 0.0
        r1 = plt.Rectangle((bottom_x, bottom_y), timeoff -
                           timeon, 1, fc=colorIn, zorder=-1, alpha=alpha)
        r2 = plt.Rectangle((bottom_x+24.0, bottom_y),
                           timeoff-timeon, 1, fc=colorIn, zorder=1, alpha=alpha)
        return((r1, r2))

    def addRect(self, timeon, timeoff, colorIn='white', plt_option='both'):
        """Used to add a rectangle to the axes"""
        r = self.getRectangles(timeon, timeoff, colorIn)
        if plt_option == 'left':
            self.ax.add_patch(r[0])
            return
        if plt_option == 'right':
            self.ax.add_patch(r[1])
            return
        self.ax.add_patch(r[0])
        self.ax.add_patch(r[1])

    def addLightSchedule(self, zeit: np.ndarray, 
                         threshold: float, 
                         plt_option: str = 'both', 
                         color='black'):
        """
            Add the light schedule as colored rectangles to the axes

        """

        lightdata = zeit
        timedata = self.time_total
        lightsOn = False
        if (lightdata[0] > threshold):
            lightsOn = True
            lightStart = timedata[0]
        else:
            darkOn = True
            darkStart = timedata[0]

        dayCounter = int(timedata[0]/24.0)  # count the days in the data set
        for i in range(1, len(lightdata)):
            currentDay = int(timedata[i]/24.0)
            if (currentDay != dayCounter):
                dayCounter = currentDay
                if (lightsOn == True):
                    self.addRect(
                        lightStart, timedata[i], plt_option=plt_option)
                    if (i+1 < len(timedata)):
                        # reset the light counter to start over the next day
                        lightStart = timedata[i+1]
                else:
                    self.addRect(
                        darkStart, timedata[i], colorIn=color, plt_option=plt_option)
                    if (i+1 < len(timedata)):
                        darkStart = timedata[i+1]

            if (lightdata[i] < threshold and lightsOn == True):
                self.addRect(lightStart, timedata[i-1], plt_option=plt_option)
                lightsOn = False
                darkOn = True
                darkStart = timedata[i]
            if (lightsOn == False and lightdata[i] >= threshold):
                lightsOn = True
                lightStart = timedata[i]
                darkOn = False
                self.addRect(
                    darkStart, timedata[i-1], colorIn=color, plt_option=plt_option)

    def plot_phasemarker(self, phase_marker_times: np.ndarray,
                         error: np.ndarray = None,
                         alpha=1.0,
                         alpha_error=0.30,
                         scatter=False,
                         *args, **kwargs):
        """
        This method takes in a list of times which are assumed to occur at the same 
        circadian phase (e.g. DLMO, CBTmin). These are plotted as points 
        on the actogram.  
            plot_phasemarker(self, phase_marker_times: np.ndarray, *args, **kwargs)
        """

        xvals = deepcopy(phase_marker_times)
        yvals = deepcopy(phase_marker_times)

        xvals = np.fmod(xvals, 24.0)
        yvals = np.floor(yvals / 24.0) + 0.5

        if scatter:
            self.ax.scatter(xvals, yvals, *args, **kwargs)
            self.ax.scatter(xvals+24.0, yvals, *args, **kwargs)

        idx_split = (np.absolute(np.diff(xvals)) > 6.0).nonzero()[0]+1
        xvals_split = np.split(xvals, idx_split)
        yvals_split = np.split(yvals, idx_split)
        if error is not None:
            error_split = np.split(error, idx_split)

        for (idx, xx) in enumerate(xvals_split):
            self.ax.plot(xx, yvals_split[idx], alpha=alpha, *args, **kwargs)
            self.ax.plot(
                xx+24.0, yvals_split[idx], alpha=alpha, *args, **kwargs)
            if error is not None:
                self.ax.fill_betweenx(
                    yvals_split[idx], xx-error_split[idx], xx+error_split[idx], alpha=alpha_error, *args, **kwargs)
                self.ax.fill_betweenx(
                    yvals_split[idx], xx-error_split[idx]+24.0, xx+error_split[idx]+24.0, alpha=alpha_error, *args, **kwargs)

    def plot_phasetimes(self, times: np.ndarray, phases: np.ndarray, error: np.ndarray = None,
                        alpha_error=0.30, alpha=1.0, *args, **kwargs):
        """
            This method takes observations of the phase and times (same length arrays)
            and adds them to the actogram.    

            plot_phasetimes(self, times: np.ndarray, phases: np.ndarray, *args, **kwargs)
        """
        xvals = deepcopy(phases)
        xvals = np.arctan2(np.sin(xvals), np.cos(xvals))
        for i in range(len(xvals)):
            if xvals[i] < 0.0:
                xvals[i] += 2*np.pi

        xvals = np.fmod(xvals, 2*np.pi)
        xvals *= 12.0/np.pi

        xvals = np.fmod(xvals, 24.0)
        yvals = deepcopy(times)
        yvals = np.floor(yvals / 24.0) + 0.5

        idx_split = (np.absolute(np.diff(xvals)) > 6.0).nonzero()[0]+1
        xvals_split = np.split(xvals, idx_split)
        yvals_split = np.split(yvals, idx_split)
        if error is not None:
            error_split = np.split(error, idx_split)

        for (idx, xx) in enumerate(xvals_split):
            self.ax.plot(xx, yvals_split[idx], alpha=alpha, *args, **kwargs)
            self.ax.plot(
                xx+24.0, yvals_split[idx], alpha=alpha, *args, **kwargs)
            if error is not None:
                self.ax.fill_betweenx(
                    yvals_split[idx], xx-error_split[idx], xx+error_split[idx], alpha=alpha_error, *args, **kwargs)
                self.ax.fill_betweenx(
                    yvals_split[idx], xx-error_split[idx]+24.0, xx+error_split[idx]+24.0, alpha=alpha_error, *args, **kwargs)


def plot_mae(dlmo_actual: np.ndarray, dlmo_pred: np.ndarray, norm_to: float = None, ax=None,  *args, **kwargs):

    dlmo_actual = np.fmod(dlmo_actual, 24.0)
    dlmo_pred = np.fmod(dlmo_pred, 24.0)

    # Make the plot range from from -12 to 12
    dlmo_pred = np.array([cut_phases_12(d) for d in list(dlmo_pred)])
    dlmo_actual = np.array([cut_phases_12(d) for d in list(dlmo_actual)])

    if norm_to is not None:
        dlmo_actual = dlmo_actual - np.mean(dlmo_actual)+norm_to

    if ax is None:
        plt.figure()
        ax = plt.gca()

    errors = dlmo_pred-dlmo_actual
    print(f"The MAE is: {np.mean(abs(errors))}")
    print(f"Within one hour {np.sum(abs(errors)<=1.0)}/{len(dlmo_pred)}")

    print(errors)

    ax.scatter(dlmo_actual, dlmo_pred, *args, **kwargs)
    ax.plot(np.arange(-12, 12, 1), np.arange(-12, 12, 1),
            ls='--', color='gray', lw=2.0)
    ax.plot(np.arange(-12, 12, 1), np.arange(-12, 12, 1)+1,
            ls='--', color='blue', lw=1.0)
    ax.plot(np.arange(-12, 12, 1), np.arange(-12, 12, 1)-1,
            ls='--', color='blue', lw=1.0)

    ax.set_ylabel("Model Prediction (hrs)")
    ax.set_xlabel("Experimental DLMO (hrs)")


def plot_torus(phase1: np.ndarray, phase2: np.ndarray, scale24=False, ax=None, *args, **kwargs):

    if ax is None:
        plt.figure()
        ax = plt.gca()

    if scale24:
        phase1 = np.fmod(phase1, 24.0) * np.pi/12.0
        phase2 = np.fmod(phase2,  24.0) * np.pi/24.0

    phase1 = np.arctan2(np.sin(phase1), np.cos(phase1))
    phase2 = np.arctan2(np.sin(phase2), np.cos(phase2))

    ax.scatter(phase1, phase2, *args, **kwargs)


# %%
if __name__ == '__main__':

    from models import SinglePopModel
    from light import *

    times = np.linspace(0, 30*24.0, 10000)
    light_obj = Light(lambda t: SlamShift(t), duration=30*24.0)
    act = Actogram(times, light_obj(times))

    spm = SinglePopModel()

    dlmo = spm.integrate_observer(SlamShift,
                                  (0.0, 30*24.0), [1.0, np.pi, 0.0], SinglePopModel.DLMOObs)

    act.plot_phasemarker(dlmo, color='red')
    plt.show()


def plot_dashboard(steps, spm, hr_norm=None, kuramoto=None,
                   start_dttm=None, title=None, legend=False):
    # Used to determine how much to plot
    hr = (kuramoto is not None) and (hr_norm is not None)

    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    fig, ax = plt.subplots(nrows=4, constrained_layout=True)

    # Set up Single Pop Model and steps vectors for plotting
    spm_real = spm[0, :] * np.cos(spm[1, :])
    spm_imag = spm[0, :] * np.sin(spm[1, :])
    sleep = [1.0 if v < 0.0 else np.nan for v in np.diff(spm[3, :])]

    step_time = np.linspace(0.0, 0.1*spm_real.shape[0], spm_real.shape[0])

    ax[0].plot(step_time, spm_real, color='blue', label="R cos(psi) from SPM")
    ax[0].plot(step_time, spm_imag + 2.1, color='blue',
               label="R sin(psi) from SPM")

    ax[1].plot(step_time, steps+1.0, color='blue', label="Processed Step data")
    ax[1].plot(step_time[1:], sleep, color="black", label="Sleep")
    ax[1].set_yscale('log')

    H_minus = 0.17
    H_plus = 0.6
    homeostat_a = 0.1
    ax[3].plot(step_time, spm[3, :], color="black", label="homeostat")
    ax[3].plot(step_time, H_plus + homeostat_a*spm_real, color="darkgreen")
    ax[3].plot(step_time, H_minus + homeostat_a*spm_real, color="blue")

    # Set up Kuramoto Model and Heartrate vectors for plotting
    if hr:
        kuramoto_real = kuramoto[0, :] * np.cos(kuramoto[1, :])
        kuramoto_imag = kuramoto[0, :] * np.sin(kuramoto[1, :])

        ax[0].plot(step_time, kuramoto_real, ":",
                   color='red', label="R cos(psi) from KNN")
        ax[0].plot(step_time, kuramoto_imag + 2.1, ":",
                   color='red', label="R sin(psi) from KNN")

        ax[2].plot(step_time, hr_norm, color='red',
                   label="Processed HR + Steps data")
        ax[2].plot(step_time[1:], np.array(sleep)-1.0, color="black")
    else:
        ax[2].axvspan(step_time[0], step_time[-1], color='grey', alpha=0.3,
                      label="No HR data")

    if start_dttm is not None:
        ax[0].set_xlabel(f"Hours after {start_dttm}", fontsize=14)

    if legend:
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()

    ax[0].grid(b=True, which='both', axis="x")
    ax[1].grid(b=True, which='both', axis="x")
    ax[2].grid(b=True, which='both', axis="x")

    xticks_vals = np.arange(0.0, step_time[-1], 24.0)
    for p in ax:
        p.set_xticks(xticks_vals)

    if title is None:
        title = "Modeling results"
    fig.suptitle(title, fontsize=16)
    plt.show()
