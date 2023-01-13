# %%


from copy import deepcopy
import datetime
from datetime import datetime
import numpy as np
import pandas as pd
import pylab as plt
from dataclasses import dataclass
from scipy.stats import linregress

import json
import gzip
from typing import List
import os
from .utils import times_to_angle, split_missing_data


@dataclass
class WearableObservation:
    steps: float
    heartrate: float
    wake: float


@dataclass
class AppleWatch:
    date_time: List[np.ndarray]
    time_total: List[np.ndarray]
    steps: List[np.ndarray]
    heartrate: List[np.ndarray] = None
    wake: List[np.ndarray] = None
    phase_measure: np.ndarray = None
    phase_measure_times: np.ndarray = None
    subject_id: str = "Anon"
    data_id: str = "None"

    def __post_init__(self):
        pass

    def build_sleep_chunks(self, chunk_jump_hrs: float = 12.0):

        time_total = np.hstack(self.time_total)
        steps = np.hstack(self.steps)
        heartrate = np.hstack(self.heartrate)
        wake = np.hstack(self.wake)

        data = np.stack((steps, heartrate, wake), axis=0)
        j_idx = np.where(np.diff(time_total) > chunk_jump_hrs)[0]
        return np.split(data, j_idx, axis=1)

    def find_disruptions(self, cutoff: float = 5.0):
        """
            Find time periods where the schedule changes by a large amount
            Should return a list of days which are signidicantly different
            then the day prior. 
        """
        idx_per_day = 240
        days_window = 3

        times_changes = []
        changes = []

        for (batch_idx, ts_batch) in enumerate(self.time_total):
            if np.sum(self.steps[batch_idx]) > 0:
                binary_steps = np.where(self.steps[batch_idx] > 1, 1.0, 0.0)
                idx_start = days_window*idx_per_day
                for day_idx in range(idx_start, len(binary_steps)-days_window*idx_per_day, 1):
                    t_before = ts_batch[day_idx -
                                        idx_per_day*days_window: day_idx]
                    t_before = t_before[binary_steps[day_idx -
                                                     idx_per_day*days_window: day_idx] < 1.0]
                    R_prior, psi_prior = times_to_angle(t_before)

                    t_after = ts_batch[day_idx:day_idx+idx_per_day*days_window]
                    t_after = t_after[binary_steps[day_idx:day_idx +
                                                   idx_per_day*days_window] < 1.0]
                    R_post, psi_post = times_to_angle(t_after)
                    psi_diff_hrs = 12.0/np.pi * \
                        np.angle(np.exp(1j*(psi_prior-psi_post)))
                    times_changes.append(ts_batch[day_idx]/24.0)
                    changes.append(psi_diff_hrs)

        return list(set([int(np.floor(x)) for (k, x) in enumerate(times_changes) if abs(changes[k]) >= cutoff]))

    def flatten(self):
        """
            Make all of the wearable time series flatten out,
            this will makes all of the time series properties
            be a list with a single element which is a numpy
            array with those values. 
        """
        self.date_time = [np.hstack(self.date_time)]
        self.time_total = [np.hstack(self.time_total)]
        self.steps = [np.hstack(self.steps)]
        self.heartrate = [np.hstack(self.heartrate)]
        if self.wake is not None:
            self.wake = [np.hstack(self.wake)]

    def get_date(self, time_hr: float):
        idx = np.argmin(np.abs(np.array(self.time_total) - time_hr))
        return pd.to_datetime(self.date_time[idx], unit='s')

    def steps_hr_loglinear(self):
        """
        Find the log steps to hr linear regression parameters .
        hr=beta*log(steps+1.0)+alpha
        Returns beta,alpha
        """
        x = np.log(np.hstack(self.steps)+1.0)
        y = np.hstack(self.heartrate)
        x = x[y > 0]
        y = y[y > 0]
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        return slope, intercept

    def get_timestamp(self, time_hr: float):
        idx = np.argmin(np.abs(np.array(np.hstack(self.time_total)) - time_hr))
        return np.hstack(self.date_time)[idx]

    def trim_data_idx(self, idx1, idx2=None):

        if idx2 is None:
            idx2 = idx1
            idx1 = 0
        self.time_total = np.hstack(self.time_total)[idx1:idx2]
        self.steps = np.hstack(self.steps)[idx1:idx2]
        if self.heartrate is not None:
            self.heartrate = np.hstack(self.heartrate)[idx1:idx2]
        self.date_time = np.hstack(self.date_time)[idx1:idx2]
        self.time_total, self.steps, self.heartrate = split_missing_data(
            self.time_total, self.steps, self.heartrate, break_threshold=96.0)

    def trim_data(self, t1: float, t2: float):
        self.flatten()

        idx_select = (self.time_total[0] >= t1) & (self.time_total[0] <= t2)
        self.time_total[0] = self.time_total[0][idx_select]
        self.steps[0] = self.steps[0][idx_select]
        self.date_time[0] = self.date_time[0][idx_select]

        if self.heartrate is not None:
            self.heartrate[0] = self.heartrate[0][idx_select]
        if self.wake is not None:
            self.wake[0] = self.wake[0][idx_select]

        self.date_time, self.time_total, self.steps, self.heartrate = split_missing_data(self.date_time[0],
                                                                                         self.time_total[0], self.steps[0], self.heartrate[0], break_threshold=96.0)

    def get_light(self, multiplier: float = 1.0):
        steps = np.hstack(self.steps)
        time_total = np.hstack(self.time_total)
        return lambda x: np.interp(x, time_total, multiplier*steps)

    def get_bounds(self):
        time_total = np.hstack(self.time_total)
        return (time_total[0], time_total[-1])

    def plot(self, t1: float = None, t2: float = None, *args, **kwargs):

        time_total = np.hstack(self.time_total)
        steps = np.hstack(self.steps)
        heartrate = np.hstack(self.heartrate)

        if self.heartrate is not None:
            hr = deepcopy(heartrate)
            hr[hr == 0] = np.nan

        time_start = t1 if t1 is not None else time_total[0]/24.0
        time_end = t2 if t2 is not None else time_total[-1]/24.0

        fig = plt.figure()
        gs = fig.add_gridspec(2, hspace=0.0)
        ax = gs.subplots(sharex=True)
        fig.suptitle(
            f"{self.data_id} Applewatch Data: Subject {self.subject_id}")
        
        if self.heartrate is not None:
            ax[0].plot(time_total / 24.0, hr, color='red', *args, **kwargs)
            
        ax[1].plot(time_total / 24.0, steps,
                   color='darkgreen', *args, **kwargs)

        if self.wake is not None:
            ax[1].plot(time_total / 24.0, np.array(self.wake) *
                       max(np.median(steps), 50.0), color='k')

        if self.phase_measure_times is not None:
            [ax[0].axvline(x=_x / 24.0, ls='--', color='blue')
             for _x in self.phase_measure_times]
            [ax[1].axvline(x=_x / 24.0, ls='--', color='blue')
             for _x in self.phase_measure_times]

        ax[1].set_xlabel("Days")
        ax[0].set_ylabel("BPM")
        ax[1].set_ylabel("Steps")
        ax[0].grid()
        ax[1].grid()
        ax[0].set_xlim((time_start, time_end))
        ax[1].set_xlim((time_start, time_end+3.0))
        ax[0].set_ylim((0, 200))
        plt.show()

    def scatter_hr_steps(self, take_log: bool = True, *args, **kwargs):
        
        if self.heartrate is None:
            print("No heartrate data")
        fig = plt.figure()
        ax = plt.gca()

        steps = np.hstack(self.steps)
        heartrate = np.hstack(self.heartrate)

        if take_log:
            ax.scatter(np.log10(steps[heartrate > 0]+1.0),
                       np.log10(heartrate[heartrate > 0]),
                       color='red',
                       *args,
                       **kwargs)
        else:
            ax.scatter(steps[heartrate > 0], heartrate[heartrate > 0],
                       color='red',
                       *args,
                       **kwargs)

        ax.set_ylabel('BPM')
        ax.set_xlabel('Steps')
        ax.set_title('Heart Rate Data')
        plt.show()

    def plot_heartrate(self, t1=None, t2=None, *args, **kwargs):

        time_total = np.hstack(self.time_total)
        steps = np.hstack(self.steps)
        heartrate = np.hstack(self.heartrate)
        time_start = t1 if t1 is not None else time_total[0]
        time_end = t2 if t2 is not None else time_total[-1]

        hr = deepcopy(heartrate)
        hr[hr == 0] = np.nan
        fig = plt.figure()
        ax = plt.gca()

        ax.plot(time_total / 24.0, hr, color='red', *args, **kwargs)
        ax.set_xlabel('Days')
        ax.set_ylabel('BPM')
        ax.set_title('Heart Rate Data')
        ax.set_ylim((0, 220))
        plt.show()

    def to_json(self):
        """
            Create a JSON version of the dataclass, can be loaded back later
        """
        for idx in range(len(self.time_total)):
            json_dict = {
                'date_time': list(self.date_time[idx]),
                'time_total': list(self.time_total[idx]),
                'steps': list(self.steps[idx]),
                'heartrate': list(self.heartrate[idx]),
                'data_id': str(self.data_id),
                'subject_id': str(self.subject_id)
            }

            if self.wake is not None:
                json_dict['wake'] = list(self.wake[idx])

            filename = 'wdap_frag_' + self.subject_id + \
                '_' + str(idx) + '.json'
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(json_dict, f, ensure_ascii=False,
                          indent=4, cls=NpEncoder)

    @classmethod
    def from_json(cls, filename):
        """
            Load data using the format specified above
        """
        jdict = json.load(open(filename, 'r'))
        cls = AppleWatch([], [], [], [])
        for s in jdict.keys():
            if isinstance(jdict[s], list):
                setattr(cls, s, [np.array(jdict[s])])
            else:
                setattr(cls, s, jdict[s])

        return cls
