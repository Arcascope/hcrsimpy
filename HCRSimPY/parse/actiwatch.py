# %%

import os
from .utils import *
from scipy.ndimage import gaussian_filter1d
import datetime
from datetime import datetime
from os import read
import numpy as np
import pandas as pd
import pylab as plt
from dataclasses import dataclass
pd.options.mode.chained_assignment = None


@dataclass
class Actiwatch:
    date_time: np.ndarray
    time_total: np.ndarray
    lux: np.ndarray
    steps: np.ndarray
    wake: np.ndarray = None
    phase_measure: np.ndarray = None
    phase_measure_times: np.ndarray = None
    subject_id: str = "Anon"
    data_id: str = "None"

    def get_light(self, multiplier: float = 1.0):
        self.lux = np.absolute(np.array(self.lux))
        return lambda x: np.interp(x, self.time_total, multiplier*self.lux, left=0.0, right=0.0)

    def get_activity_light(self, multiplier: float = 1.0):
        self.steps = np.array(self.steps)
        return lambda x: np.interp(x, self.time_total, multiplier*self.steps, left=0.0, right=0.0)

    def to_dataframe(self):
        """
            Get a pandas dataframe verison of the actiwatch data 
        """
        df = pd.DataFrame({'date_time': self.date_time,
                           'time_total': self.time_total,
                          'lux': self.lux,
                           'steps': self.steps,
                           'wake': self.wake
                           })
        return df

    def get_bounds(self):
        return (self.time_total[0], self.time_total[-1])

    def plot(self, show=True, vlines=None, *args, **kwargs):

        fig = plt.figure()
        gs = fig.add_gridspec(2, hspace=0)
        ax = gs.subplots(sharex=True)
        fig.suptitle(
            f"{self.data_id} Actiwatch Data: Subject {self.subject_id}")
        ax[0].plot(self.time_total / 24.0, np.log10(self.lux+1), color='red')
        ax[1].plot(self.time_total / 24.0, self.steps, color='darkgreen')
        print(np.median(self.steps))
        print(self.wake)
        try:
            ax[1].plot(self.time_total / 24.0, self.wake *
                       np.median(self.steps), color='k')
        except:
            print(f"Error with wake plot with {self.subject_id}")

        if self.phase_measure_times is not None:
            [ax[0].axvline(x=_x / 24.0, ls='--', color='blue')
             for _x in self.phase_measure_times]
            [ax[1].axvline(x=_x / 24.0, ls='--', color='blue')
             for _x in self.phase_measure_times]

        if vlines is not None:
            [ax[0].axvline(x=_x / 24.0, ls='--', color='cyan')
             for _x in vlines]
            [ax[1].axvline(x=_x / 24.0, ls='--', color='cyan')
             for _x in vlines]

        ax[1].set_xlabel("Days")
        ax[0].set_ylabel("Lux (log 10)")
        ax[1].set_ylabel("Activity Counts")
        ax[0].grid()
        ax[1].grid()
        if show:
            plt.show()
        else:
            return ax
