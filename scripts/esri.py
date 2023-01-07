
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import sys
import torch
from torch import jit
import hcrsimpy
from hcrsimpy.parse import Actiwatch, AppleWatch, ActiwatchReader, AppleWatchReader
from hcrsimpy.models import SinglePopModel
from hcrsimpy.utils import phase_ic_guess
from hcrsimpy.metrics import compactness, compactness_trajectory
from hcrsimpy.utils import simple_norm_stepshr_sleep_classifier

from plots import Actogram
from pytz import timezone

mytz = timezone('EST')


parser = argparse.ArgumentParser(
    description="""Compute the Entrainment Signal Regularity Index for a data set""")

parser.add_argument("-a", "--actogram",
                    required=False,
                    action='store_true',
                    default=False
                    )

parser.add_argument('-j', '--json',
                    required=False,
                    action='store',
                    type=str,
                    help="Specify a json file with the data"
                    )
parser.add_argument('-c', '--csv',
                    required=False,
                    action='store',
                    type=str,
                    help="Specify a directory with csv data in it"
                    )

parser.add_argument('-s', '--sigma',
                    required=False,
                    type=float,
                    action='store',
                    default=0.5,
                    help="Smooth the light(steps) data"
                    )

parser.add_argument('-t1', '--t1',
                    required=False,
                    type=float,
                    action='store',
                    default=None,
                    help="Trim data before this time, in days"
                    )
parser.add_argument('-t2', '--t2',
                    required=False,
                    type=float,
                    action='store',
                    default=None,
                    help="Trim data after this time, in days"
                    )

parser.add_argument("-sm", "--sleepmodel",
                    required=False,
                    action='store',
                    type=str,
                    help="Machine learning pytorch model for steps and heartrate to wake prediction"
                    )

parser.add_argument('-t', '--threshold',
                    required=False,
                    action='store',
                    type=float,
                    help="Threshold for displaying as light on the actogram",
                    default=1.0
                    )


args = parser.parse_args()


def generated_sleep(ml_model_path, steps, hr):
    ml_model = jit.load(ml_model_path)
    data = torch.vstack((torch.tensor(steps), torch.tensor(hr)))
    data = simple_norm_stepshr_sleep_classifier(data).unsqueeze(0).float()
    wake_predicted = torch.sigmoid(ml_model(data)).squeeze(
        0).squeeze(0).detach().numpy()
    len_out = wake_predicted.shape[0]
    wake_predicted = np.hstack((wake_predicted, len(steps) - len_out))
    wake_predicted = np.where(wake_predicted > 0.50, 1.0, 0.0)
    return wake_predicted


reader = AppleWatchReader()
if args.json:
    awObj = reader.read_standard_json(args.json)

if args.csv:
    awObj = reader.read_standard_csv(args.csv)

if args.csv or args.json:
    if args.t1 or args.t2:
        t1 = args.t1 or 0
        t2 = args.t2 or np.floor(np.squeeze(awObj.time_total)[-1]/24.0)+1
        awObj.trim_data(t1*24.0, t2*24.0)
    hr = awObj.heartrate
    ts = awObj.time_total
    steps = awObj.steps

    if args.sleepmodel:
        awObj.wake = generated_sleep(
            ml_model_path=args.sleepmodel, steps=steps, hr=hr)


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


smooth_length = 100
num_days_compactness = 4.5

ts_compact, compactness = compactness_trajectory(
    awObj, gamma=0.0, multiplier=1.0, num_days=num_days_compactness)
compactness_smooth = moving_average(compactness, smooth_length)
plt.figure()
ax1 = plt.gca()
ax1.plot(np.array(compactness)*24.0, np.array(ts_compact) /
         24.0 + num_days_compactness,  alpha=0.50)
ax1.plot(compactness_smooth*24.0,
         np.array(ts_compact[smooth_length-1:])/24.0 + num_days_compactness, color='red')
ax1.set_title(f"ESRI")
ax1.set_ylabel("Days")
ax1.set_xlabel("ESRI Score")
ax1.set_ylim((0, max(np.array(ts_compact) / 24.0)))
ax1.invert_yaxis()

acto = Actogram(np.hstack(ts),
                np.hstack(steps),
                ax=ax1,
                threshold=args.threshold,
                opacity=1.0,
                sigma=[args.sigma, args.sigma])

plt.show()
