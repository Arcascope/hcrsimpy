#! python3

import torch
from torch import jit
from datetime import datetime
import argparse
import numpy as np
from hcrsimpy.plots import Actogram, plot_dashboard
from hcrsimpy.models import SinglePopModel
from hcrsimpy.parse import AppleWatch, AppleWatchReader
from hcrsimpy.utils import phase_ic_guess, cluster_sleep_periods_scipy, sleep_midpoint
from hcrsimpy.utils import simple_norm_stepshr_sleep_classifier
from datetime import datetime
import matplotlib.pylab as plt
from pytz import timezone
mytz = timezone('EST')


parser = argparse.ArgumentParser(description="""Make an actogram""")

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
parser.add_argument("-sm", "--sleepmodel",
                    required=False,
                    action='store',
                    type=str,
                    help="Machine learning pytorch model for steps and heartrate to wake prediction"
                    )
parser.add_argument('-d', '--dlmo',
                    required=False,
                    action='store_true',
                    default=False,
                    help="Integrate the model and plot dlmo times"
                    )
parser.add_argument('-cbt', '--cbt',
                    required=False,
                    action='store_true',
                    default=False,
                    help="Integrate the model and plot core body temperature times"
                    )
parser.add_argument('--sleep',
                    required=False,
                    action='store_true',
                    default=False,
                    help="Add sleep midpoints"
                    )
parser.add_argument('-t', '--threshold',
                    required=False,
                    action='store',
                    type=float,
                    help="Threshold for displaying as light on the actogram",
                    default=1.0
                    )
parser.add_argument('-m', '--multiplier',
                    required=False,
                    action='store',
                    type=float,
                    help="Steps to light multiplier",
                    default=1.0
                    )
parser.add_argument('-p', '--period',
                    required=False,
                    action='store',
                    default=23.84,
                    type=float,
                    help="Set the SPM period"
                    )
parser.add_argument('-r', '--raw',
                    required=False,
                    action='store_true',
                    default=False,
                    help="Plot the raw steps and heartrate data"
                    )
parser.add_argument('--scatter',
                    required=False,
                    action='store_true',
                    default=False,
                    help="Plot the raw steps and heartrate data"
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
parser.add_argument('-s', '--sigma',
                    required=False,
                    type=float,
                    action='store',
                    default=0.5,
                    help="Smooth the light(steps) data"
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

if args.raw:
    awObj.plot()

ic = np.array([0.70, phase_ic_guess(ts[0]), 0.0])

if args.scatter:
    awObj.scatter_hr_steps()


plt.figure()
ax = plt.gca()
acto = Actogram(np.hstack(ts),
                np.hstack(steps),
                ax=ax,
                threshold=args.threshold,
                opacity=1.0,
                sigma=[args.sigma, args.sigma])

if awObj.wake is not None and args.sleep:
    print("Adding sleep layer")
    acto = Actogram(np.hstack(ts),
                    np.hstack(awObj.wake),
                    ax=ax,
                    threshold=args.threshold,
                    smooth=False,
                    opacity=0.50,
                    color='green')

if args.dlmo:
    ic = np.array([0.70, phase_ic_guess(ts[0]), 0.0])
    spm2 = SinglePopModel({'tau': args.period})
    dlmo_runs = spm2.integrate_observer(ts, args.multiplier*steps, ic)
    acto.plot_phasemarker(
        dlmo_runs, error=np.ones(len(dlmo_runs)), color='blue')
    print(f"DLMO mean: {np.mean(np.fmod(dlmo_runs, 24.0))}")
    print("Last 14 days of DLMOs")
    print(np.fmod(dlmo_runs[-14:], 24.0))

if args.cbt:
    ic = np.array([0.70, phase_ic_guess(ts[0]), 0.0])
    spm2 = SinglePopModel({'tau': args.period})
    cbt_runs = spm2.integrate_observer(
        ts, args.multiplier*steps, ic, observer=SinglePopModel.CBTObs)
    acto.plot_phasemarker(
        cbt_runs, error=np.ones(len(cbt_runs)), color='red')
    print(f"CBT mean: {np.mean(np.fmod(cbt_runs, 24.0))}")
    print("Last 14 days of CBTs")
    print(np.fmod(cbt_runs[-14:], 24.0))


# if args.sleep:
#     ts_flat = np.hstack(ts)
#     steps_flat = np.hstack(steps)
#     if awObj.wake is None:
#         spm2 = SinglePopModel({'tau' : args.period})
#         sol= spm2.integrate_model(ts_flat, args.multiplier*steps_flat, phase_ic_guess(ts_flat[0]))
#         wake_score = np.diff(sol[3,:], prepend=0) > 0
#     else:
#         print("Using the provided wake column")
#         wake_score=np.hstack(awObj.wake)

#     sleep_clusters = cluster_sleep_periods_scipy(wake_score,
#                                            90.0,
#                                            makeplot=False)
#     sleep_mid, duration = sleep_midpoint(ts_flat, sleep_clusters)
#     acto.plot_phasemarker(sleep_mid, error=(
#         duration/2.0), color="green", alpha=1.0)
#     print(f"The median sleep duration is {np.median(duration)}")
#     print(f"The average bedtime is {np.mean(np.fmod(sleep_mid-0.5*duration, 24.0))}")
#     print(f"The average waketime is {np.mean(np.fmod(sleep_mid,24.0)+0.5*duration)}")


plt.show()
