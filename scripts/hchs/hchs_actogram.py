from __future__ import print_function
import random
import os
from joblib import Parallel, delayed
from HCRSimPY.plots import actogram

import sys
from builtins import str
from math import *

import numpy as np
import pandas as pd
import pylab as plt
import scipy as sp
from HCRSimPY import light_schedules
from HCRSimPY.utils import circular_stats
from HCRSimPY.plots import latexify 
from scipy import interpolate
from HCRSimPY.models import *

latexify()


# LOCATION OF HCHS DATA FILES, UPDATE THIS FOR YOUR SYSTEM

hchs_files_location = "../../HumanData/HCHS/hchs-sol-sueno-"
hchs_files_directory = "../../HumanData/HCHS/"


def findKeyDLMOTimes(tsdf):
    """Find the DLMO and CBT times for a given time series prediction"""

    wrapped_time = np.round([fmod(x, 24.0) for x in list(tsdf.Time)], 2)
    df = pd.DataFrame({'Time': wrapped_time, 'Phase': tsdf.Phase})
    df2 = df.groupby('Time')['Phase'].agg(
        {'Circular_Mean': circular_mean, 'Phase_Coherence': phase_coherence, 'Samples': np.size})
    mean_func = sp.interpolate.interp1d(
        np.array(
            df2['Circular_Mean']), np.array(
            df2.index))
    coherence_func = sp.interpolate.interp1d(
        np.array(
            df2['Circular_Mean']), np.array(
            df2['Phase_Coherence']))
    return ((mean_func(1.309), coherence_func(1.309)))


def record_diff(tsdfS, tsdfV, tsdfT):
    """Find the differences in the DLMO timing of the three models for that given light schedule"""

    d1, r1 = findKeyDLMOTimes(tsdfS)
    d2, r2 = findKeyDLMOTimes(tsdfV)
    d3, r3 = findKeyDLMOTimes(tsdfT)

    return ((d1, d2, d3, r1, r2, r3))


def get_diff(f):
    """Used to find the average DLMO times of all data in the hchs data set, in a parallel fashion"""
    fnum = f.split('-')[-1].split('.')[0]

    trans_days = 50
    hc = hchs_light(f)
    sm = hc.findMidSleep()
    init = guessICData(hc.LightFunctionInitial, 0.0, length=trans_days)
    initVDP = guessICDataVDP(hc.LightFunctionInitial, 0.0, length=trans_days)
    initTwo = guessICDataTwoPop(
        hc.LightFunctionInitial, 0.0, length=trans_days)

    a = SinglePopModel(hc.LightFunctionInitial)
    b = vdp_model(hc.LightFunctionInitial)
    c = TwoPopModel(hc.LightFunctionInitial)
    ent_angle = a.integrateModelData((0.0, 40.0 * 24.0), initial=init)
    ent_angle_vdp = b.integrateModelData((0.0, 40.0 * 24.0), initial=initVDP)
    ent_angle_two = c.integrateModelData((0.0, 40.0 * 24.0), initial=initTwo)
    tsdf = a.getTS()
    tsdf_vdp = b.getTS()
    tsdf_two = c.getTS()
    fnum = f.split('-')[-1].split('.')[0]
    d1, d2, d3, r1, r2, r3 = record_diff(tsdf, tsdf_vdp, tsdf_two)
    return (
        fnum +
        ", " +
        str(d1) +
        ", " +
        str(d2) +
        ", " +
        str(d3) +
        ", " +
        str(r1) +
        ", " +
        str(r2) +
        ", " +
        str(r3) +
        ", " +
        str(sm) +
        "\n")


def get_all_hchs_files():
    """ Get a list of every hchs file """

    file_list = []

    for file in os.listdir(hchs_files_directory):
        if file.endswith(".csv"):
            file_list.append(str(os.path.join(hchs_files_directory, file)))

    return (file_list)


def runParticularData(filenumber, trans_days=50):
    """Given a id number run that system"""
    fileName = hchs_files_location + str(filenumber) + ".csv"

    hc = hchs_light(fileName)

    init = guessICData(hc.LightFunctionInitial, 0.0, length=trans_days)
    initVDP = guessICDataVDP(hc.LightFunctionInitial, 0.0, length=trans_days)
    initTwo = guessICDataTwoPop(
        hc.LightFunctionInitial, 0.0, length=trans_days)

    a = SinglePopModel(hc.LightFunctionInitial)
    b = vdp_model(hc.LightFunctionInitial)
    c = TwoPopModel(hc.LightFunctionInitial)
    ent_angle = a.integrateModelData((0.0, 40.0 * 24.0), initial=init)
    ent_angle_vdp = b.integrateModelData((0.0, 40.0 * 24.0), initial=initVDP)
    ent_angle_two = c.integrateModelData((0.0, 40.0 * 24.0), initial=initTwo)
    tsdf = a.getTS()
    tsdf_vdp = b.getTS()
    tsdf_two = c.getTS()
    plt.figure()
    ax = plt.gca()
    acto = actogram(ax, tsdf)  # add an actogram to those axes
    acto.addCircadianPhases(tsdf_vdp, col='darkgreen')
    acto.addCircadianPhases(tsdf_two, col='red')
    ax.set_title('HCHS Actogram')
    plt.show()

    print(("Filenumber: ", filenumber))


def runAllFilesDLMO():
    """
       Run through all the files and measure the DLMO predicted differences for the three models.

       This should run in parallel as it will take a file to run. This generates the hchs_model_diff file needed by the
       compare_hchs_manuscript program.

    """
    fl = get_all_hchs_files()
    print(("Total Files: ", len(fl)))
    allOutputs = Parallel(n_jobs=32)(delayed(get_diff)(f) for f in fl)

    outfile = open('hchs_model_diff.csv', 'w')
    outfile.write(
        'Filename, SP_DLMO, VDP_DLMO, TP_DLMO, R_SP, R_VDP, R_TP, Est_Sleep_MP\n')

    for o in allOutputs:
        outfile.write(o)


def plotRandomSP(trans_days=100):
    """
    This plots are randomly chosen hchs light schedule with the single population model predictions for DLMO and CBT min
    """

    fl = get_all_hchs_files()
    fileName = random.choice(fl)
    hc = hchs_light(fileName)

    init = guessICData(hc.LightFunctionInitial, 0.0, length=trans_days)

    a = SinglePopModel(hc.LightFunctionInitial)

    ent_angle = a.integrateModelData((0.0, 10.0 * 24.0), initial=init)
    tsdf = a.getTS()
    plt.figure()
    ax = plt.gca()
    acto = actogram(ax, tsdf)  # add an actogram to those axes

    ax.set_title('HCHS Actogram')
    plt.tight_layout()
    plt.savefig('Modern_Light_actogram.eps')
    plt.show()

    print(("Filename chosen: ", fileName))


def chooseRandomData(trans_days=50):
    """ Get a randomly chosen hchs participant """

    fl = get_all_hchs_files()
    fileName = random.choice(fl)
    hc = hchs_light(fileName)

    init = guessICData(hc.LightFunctionInitial, 0.0, length=trans_days)
    initVDP = guessICDataVDP(hc.LightFunctionInitial, 0.0, length=trans_days)
    initTwo = guessICDataTwoPop(
        hc.LightFunctionInitial, 0.0, length=trans_days)

    a = SinglePopModel(hc.LightFunctionInitial)
    b = vdp_model(hc.LightFunctionInitial)
    c = TwoPopModel(hc.LightFunctionInitial)
    ent_angle = a.integrateModelData((0.0, 40.0 * 24.0), initial=init)
    ent_angle_vdp = b.integrateModelData((0.0, 40.0 * 24.0), initial=initVDP)
    ent_angle_two = c.integrateModelData((0.0, 40.0 * 24.0), initial=initTwo)
    tsdf = a.getTS()
    tsdf_vdp = b.getTS()
    tsdf_two = c.getTS()
    d1, d2, d3 = record_diff(tsdf, tsdf_vdp, tsdf_two)

    plt.figure()
    ax = plt.gca()
    acto = actogram(ax, tsdf)  # add an actogram to those axes
    acto.addCircadianPhases(tsdf_vdp, col='darkgreen')
    acto.addCircadianPhases(tsdf_two, col='red')
    ax.set_title('HCHS Actogram')
    plt.show()

    print(("Filename Chosen: ", fileName))


def runAllShiftWorkers():
    """
        Run through all the shift workers in the data set and makes an actogram for their recorded light schedules
    """
    swlist = open('HCHS_ShiftWorkers_PID.csv').readlines()[1:]

    swlist = [s.strip().strip("\"") for s in swlist]

    allOutputs = Parallel(
        n_jobs=32)(
        delayed(chooseShiftWorker)(f) for f in swlist)

    print(("Total Number of files generated: ", sum(allOutputs)))


def chooseShiftWorker(sw, trans_days=50.0):
    """Choose a shift worker and make an actogram"""

    try:
        fileName = hchs_files_location + sw + ".csv"

        hc = hchs_light(fileName)

        init = guessICData(hc.LightFunctionInitial, 0.0, length=trans_days)

        a = SinglePopModel(hc.LightFunctionInitial)

        ent_angle = a.integrateModelData((0.0, 10.0 * 24.0), initial=init)
        tsdf = a.getTS()
        plt.figure()
        ax = plt.gca()
        acto = actogram(ax, tsdf)  # add an actogram to those axes
        dlmo = acto.getDLMOtimes()
        # print "Phase Coherence DLMO: ", phase_coherence_clock(dlmo)
        saveString = sw + "\t" + str(phase_coherence_clock(dlmo))
        print(saveString)

        mytitle = 'HCHS Actogram ' + sw
        ax.set_title(mytitle)
        plt.tight_layout()
        figname = "ShiftWorkerPlots/sw_actogram_" + sw + ".eps"
        plt.savefig(figname)
        return (1)
    except BaseException:
        return (0)


if __name__ == '__main__':

    # runAllShiftWorkers()

    runAllFilesDLMO()

    sys.exit(0)

    if len(sys.argv) < 2:
        plotRandomSP()
    else:
        if len(sys.argv) == 2:
            runParticularData(sys.argv[1])
        else:
            runParticularData(sys.argv[1], int(sys.argv[2]))


