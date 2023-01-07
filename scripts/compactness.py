
import numpy as np
import matplotlib.pyplot as plt 
import argparse 
from pathlib import Path
import sys 

import hcrsimpy 
from hcrsimpy.parse import Actiwatch, AppleWatch, ActiwatchReader, AppleWatchReader
from hcrsimpy.models import SinglePopModel
from hcrsimpy.utils import phase_ic_guess
from hcrsimpy.metrics import compactness, compactness_trajectory

from plots import Actogram



def run_compactness_jenny():
    fig=plt.figure() 
    ax=plt.gca()
    summer=[] 
    school=[]
    reader=ActiwatchReader()
    allsubjects=open(reader.data_directory+"/actiwatch_no_dlmo/jenny_data/DATA/subject_index.txt").read().splitlines() 
    cscores=[] 
    subject_ids=[]
    for subject in allsubjects:
        aw=reader.read_jenny_data(subject_id=subject)
        ts, sol_compactness=compactness(aw, gamma = 0.0, multiplier = 1.0, num_days=4.5)
        if "B" in aw.subject_id:
            ax.plot(ts, sol_compactness[0,:], color='blue', alpha=0.50)
            school.append(sol_compactness[0,-1])
        else:
            ax.plot(ts, sol_compactness[0,:], color='darkgreen', alpha=0.50)
            summer.append(sol_compactness[0,-1])
        print(f"Subject {subject}, Compactness: {sol_compactness[0,-1]}")
        cscores.append(sol_compactness[0,-1]) 
        subject_ids.append(subject)
    
    ax.set_ylim((0,1))
    ax.set_xlabel('Time (Hours)') 
    ax.set_ylabel('Compactness (0-1)')
    print(f"Mean summer is {np.mean(summer)}, mean school is {np.mean(school)}")
    plt.show() 

    data_out=zip(cscores, subject_ids) 
    with open('./data/compactness_results_jenny.csv', "w") as fout:
        fout.write("Subject,Compactness\n")
        for d in data_out:
            fout.write(str(d[1])+","+str(d[0])+"\n") 
        fout.close()
        print("Wrote the file!")
        
        
def run_compactness_phil():
    
    reader = ActiwatchReader()
    awObj = reader.read_phil_data()
    ts, compactness=compactness_trajectory(awObj, gamma = 0.0, multiplier = 1.0, num_days=4.5) 
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(np.array(ts) / 24.0, compactness) 
    ax1.set_title(f"ESRI for {awObj.subject_id}")
    ax1.set_xlabel("Days") 
    ax1.set_ylabel("ESRI Score")
    
    acto = Actogram(awObj.time_total, awObj.lux, second_zeit=awObj.steps, threshold=0.10, threshold2=10.0, ax=ax2)
    
    plt.show()
    
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def run_compactness_longterm():
    
    reader = AppleWatchReader()
    #awObj = reader.read_arcascope_json("/Users/khannay/Desktop/person.json")
    awObj = reader.read_standard_json("/Users/khannay/data/WearableData/LongTermUnlabeled/exporter/kmhAllData.json")
    #awObj = reader.read_json_data("/Users/khannay/Arcascope/Python/TrialDataTools/download/02737ec6ea3d15c132c6d82ccc0fa7c0/combined_data.json")
    ts, compactness=compactness_trajectory(awObj, gamma = 0.0, multiplier = 1.0, num_days=4.5) 
    
    compactness_smooth = moving_average(compactness, 200)  ## moving average
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(compactness, np.array(ts)/ 24.0,  alpha=0.50) 
    ax1.plot(compactness_smooth, np.array(ts[0:len(compactness_smooth)]) / 24.0, color='red') 
    ax1.set_title(f"ESRI")
    ax1.set_ylabel("Days") 
    ax1.set_xlabel("ESRI Score")
    ax1.set_ylim((0, max(np.array(ts)/ 24.0)))
    ax1.invert_yaxis()
    
    acto = Actogram(awObj.time_total, awObj.steps, ax=ax2)
    plt.show()

def run_compactness_hchs():
    fig=plt.figure() 
    ax=plt.gca()
    
    reader=ActiwatchReader()
    allsubjects=open(reader.data_directory+"/actiwatch_no_dlmo/hchs/all_subjects.txt").read().splitlines() 
    cscores=[] 
    subject_ids=[]
    for subject in allsubjects[4:]:
        try:
            aw=reader.read_hchs_data(subject_id=subject)
            ts, sol_compactness=compactness(aw, gamma = 0.0, multiplier = 1.0, num_days=4.5)
            if sol_compactness[0,-1] > 0.0:
                print(f"{subject} has Compactness: {sol_compactness[0,-1]}") 
                cscores.append(sol_compactness[0,-1]) 
                subject_ids.append(subject) 
            
        except:
            print(f"Error with subject {subject}")
        
    #ax.set_ylim((0,1))
    #ax.set_xlabel('Time (Hours)') 
    #ax.set_ylabel('Compactness (0-1)')
    #plt.show() 

    data_out=zip(cscores, subject_ids) 
    with open('./data/compactness_results_hchs.csv', "w") as fout:
        fout.write("Subject,Compactness\n")
        for d in data_out:
            fout.write(str(d[1])+","+str(d[0])+"\n") 
        fout.close()



if __name__ == "__main__":
    #run_compactness_jenny()
    #run_compactness_hchs()

    #run_compactness_phil()
    run_compactness_longterm()