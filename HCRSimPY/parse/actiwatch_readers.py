

import glob
import os
import random
import difflib
from .utils import *
from scipy.ndimage import gaussian_filter1d
from .actiwatch import *
import datetime
from datetime import datetime
from os import read
import numpy as np
import pandas as pd
import pylab as plt
from dataclasses import dataclass
pd.options.mode.chained_assignment = None


WDATA = os.environ.get("WDATA")


class ActiwatchReader(object):

    def __init__(self, data_directory=None):

        if data_directory is None and WDATA is not None:
            self.data_directory = WDATA
        else:
            self.data_directory = data_directory

    def utc_to_hrs(self, d: datetime):
        return d.hour+d.minute/60.0+d.second/3600.0

    def find_header_ddd(self, filename: str, section_word: str):

        f = open(filename, 'r').readlines()
        found_epoch = False
        counter = 0
        for line in f:
            if section_word in line:
                found_epoch = True
            if (section_word and "Off-Wrist" in line and "Red Light" in line):
                return counter
            counter += 1

        return 0

    def process_actiwatch(self, df: pd.DataFrame,
                          MIN_LIGHT_THRESHOLD=5000,
                          round_data=True,
                          bin_minutes=6,
                          dt_format: str = "%m/%d/%Y %I:%M:%S %p"):
        """
            Takes in a dataframe with columns 
                Date : str 
                Time : str 
                White Light: float 
                Sleep/Wake: float 
                Activity: float
            returns a cleaned dataframe with columns
                "DateTime", "TimeTotal", "UnixTime", "Activity", "Lux", "Wake"
        """
        df['DateTime'] = df['Date']+" "+df['Time']
        df['DateTime'] = pd.to_datetime(
            df.DateTime, format=dt_format)

        df['UnixTime'] = (
            df['DateTime'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        df['Lux'] = df['White Light']
        df.rename(columns={'Sleep/Wake': 'Wake'}, inplace=True)

        df['Lux'].fillna(0, inplace=True)
        df['LightSum'] = np.cumsum(df.Lux.values)
        df['LightSumReverse'] = np.sum(
            df.Lux.values) - np.cumsum(df.Lux.values) + 1.0

        df = df[(df.LightSum > MIN_LIGHT_THRESHOLD) & (
            df.LightSumReverse > MIN_LIGHT_THRESHOLD)]

        time_start = self.utc_to_hrs(df.DateTime.iloc[0])
        df2 = df[['UnixTime']].copy(deep=True)
        base_unix_time = df2['UnixTime'].iloc[0]
        df['TimeTotal'] = time_start + \
            (df2.loc[:, ['UnixTime']]-base_unix_time)/3600.0

        df = df[["DateTime", "TimeTotal", "UnixTime", "Activity", "Lux", "Wake"]]
        if round_data:
            df.set_index('DateTime', inplace=True)
            df = df.resample(str(int(bin_minutes))+'Min').agg({'TimeTotal': 'min',
                                                              'UnixTime': 'min',
                                                               'Activity': 'sum',
                                                               'Lux': 'median',
                                                               'Wake': 'max'})
            df.reset_index(inplace=True)

        # Not sure why hchs needs this
        df['TimeTotal'].interpolate(inplace=True)
        df.fillna(0, inplace=True)
        return df

    def read_phil_data(self,
                       subject_id: str = None,
                       MIN_LIGHT_THRESHOLD: float = 5000.0,
                       round_data: bool = True,
                       bin_minutes: int = 6,
                       ):

        phil_acti_directory = self.data_directory + \
            "/actiwatch_dlmo/phil_shift_worker/actiwatch/"
        #myfiles = glob.glob(phil_acti_directory+"/*.csv")
        myfiles = open(phil_acti_directory+"subject_index.txt").readlines()
        myfiles = [f.strip() for f in myfiles]

        if subject_id is None:
            filename = random.choice(myfiles)
        else:
            filename = difflib.get_close_matches(
                subject_id, myfiles, cutoff=0.0, n=1)[0]
            print(f"Found the file: {filename.split('/')[-1]}")

        subject_id = filename.split("/")[-1].split("-")[0]

        df = pd.read_csv(phil_acti_directory+filename)

        df = self.process_actiwatch(df,
                                    MIN_LIGHT_THRESHOLD=MIN_LIGHT_THRESHOLD,
                                    round_data=round_data,
                                    bin_minutes=bin_minutes)

        # Get the DLMO time
        time_start = self.utc_to_hrs(df.DateTime.iloc[0])

        df_dlmo = pd.read_csv(
            self.data_directory+"/actiwatch_dlmo/phil_shift_worker/dlmo/DLMOCompiled.csv")
        df_dlmo = df_dlmo[["Subject", "DLMO.max_FULL"]]
        df_dlmo.rename(columns={'DLMO.max_FULL': 'DLMO'}, inplace=True)
        df_dlmo.DLMO = pd.to_datetime(df_dlmo.DLMO, format="%m/%d/%Y %H:%M")

        subject_list = df_dlmo.Subject.to_list()

        if 'PH' in subject_id:
            subject_list_filtered = [s for s in subject_list if 'PH' in s]
        else:
            subject_list_filtered = [s for s in subject_list if 'PH' not in s]
        matched_entry = difflib.get_close_matches(
            subject_id, subject_list_filtered, cutoff=0.0, n=3)
        df_dlmo.set_index('Subject', inplace=True)
        dlmo_dt = df_dlmo.loc[matched_entry[0]]['DLMO']

        dlmo_ts = (dlmo_dt - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        phase_measure_time = time_start + (dlmo_ts-df.UnixTime.iloc[0])/3600.0
        # make the file be subject id and unix time stamps

        aw = Actiwatch(date_time=df['UnixTime'].to_numpy(),
                       time_total=df['TimeTotal'].to_numpy(),
                       lux=df['Lux'].to_numpy(),
                       steps=df['Activity'].to_numpy(),
                       wake=df['Wake'].to_numpy(),
                       phase_measure=np.array([5*np.pi/12.0]),
                       phase_measure_times=np.array([phase_measure_time]),
                       subject_id=subject_id,
                       data_id='Phil-SW'
                       )

        return(aw)

    def read_phil_dd(self,
                     subject_id: str = None,
                     MIN_LIGHT_THRESHOLD=5000,
                     round_data=True,
                     bin_minutes=6):

        idx_file = "dd_sw_list.txt"
        possible_ids = open(
            self.data_directory+"/actiwatch_dlmo/phil_double_dlmo_data/DDD/"+idx_file, 'r').readlines()
        possible_ids = [p.strip() for p in possible_ids]

        phil_acti_directory = self.data_directory + \
            "/actiwatch_dlmo/phil_double_dlmo_data/DDD/Actigraphy"
        myfiles = glob.glob(phil_acti_directory+"/*.csv")
        if subject_id is None:
            subject_id = random.choice(possible_ids)
        else:
            subject_id = difflib.get_close_matches(
                subject_id, possible_ids, cutoff=0.0, n=1)[0]
            print(f"Found the subject: {subject_id}")

        f1 = difflib.get_close_matches(
            subject_id+"V1", myfiles, cutoff=0.0, n=1)[0]
        f2 = difflib.get_close_matches(
            subject_id+"V2", myfiles, cutoff=0.0, n=1)[0]

        #print(f"The two files are f{f1} and f{f2}")

        header_idx1 = self.find_header_ddd(f1, "Epoch-by-Epoch")
        header_idx2 = self.find_header_ddd(f2, "Epoch-by-Epoch")

        #print(header_idx1, header_idx2)
        df1 = pd.read_csv(f1, header=header_idx1, skip_blank_lines=False)
        df2 = pd.read_csv(f2, header=header_idx2, skip_blank_lines=False)
        df1 = df1.iloc[1:]
        df2 = df2.iloc[1:]

        df = pd.concat([df1, df2])

        df = self.process_actiwatch(df,
                                    MIN_LIGHT_THRESHOLD=MIN_LIGHT_THRESHOLD,
                                    round_data=round_data,
                                    bin_minutes=bin_minutes)

        # Get the dlmo data
        # Get the DLMO time associated with that
        dlmo_filename = self.data_directory + \
            "/actiwatch_dlmo/phil_double_dlmo_data/DDD/doubleDLMO.csv"

        df_dlmo = pd.read_csv(dlmo_filename)
        df_dlmo.v1Date = pd.to_datetime(df_dlmo['v1Date'], format="%m/%d/%Y")
        df_dlmo.v2Date = pd.to_datetime(df_dlmo['v2Date'], format="%m/%d/%Y")

        df_dlmo['v1Unix'] = (
            df_dlmo['v1Date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        df_dlmo['v2Unix'] = (
            df_dlmo['v2Date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

        df_dlmo['v1Unix'] += df_dlmo['DLMOv1_org']*60*60*24.0
        df_dlmo['v2Unix'] += df_dlmo['DLMOv2_org']*60*60*24.0

        time_start = self.utc_to_hrs(df.DateTime.iloc[0])
        df_dlmo['tt1'] = time_start + \
            (df_dlmo['v1Unix'].to_numpy()-df.UnixTime.iloc[0])/3600.0
        df_dlmo['tt2'] = time_start + \
            (df_dlmo['v2Unix'].to_numpy()-df.UnixTime.iloc[0])/3600.0

        subject_list = df_dlmo.ID.to_list()

        matched_entry = difflib.get_close_matches(
            subject_id, subject_list, cutoff=0.0, n=3)
        df_dlmo.set_index('ID', inplace=True)
        dlmo1 = df_dlmo.loc[matched_entry[0]]['tt1']
        dlmo2 = df_dlmo.loc[matched_entry[0]]['tt2']

        phase_measure_times = np.array([dlmo1, dlmo2])

        aw = Actiwatch(date_time=df['UnixTime'].to_numpy(),
                       time_total=df['TimeTotal'].to_numpy(),
                       lux=df['Lux'].to_numpy(),
                       steps=df['Activity'].to_numpy(),
                       wake=df['Wake'].to_numpy(),
                       phase_measure=np.array([5*np.pi/12.0, 5*np.pi/12.0]),
                       phase_measure_times=phase_measure_times,
                       subject_id=subject_id,
                       data_id='Phil-DD'
                       )

        return(aw)

    def read_jenny_data(self,
                        subject_id: str = None,
                        MIN_LIGHT_THRESHOLD=5000,
                        round_data=True,
                        bin_minutes=6):

        jenny_acti_directory = self.data_directory+"/actiwatch_no_dlmo/jenny_data/DATA"
        myfiles = glob.glob(jenny_acti_directory+"/*.csv")

        if subject_id is None:
            filename = random.choice(myfiles)
        else:
            filename = difflib.get_close_matches(
                subject_id, myfiles, cutoff=0.0, n=1)[0]

        #print(f"Found the file: {filename.split('/')[-1]}")

        subject_id = filename.split("/")[-1].split("_")[0]

        df = pd.read_csv(filename, header=2)

        # Special case! File format diff
        if "10067" in subject_id:
            df = pd.read_csv(filename, header=3)
        df.rename(columns={
                  'Steps': 'Activity', 'Sleep or Awake?': 'Sleep/Wake', 'Lux': 'White Light'}, inplace=True)
        df = self.process_actiwatch(df,
                                    MIN_LIGHT_THRESHOLD=MIN_LIGHT_THRESHOLD,
                                    round_data=round_data,
                                    bin_minutes=bin_minutes,
                                    dt_format="%m/%d/%Y %I:%M %p"
                                    )
        df['Wake'].replace({'W': 1.0, 'S': 0.0}, inplace=True)

        aw = Actiwatch(date_time=df['UnixTime'].to_numpy(),
                       time_total=df['TimeTotal'].to_numpy(),
                       lux=df['Lux'].to_numpy(),
                       steps=df['Activity'].to_numpy(),
                       wake=df['Wake'].to_numpy(),
                       subject_id=subject_id,
                       data_id='Jenny'
                       )

        return(aw)

    def get_all_hchs_subjects(self):
        hchs_acti_directory = self.data_directory+"/actiwatch_no_dlmo/hchs/actigraphy/"
        myfiles = glob.glob(hchs_acti_directory+"/*.csv")
        return myfiles

    def read_hchs_data(self,
                       subject_id: str = None,
                       MIN_LIGHT_THRESHOLD=5000,
                       round_data=True,
                       bin_minutes=6):

        hchs_acti_directory = self.data_directory+"/actiwatch_no_dlmo/hchs/actigraphy/"
        myfiles = glob.glob(hchs_acti_directory+"/*.csv")

        if subject_id is None:
            filename = random.choice(myfiles)
        else:
            filename = difflib.get_close_matches(
                subject_id, myfiles, cutoff=0.0, n=1)[0]

        #print(f"Found the file: {filename.split('/')[-1]}")

        subject_id = filename.split("/")[-1].split("-")[-1]
        subject_id = subject_id.split(".")[0]

        df = pd.read_csv(filename)
        df.rename(columns={'time': 'Time', 'activity': 'Activity',
                  'wake': 'Sleep/Wake', 'whitelight': 'White Light'}, inplace=True)
        df['Date'] = [datetime(2020, 1, x+1).strftime("%m/%d/%Y")
                      for x in df['day'].to_list()]

        df = self.process_actiwatch(df,
                                    MIN_LIGHT_THRESHOLD=MIN_LIGHT_THRESHOLD,
                                    round_data=round_data,
                                    bin_minutes=bin_minutes,
                                    dt_format="%m/%d/%Y %H:%M:%S"
                                    )
        df['Wake'].replace({'W': 1.0, 'S': 0.0}, inplace=True)

        aw = Actiwatch(date_time=df['UnixTime'].to_numpy(),
                       time_total=df['TimeTotal'].to_numpy(),
                       lux=df['Lux'].to_numpy(),
                       steps=df['Activity'].to_numpy(),
                       wake=df['Wake'].to_numpy(),
                       subject_id=subject_id,
                       data_id='HCHS'
                       )

        return(aw)

    def get_all_jenny_dlmo_subjects(self):
        jenny_acti_directory = self.data_directory + "/actiwatch_dlmo/jenny_dlmo/"
        subjects = open(jenny_acti_directory +
                        "subject_list_dlmo.txt", 'r').readlines()
        subjects = [f.strip() for f in subjects]
        return subjects

    def read_jenny_dlmo(self,
                        subject_id: str = None,
                        MIN_LIGHT_THRESHOLD=5000,
                        round_data=True,
                        bin_minutes=6):

        jenny_acti_directory = self.data_directory + "/actiwatch_dlmo/jenny_dlmo/"
        subjects = open(jenny_acti_directory +
                        "subject_list_dlmo.txt", 'r').readlines()
        subjects = [f.strip() for f in subjects]

        if subject_id is None:
            subject_id = random.choice(subjects)
        else:
            subject_id = difflib.get_close_matches(
                subject_id, subjects, cutoff=0.0, n=1)[0]

        myfiles = glob.glob(jenny_acti_directory+"actigraphy/*.csv")

        filename = difflib.get_close_matches(
            subject_id, myfiles, cutoff=0.0, n=1)[0]
        #print(f"Found the subject: {subject_id} with {filename.split('/')[-1]}")
        df = pd.read_csv(filename, skiprows=3)
        df.rename(columns={
                  'Steps': 'Activity', 'Sleep or Awake?': 'Sleep/Wake', 'Lux': 'White Light'}, inplace=True)
        df = self.process_actiwatch(df,
                                    MIN_LIGHT_THRESHOLD=MIN_LIGHT_THRESHOLD,
                                    round_data=round_data,
                                    bin_minutes=bin_minutes,
                                    dt_format="%m/%d/%Y %I:%M %p"
                                    )
        df['Wake'].replace({'W': 1.0, 'S': 0.0}, inplace=True)

        # Get the DLMO time, need to calculate the DLMO times
        dlmo_filename = jenny_acti_directory + "DLMO_iheartrhythms.csv"
        df_dlmo = pd.read_csv(dlmo_filename, dtype={'Subject': str})
        df_dlmo.Date = pd.to_datetime(df_dlmo['Date'], format="%m/%d/%y")

        df_dlmo['Unix'] = (
            df_dlmo['Date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

        df_dlmo['Timestamp'] = df_dlmo['Unix']+df_dlmo['Time']*3600.0

        time_start = self.utc_to_hrs(df.DateTime.iloc[0])

        df_dlmo['tt1'] = df.TimeTotal.iloc[0] + \
            (df_dlmo['Timestamp']-df.UnixTime.iloc[0])/3600.0
        subject_list = df_dlmo.Subject.to_list()

        matched_entry = difflib.get_close_matches(
            subject_id, subject_list, cutoff=0.0, n=3)
        df_dlmo.set_index('Subject', inplace=True)
        dlmo = df_dlmo.loc[matched_entry[0]]['tt1']
        phase_measure_times = np.array([dlmo])

        aw = Actiwatch(date_time=df['UnixTime'].to_numpy(),
                       time_total=df['TimeTotal'].to_numpy(),
                       lux=df['Lux'].to_numpy(),
                       steps=df['Activity'].to_numpy(),
                       wake=df['Wake'].to_numpy(),
                       subject_id=subject_id,
                       phase_measure_times=phase_measure_times,
                       phase_measure=np.array([5*np.pi/12.0]),
                       data_id='Jenny-DLMO',
                       )

        return(aw)
