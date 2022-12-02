
from hcrsimpy.parse.applewatch import *
import random
import difflib
from hcrsimpy.parse.utils import timezone_mapper, split_drop_data

WDATA = os.environ.get("WDATA")


class AppleWatchReader(object):

    def __init__(self, data_directory=None):
        if data_directory is None and WDATA is not None:
            self.data_directory = WDATA
        else:
            self.data_directory = data_directory

    def utc_to_hrs(self, d: datetime):
        return d.hour+d.minute/60.0+d.second/3600.0

    def process_applewatch_pandas(self,
                                  steps: pd.DataFrame,
                                  heartrate: pd.DataFrame,
                                  wake: pd.DataFrame = None,
                                  bin_minutes: int = 6,
                                  debug=False,
                                  inner_join: bool = False
                                  ):

        if debug:
            mean_hr_ts = np.median(
                np.diff(np.sort(heartrate['Time'].to_numpy()))/60.0)
            print(f"The median diff in hr timestamps is: {mean_hr_ts} minutes")

        steps['Time_Start'] = pd.to_datetime(steps.Time_Start, unit='s')
        steps['Time_End'] = pd.to_datetime(steps.Time_End, unit='s')

        s1 = steps.loc[:, ['Time_Start', "Steps"]]
        s2 = steps.loc[:, ['Time_End', 'Steps']]
        s1.rename(columns={'Time_Start': 'Time'}, inplace=True)
        s2.rename(columns={'Time_End': 'Time'}, inplace=True)
        steps = pd.concat([s1, s2])
        steps.set_index('Time', inplace=True)
        steps = steps.resample(str(int(bin_minutes)) +
                               'Min').agg({'Steps': 'sum'})
        steps.reset_index(inplace=True)

        if 'Time' not in heartrate:
            heartrate['Time'] = steps['Time']
            heartrate['HR'] = np.zeros(len(steps['Time']))
        heartrate['Time'] = pd.to_datetime(heartrate.Time, unit='s')
        heartrate.set_index('Time', inplace=True)
        heartrate = heartrate.resample(
            str(int(bin_minutes))+'Min').agg({'HR': 'max'})
        heartrate.reset_index(inplace=True)

        merge_method = 'inner' if inner_join else 'left'
        df = pd.merge(steps, heartrate, on='Time', how=merge_method)
        df = df.fillna(0)

        if wake is not None:
            wake['Time'] = pd.to_datetime(wake.timestamp, unit='s')
            wake.set_index('Time', inplace=True)
            wake = wake.resample(
                str(int(bin_minutes))+'Min').agg({'wake': 'max'})
            wake.reset_index(inplace=True)
            if inner_join:
                df = pd.merge(df, wake, on='Time', how='inner')
            else:
                df = pd.merge(df, wake, on='Time', how='left')
            df.rename(columns={'wake': 'Wake'}, inplace=True)
        else:
            print("No sleep data found")

        df['UnixTime'] = (
            df['Time'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

        time_start = self.utc_to_hrs(df.Time.iloc[0])
        df['TimeTotal'] = time_start + (df.UnixTime-df.UnixTime.iloc[0])/3600.0
        df['DateTime'] = pd.to_datetime(df['UnixTime'], unit='s')
        if wake is not None:
            return df[['DateTime', 'UnixTime', 'TimeTotal', 'Steps', 'HR', 'Wake']]
        else:
            return df[['DateTime', 'UnixTime', 'TimeTotal', 'Steps', 'HR']]

    

    def read_standard_csv(self,
                           directory_path: str,
                           bin_minutes: int = 6,
                           subject_id="",
                           data_id="Exporter"
                           ):

        steps = pd.read_csv(directory_path+"/combined_steps.csv")
        hr = pd.read_csv(directory_path+"/combined_heartrate.csv")

        steps.rename(columns={steps.columns[0]: 'Time_Start',
                     steps.columns[1]: 'Time_End', steps.columns[2]: 'Steps'}, inplace=True)
        hr.rename(columns={hr.columns[0]: 'Time', hr.columns[1]: 'HR'}, inplace=True)

        df = self.process_applewatch_pandas(steps, hr, bin_minutes=bin_minutes)

        aw = AppleWatch(date_time=df.UnixTime.to_numpy(),
                        time_total=df.TimeTotal.to_numpy(),
                        heartrate=df.HR.to_numpy(),
                        steps=df.Steps.to_numpy(),
                        subject_id=subject_id,
                        data_id=data_id
                        )

        return aw

    def read_standard_json(self, filepath: str,
                            bin_minutes: int = 6,
                            subject_id="",
                            data_id="Exporter",
                            sleep_trim: bool = False,
                            gzip_opt: bool = False
                            ):

        gzip_opt = gzip_opt if gzip_opt else filepath.endswith(".gz")
        fileobj = gzip.open(filepath, 'r') if gzip_opt else open(filepath, 'r')
        rawJson = json.load(fileobj)

        if 'wearables' in rawJson.keys():
            rawJson = rawJson['wearables']

        sleep_wake_upper_cutoff_hrs = 18.0
        wake_period_minimum_cutoff_seconds = 360.0
        sleep_interp_bin_secs = 60.0

        steps = pd.DataFrame(rawJson['steps'])
        hr = pd.DataFrame(rawJson['heartrate'])

        try:
            sleepList = [(s['interval']['start'], s['interval']['duration'], 0.0)
                         for s in rawJson['sleep'] if 'sleep' in s['value'].keys()]
        except:
            sleepList = [(s['interval']['start'], s['interval']['duration'], 0.0)
                         for s in rawJson['sleep'] if s['value'] == 'sleep']
        sleepList.sort(key=lambda x: x[0])
        for idx in range(1, len(sleepList)):
            last_sleep_end = sleepList[idx-1][0] + sleepList[idx-1][1]
            next_sleep_start = sleepList[idx][0]
            wake_duration = next_sleep_start - last_sleep_end
            if wake_duration < sleep_wake_upper_cutoff_hrs*3600.0 and wake_duration >= wake_period_minimum_cutoff_seconds:
                sleepList.append(
                    (last_sleep_end + 1.0, wake_duration - 1.0, 1.0))

        flattenSleepWakeList = []
        for s in sleepList:
            end_time = s[0] + s[1]
            last_time = s[0]

            while last_time + sleep_interp_bin_secs < end_time:
                flattenSleepWakeList.append(
                    {'timestamp': last_time, 'wake': s[2]})
                last_time += sleep_interp_bin_secs

        wake = pd.DataFrame(flattenSleepWakeList)

        # Some older files use stop instead of end
        endStepPeriodName = 'end'
        if 'stop' in steps.columns:
            endStepPeriodName = 'stop'

        steps.rename(columns={'start': 'Time_Start',
                     endStepPeriodName: 'Time_End', 'steps': 'Steps'},
                     inplace=True)

        hr.rename(columns={'timestamp': 'Time',
                           'heartrate': 'HR'}, inplace=True)

        df = self.process_applewatch_pandas(
            steps, hr, wake=wake, bin_minutes=bin_minutes, inner_join=True)

        if sleep_trim:
            df = df.dropna(subset=['Wake'])
        aw = AppleWatch(date_time=df.UnixTime.to_numpy(),
                        time_total=df.TimeTotal.to_numpy(),
                        heartrate=df.HR.to_numpy(),
                        steps=df.Steps.to_numpy(),
                        wake=df.Wake.to_numpy(),
                        subject_id=subject_id,
                        data_id=data_id
                        )

        return aw

    