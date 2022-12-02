# %%
import numpy as np
import pytz
import datetime
import copy
import json


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def times_to_angle(time_vector: np.ndarray):
    """
        Take an array of times and return R, psi 
        giving the mean angle (psi) and amplitude (R)
    """
    rad_vector = np.fmod(time_vector, 24.0) * np.pi/12.0
    Z = np.sum(np.exp(rad_vector*1j))/len(rad_vector)
    return(np.abs(Z), np.angle(Z))


def timezone_mapper(dt_object: datetime, timezone: str = 'America/Detroit'):
    """
        Take in local time as datetime object and give back UTC with 
        day lights savings accounted for as a timestamp
    """

    local_timezone = pytz.timezone(timezone)
    return local_timezone.localize(dt_object).timestamp()


def split_missing_data(date_time, ts, y, hr=None, break_threshold=96.0):

    # Find idx at start and end of long periods of zeros

    idx_start = None
    idx_end = None
    in_region = False
    crop_regions = []

    for (k, t) in enumerate(ts):
        if y[k] <= 0.0 and not in_region:
            idx_start = k
            in_region = True
        if y[k] > 0.0 and in_region:
            idx_end = k-1
            in_region = False
            if ts[idx_end]-ts[idx_start] >= break_threshold:
                crop_regions += [idx_start, idx_end]
    ts_split = np.split(ts, crop_regions)
    y_split = np.split(y, crop_regions)

    if hr is not None:
        hr_split = np.split(hr, crop_regions)

    print(f"Splitting data into {len(y_split)} regions")

    if hr is not None:
        return np.split(date_time, crop_regions), ts_split, y_split, hr_split
    else:
        return np.split(date_time, crop_regions), ts_split, y_split


def split_drop_data(date_time, ts, steps, hr, wake, break_threshold=96.0, min_length: float = 30.0):
    """
        Used to split long JSON into contin data steaks of at 
        least X=30 days.

        Uses that missing data will be zeros for steps and hr  and 
        0.5 for the wake data. 

        min_length is in days
    """

    idx_start = None
    idx_end = None
    in_region = False
    crop_regions = []

    for (k, t) in enumerate(ts):
        if (steps[k] <= 0.0 or hr[k] <= 0 or wake[k] == 0.50) and not in_region:
            idx_start = k
            in_region = True
        if steps[k] > 0.0 and hr[k] > 0 and wake[k] != 0.50 and in_region:
            idx_end = k-1
            in_region = False
            if ts[idx_end]-ts[idx_start] >= break_threshold:
                crop_regions += [idx_start, idx_end]

    ts_split = np.split(ts, crop_regions)
    steps_split = np.split(steps, crop_regions)
    hr_split = np.split(hr, crop_regions)
    wake_split = np.split(wake, crop_regions)
    date_time = np.split(date_time, crop_regions)

    # Find idxs for regions which are longer than min_length

    idx_long = [k for (k, val) in enumerate(ts_split)
                if (val[-1]-val[0])/24.0 >= min_length]

    if len(idx_long) > 0:
        return ([date_time[i] for i in idx_long], [ts_split[idx] for idx in idx_long], [steps_split[i] for i in idx_long],
                [hr_split[i] for i in idx_long], [wake_split[i] for i in idx_long])
    else:
        return None
