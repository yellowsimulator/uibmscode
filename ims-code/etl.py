"""
Extract data, Transform data and Load
data if neccessary
"""
import os
import pandas as pd
from glob import glob
import scipy.stats
from multiprocessing import Pool

from signal_processing_method import *


def get_date_from_file(file):
    """
    return the date extension from a
    file.
    Argument:
        file: a file name.
    Return:
        datetime of the form yyyy-mm-dd hh:mm:ss
    """
    try:
        datetime = file.split("/")[-1]
        hour = ":".join(datetime.split(".")[3:])
        date = "-".join(datetime.split(".")[:3])
        new_date = "{} {}".format(date,hour)
        return new_date
    except Exception as e:
        print("Error in function 'get_date_from_file': ", e)
        return "error"


def get_all_dates(exp_numb):
    """
    Return all dates from a file.
    Argument:
        exp_numb: the experiment number.
    """
    all_files = get_experiment_data(exp_numb)
    with Pool() as p:
        datetime = p.map(get_date_from_file, all_files)
        return datetime


def get_all_data(exp_numb):
    all_files = get_experiment_data(exp_numb)
    with Pool() as p:
        data = p.map(get_dataframe, all_files)
        return data


def get_all_iqrs(files):
    all_data = []
    with Pool() as p:
        iqrs = p.map(get_iqr, all_data)


def get_iqr(time_series):
    """
    Return the inter quantile range.
    If an error occurs, the string
    "erro will be return".
    Argument:
        time_series: the time series data
    Return:
        iqr
    """
    try:
        iqr = scipy.stats.iqr(time_series)
        return iqr
    except Exception as e:
        print("Error in function 'get_iqr': ", e)
        return "error"


def get_all_faultes(exp_numb, channel):
    all_amps = []
    all_data = []
    fault_freqs = [236.4, 296.8, 280.4]
    all_files = get_experiment_data(exp_numb)
    all_dates = get_all_dates(exp_numb)
    #with Pool() as p
        #all_data = [p.apply(get_experiment_and_channel_data, args=(channel, file)) for file in all_files]
    for file in all_files:
        data = get_experiment_and_channel_data(channel,file)
        all_data.append(data)
    for k, data in enumerate(all_data):
        print(" processing data #{} for channel {}".format(k+1, channel))
        amp = get_sum_amps(data,fault_freqs)
        all_amps.append(amp)
    df = pd.DataFrame({"timestamp": all_dates, "fault": all_amps})
    df.to_csv("amp_experiment{}_channel{}.csv".format(exp_numb, channel))
    print("amp_experiment{}_channel{}.csv".format(exp_numb, channel))
    #return all_amps

def get_sum_amps(data,fault_freqs):
    with Pool() as p:
        amps = sum([p.apply(get_fault_frequency, args=(data,freq)) for freq in fault_freqs])
        return amps






def health_index(faults_amps, iqr):
    """
    This function return the health index.
    If an error occured, the string "error"
    will be return
    Arguments:
        faults_amps: a dictionary of bearing faults
                     amplitude of the form {"bpfo":0.4,
                     "bpfi":2., ...}
        iqr: the inter quantile range of a time series
    Return:
        H: the health index of a time series.
    """
    try:
        H = iqr*sum(faults_amps.values())
        return H
    except Exception as e:
        priint("Error in function 'health_index': ", e)
        return "error"


def get_all_files(path):
    """
    This function retirns all files from
    a directory specified by a path.
    If an erro occurred the string "error"
    will be return and an error will be printed.
    Argument:
        path: directory path
    Return:
        files: all files in the directory
        specified by path argument.
    """
    try:
        files = glob("{}/*".format(path))
        return files
    except Exception as e:
        print("Error in function 'get_all_files': ", e)
        return "error"


def get_dataframe(path):
    """
    Return a dataframe from.
    If an erro occurs return the
    string "error".
    Argument:
        path: file path
    Return:
        dataframe: a pandas dataframe
    """
    try:
        dataframe = pd.read_csv(path,header=None,delim_whitespace=True)
        return dataframe
    except Exception as e:
        print("Error in function 'get_dataframe': ", e)
        return "error"


def get_experiment_data(exp_numb):
    """
    Return all file for a given experiment.
    Arguments:
        exp_numb: experiment number: intenger 1,2
        or 3
    Return:
        all_files: all files
    """
    experiments = {"1": "1st_test",
                   "2": "2nd_test",
                   "3": "3rd_test"}
    experiment = experiments["{}".format(exp_numb)]

    main_path = os.path.abspath("../data/IMS/{}".format(experiment))
    try:
        all_files = get_all_files(main_path)

        return all_files
    except Exception as e:
        print("Error in function 'get_experiment_data': ", e)
        return "error"


def get_bearing_data(file):
    """
    Return data for a bearing.
    channel = 0,1,3,4 for bearing 1,2,3,4
    or channel = 0,1 bearing 1 axial and radial
       channel = 2,3, bearing 2 axial and radial
    similarly for bearing 3 and 4.
    Argument
        file: path to file
        channel: channel number
    """
    channel = 0 # channel must be an input argument to the function
    data_frame = get_dataframe(file)
    try:
        return data_frame[channel].values
    except Exception as e:
        print("Error in function 'get_bearing_data': ", e)
        return "error"



def generate_iqr_csv(channels,exp_numb):
    for channel in channels:
        print("processing channel {}".format(channel+1))
        all_dates = get_all_dates(exp_numb)
        all_files = get_experiment_data(exp_numb)
        for file in all_files:
            data = get_dataframe(file)
        with Pool() as p:
            data = p.map(get_bearing_data, all_files)
            iqrs = p.map(get_iqr, data)
            df = pd.DataFrame({"timestamp": all_dates,"iqr": iqrs})
            df.to_csv("iqr_experiment{}_channel{}.csv".format(exp_numb,
                       channel+1))


def get_experiment_and_channel_data(channel, file):
    data_frame = get_dataframe(file)
    data = data_frame[channel].values
    return data


def main():
    return -1



if __name__ == '__main__':
    fault_freqs = [236.4, 296.8, 280.4]
    exp_numb = 2
    channel = 0
    for channel in [0,1,2,3]:
        get_all_faultes(exp_numb, channel)

    #get_all_faultes(exp_numb, channel)
    #get_experiment_and_channel_data(exp_numb, channel)
