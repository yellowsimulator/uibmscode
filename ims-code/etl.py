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
from wavelet_transform import *


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


def get_all_files(path):
    """
    Return all files from a directory specified by a path.
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


def get_dataframe(path):
    """
    Return a dataframe from.
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


def get_experiment_data(exp_numb):
    """
    Return all file for a given experiment.
    Arguments:
        exp_numb: experiment number: intenger 1,2or 3
    Return:
        all_files: all files
    """
    experiments = {"1": "1st_test", "2": "2nd_test", "3": "3rd_test"}
    experiment = experiments["{}".format(exp_numb)]
    main_path = os.path.abspath("../data/{}".format(experiment))
    try:
        all_files = get_all_files(main_path)
        return all_files
    except Exception as e:
        print("Error in function 'get_experiment_data': ", e)


def get_all_data_frames(exp_numb):
    """
    Return all dataframes for a given experiment.
    Argument:
    --------
    exp_numb
        experiment number (1,2 or 3)
    Return:
    ------
    all_data_frames:
        all dataframes pertaining to an experiment
    """
    try:
        all_paths = get_experiment_data(exp_numb)
        with Pool() as p:
            all_data_frames = p.map(get_dataframe, all_paths)
            return all_data_frames
    except Exception as e:
        print("Error in function 'get_all_data_frames': ", e)
        return "error"


def get_channel_data(channel,data_frame):
    """
    Return the data for a given channel.
    Argument:
    --------
    channel:
        channel specified by 0,1,2,3 of up tp 7
    data_frame:
        the datframe containg the data
    """
    return data_frame[channel].values


def get_experiment_bearing_data(exp_numb, channel):
    """
    Return a specific bearing data.
    Arguments:
    ---------
    exp_numb:
        the experiment number (1,2 or 3)
    channel:
        it specifies the bearing (0,1,2,3, ...)
    Return:
    """
    data_frames = get_all_data_frames(exp_numb)
    with Pool() as p:
        all_samples = [p.apply(get_channel_data,
                args=(channel, data_frame)) for data_frame in data_frames]
        return all_samples



def save_to_csv(destination_path, colums_arrays, columns_names):
    """
    Save data to acsv file, given a list of array and
    list of columns names.
    Arguments:
    ---------
    destination_path:
        path to the csv file
    colums_arrays:
        a list containing arrays of values.
        example [[0,1,1], [0.3, 0,4],...]
    columns_names:
        corresponding name for each array in colums_arrays:
        example:["temperature", "humidity", ...]
    """
    dictionary_data = dict(zip(columns_names, colums_arrays))
    data_frame = pd.DataFrame(dictionary_data)
    data_frame.to_csv(destination_path,index=0)


if __name__ == '__main__':
    fault_freqs = [236.4, 296.8, 280.4]








    #get_all_faultes(exp_numb, channel)
    #get_experiment_and_channel_data(exp_numb, channel)
