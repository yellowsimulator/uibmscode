"""
Extract data, Transform data and Load
data if neccessary
"""
import os
import pandas as pd
from glob import glob
import scipy.stats
from multiprocessing import Pool
import matplotlib.pyplot as plt
from stldecompose import decompose, forecast
from fft import *
from wavelet_transform import *
#from getEnvelopeModels import get_envelope_models



def get_date_from_file(file):
    """
    return the date extension from a
    file.
    Argument:
    file: a file name.
    Return: datetime of the form yyyy-mm-dd hh:mm:ss
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
    --------
    exp_numb: the experiment number.
    """
    all_files = get_experiment_data(exp_numb)
    all_files.sort()
    with Pool() as p:
        datetime = p.map(get_date_from_file, all_files)
        return datetime


def get_all_files(path):
    """
    Return all files from a directory specified by a path.
    Arguments:
    --------
    path: directory path
    Return: files: all files in the directory
            specified by path argument.
    """
    try:
        files = glob("{}/*".format(path))
        files.sort()
        return files
    except Exception as e:
        print("Error in function 'get_all_files': ", e)


def get_dataframe(path):
    """
    Return a dataframe from.
    Argument: path: file path
    Return: dataframe: a pandas dataframe
    """
    try:
        dataframe = pd.read_csv(path,header=None,delim_whitespace=True)
        return dataframe
    except Exception as e:
        print("Error in function 'get_dataframe': ", e)


def get_experiment_data(exp_numb):
    """
    Return all file for a given experiment.
    Arguments: exp_numb: experiment number: intenger 1,2or 3
    ---------
    Return: all_files: all files
    ------
    """
    experiments = {"1": "1st_test", "2": "2nd_test", "3": "3rd_test"}
    experiment = experiments["{}".format(exp_numb)]
    main_path = os.path.abspath("../data/{}".format(experiment))
    try:
        all_files = get_all_files(main_path)
        all_files.sort()
        return all_files
    except Exception as e:
        print("Error in function 'get_experiment_data': ", e)


def get_all_data_frames(exp_numb):
    """
    Return all dataframes for a given experiment.
    Argument:
    --------
    exp_numb: experiment number (1,2 or 3)
    Return:
    ------
    all_data_frames: all dataframes pertaining to an experiment
    """
    try:
        all_paths = get_experiment_data(exp_numb)
        all_paths.sort()
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
    channel: channel specified by 0,1,2,3 of up tp 7
    data_frame: the datframe containg the data
    """
    return data_frame[channel].values


def get_experiment_bearing_data(exp_numb, channel):
    """
    Return a specific bearing data.
    Arguments:
    ---------
    exp_numb: the experiment number (1,2 or 3)
    channel: it specifies the bearing (0,1,2,3, ...)
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
    colums_arrays: a list containing arrays of values.
                  example [[0,1,1], [0.3, 0,4],...]
    columns_names: corresponding name for each array in colums_arrays:
                   example:["temperature", "humidity", ...]
    """
    dictionary_data = dict(zip(columns_names, colums_arrays))
    data_frame = pd.DataFrame(dictionary_data)
    data_frame.to_csv(destination_path,index=0)


def validate_data(sample):
    """
    Validate samples based on their health
    index.
    Arguments:
    ---------
    sample: the sample containg signals.
    """
    for k, item in enumerate(sample):
        try:
            if k==0:
                if sample[k] > sample[k+1]:
                    sample.remove(sample[k])
                    print("new list length {}".format(len(sample)))
                    validate_data(sample)
            elif (sample[k] < sample[k+1]) and (sample[k] < sample[k-1]):
                sample.remove(sample[k])
                print("new list length {}".format(len(sample)))
                #print("new list {}".format(sample))
                validate_data(sample)
        except Exception as e:
            print(sample)
            break


            #return sample






if __name__ == '__main__':
    fault_freqs = [236.4, 296.8, 280.4]
    #sample = [1, 0.5, 0.2, 1, 2.3, 0.1, 9.2]
    sample = [1, 0.5, 0.6, 1, 2.3, 3.1, 9.2]
    df = pd.read_csv("test0.csv")

    #print("Original sample: ",sample)
    #new_sample = sample.copy()
    #print(new_sample)
    #new_sample = validate_data(sample)
    window1 = 5
    window2 = 1
    window_percent = (window2/len(df))*100
    print(window_percent)
    rolling_mean = df["health_index"].rolling(window=window1).mean()
    rolling_mean2 = df["health_index"].rolling(window=window2).mean()
    x = range(len(df))
    s = rolling_mean2.dropna().values
    #s = df["health_index"].values
    P = get_envelope_models(s)
    q_u = list(map(P[0],range(0,len(s))))
    q_l = list(map(P[1],range(0,len(s))))
    decomp = decompose(q_u)
    trend = decomp.trend
    plt.subplot(311)
    plt.plot(s)
    plt.subplot(312)
    plt.xlabel("sample")
    plt.ylabel("health index")
    plt.plot(trend, color="orange")
    #plt.subplot(313)
    #plt.plot(decomp.seasonal)
    #plt.plot(s);plt.xlabel('x');plt.ylabel('y')
    #plt.title('Upper and lower envelopes including all peaks and troughs')
    #plt.hold(True)
    #plt.plot(q_u)
    #plt.plot(q_l)
    #plt.grid(True)
    #plt.hold(False)
    #decomp.plot()
    plt.show()
    exit()
    #plt.plot(x, df["health_index"], label='raw')
    #plt.plot(x, rolling_mean, label='window {}'.format(window1), color='orange')
    plt.plot(x, rolling_mean2, label='window {}'.format(window2), color='magenta')
    plt.legend(loc='upper left')
    plt.show()








    #get_all_faultes(exp_numb, channel)
    #get_experiment_and_channel_data(exp_numb, channel)
