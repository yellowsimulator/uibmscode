import numpy as np

from etl import *



def get_initial_samples(exp_numb,percent):
    """
    Return percent of initial files.
    Argument:
        exp_numb: experiment number
        percent: percentage if initial file (0.1 for 10%)
    Return:
        init_files: list of files
    """
    all_files = get_experiment_data(exp_numb)
    length = len(all_files)
    init_length = int(percent*length)
    init_files = all_files[:init_length]
    return init_files


def get_health_index(exp_numb,*percent):
    """
    Return the health index.
    Arguments:
        files: all files
    """
    all_files = get_initial_samples(exp_numb,percent[0])
    data_list = get_datum(all_files)
    
























if __name__ == '__main__':
    exp_numb = 1
    percent = 0.1
    get_health_index(exp_numb,percent)
    #get_initial_samples(exp_numb,percent)
