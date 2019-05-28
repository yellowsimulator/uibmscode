"""
Do temporarely processing here
"""
from multiprocessing import Pool
from etl import *
from signal_processing_method import *


def generate_all_faults_csv(channels,exp_numb):
    #faults_freq = [236.4, 236., 280.4]
    fault_freqs = [236.4, 296.8, 280.4]
    for channel in channels:
        print("processing channel {}".format(channel+1))
        all_dates = get_all_dates(exp_numb)
        all_files = get_experiment_data(exp_numb)
        with Pool() as p:
            data = p.map(get_bearing_data, all_files)
            faults = p.apply(get_all_faultes, args=(exp_numb, channel))
            df = pd.DataFrame({"timestamp": all_dates,"faults": faults})
            df.to_csv("k_experiment{}_channel{}.csv".format(exp_numb,
                                           channel+1))

channels = [0,1,2,3,4]
exp_numb = 2
generate_all_faults_csv(channels,exp_numb)
