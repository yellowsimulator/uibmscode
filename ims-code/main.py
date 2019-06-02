"""
Objective of this file:
----------------------
This file is the main file.
functions implemented in other
files are called here. It is the experimental lab.
Refer to readme file for appropriate channel number
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from etl import *
from signal_processing_method import *
from statistics import *

def main(exp_numb, channel, dispersion_index):
    if os.path.isfile("test{}.csv".format(channel)):
        os.remove("test{}.csv".format(channel))
    samples = get_experiment_bearing_data(exp_numb, channel)
    health_indexes = [get_all_health_index(dispersion_index, samples)]
    columns_names = ["health_index"]
    destination_path = "test{}.csv".format(channel)
    save_to_csv(destination_path, health_indexes, columns_names)

    health_index_array = pd.read_csv("test{}.csv".format(channel))["health_index"].values

    coeffs = get_all_health_index_coefficients(health_index_array)
    indexes = range(len(health_index_array))
    #initialise
    plt.plot(coeffs)
    plt.show()















if __name__ == '__main__':

    exp_numb = 2; channel = 0; dispersion_index = "iqr"
    main(exp_numb, channel, dispersion_index)















    #main(exp_numb, channel, dispersion_index)
