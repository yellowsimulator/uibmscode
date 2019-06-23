"""
Objective of this file:
----------------------
This file is the main file.
functions implemented in other
files are called here. It is the experimental lab.
Refer to readme file for appropriate channel number
"""
from scipy.fftpack import fft, ifft
from pyhht.visualization import plot_imfs
from tftb.processing import ShortTimeFourierTransform
import numpy as np
import pandas as pd
import os
from glob import glob
import matplotlib.pyplot as plt
from etl import *
from fft import *
from statistics import *
from hht import get_imfs
from getEnvelopeModels import get_envelope_models
from peak_detect import detect_peaks
from wavelet_transform import *

def main(exp_numb, channel, dispersion_index):
    #if os.path.isfile("test{}_{}.csv".format(channel,dispersion_index)):
        #os.remove("test{}_{}.csv".format(channel,dispersion_index))
    samples = get_experiment_bearing_data(exp_numb, channel)
    #sample = samples[0]
    for j, sample in enumerate(samples):
        print("processing sample {}".format(j+1))
        imfs = get_imfs(sample)
        m = len(imfs)
        names = ["imf{}".format(k+1) for k in range(m)]
        #d = dict(zip(names,imfs))
        #df = pd.DataFrame(d)
        save_to_csv("../data/imfs/sample{}_imfs.csv".format(j+1),imfs, names )
    #health_indexes = [get_all_health_index(dispersion_index, samples)]
    #columns_names = ["health_index"]
    #destination_path = "test{}_{}.csv".format(channel,dispersion_index)
    #save_to_csv(destination_path, health_indexes, columns_names)

    #health_index_array = pd.read_csv("test{}.csv".format(channel))["health_index"].values

    #coeffs = get_all_health_index_coefficients(health_index_array)
    #indexes = range(len(health_index_array))
    #initialise
    #plt.plot(coeffs)
    #plt.show()




def burst_amplitude():
    amplitudes = []
    path = "../data/imfs/"
    files = glob("{}/*".format(path))
    for k, file in enumerate(files):
        print("processing file {}".format(k+1))
        df = pd.read_csv(file)
        j = 6
        try:
            s  = df["imf{}".format(j)].values
        except Exception as e:
            pass
        decomp = decompose(s)
        seasonality = decomp.seasonal
        amp = max(seasonality)
        amplitudes.append(amp)
    print("saving data to file")
    d = {"amplitude":amplitudes}
    df = pd.DataFrame(d)
    df.to_csv("burst_amplitide.csv")



def plot_wavelet(signale):
    low_freq, high_freq = get_high_low_freq(signale)
    t = np.linspace(0,1,len(high_freq))
    plt.plot(t,high_freq)
    plt.title("High frequency signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()











if __name__ == '__main__':
    #burst_amplitude()
    ##decomp = decompose(burst)
    #decomp.plot()
    #plt.plot(burst)
    #plt.show()
    #exit()

    exp_numb = 2; channel = 0; dispersion_index = "iqr"
    samples = get_experiment_bearing_data(exp_numb, channel)
    

    #k = 500
    k = 100
    sample = samples[k]
    plot_wavelet(sample)
    exit()
    path = "../data/imfs/sample{}_imfs.csv".format(k)
    df = pd.read_csv(path)
    #new_df = df.loc[:,"imf1":"imf4"]
    #new_df.plot()
    #plt.show()
    #exit()
    k = 6
    #for k in range(1,9):
    s = df["imf{}".format(k)].values

    decomp = decompose(s)
    seasonality = decomp.seasonal
    lim = 10000
    t = np.linspace(0,1, lim)
    #peaks = detect_peaks(seasonality[:1000])
    plt.subplot(311)
    #plt.title("Hilbert Huang transform for burst identification")
    plt.ylabel("Amplitude")
    plt.plot(t,sample[:lim])
    plt.subplot(312)
    plt.plot(t,s[:lim])
    #plt.title("Short and hight frequency burst")
    ##plt.ylabel("Amplitude")
    #plt.plot(t2, np.cos(2*np.pi*t2), color='tab:orange', linestyle='--')
    plt.subplot(313)
    plt.plot(t,seasonality[:lim])

    plt.xlabel("Time")

    plt.show()
    exit()
    #print(peaks)
    #print(seasonality[:1000][peaks])
    #exit()
    #P = get_envelope_models(seasonality)
    #Evaluate each model over the domain of (s)
    #q_u = list(map(P[0],range(0,len(seasonality))))
    #q_l = list(map(P[1],range(0,len(seasonality))))
    #print(seasonality[:5000])
    #exit()
    #plt.plot(q_u)
    #plt.show()
    #N = len(seasonality)
    #T = 1./N
    #imfs = get_imfs(s)
    #yf = fft(seasonality)
    #xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    #sampling_interval = 1./N
    #freq = np.linspace(0.0, 1.0/(2.0*sampling_interval), N/2)
    #plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
    #amp = 2.0/N * np.abs(yf[0:N//2])
    #plt.plot(freq,amp )
    #peaks = detect_peaks(amp)
    #j = 1
    #idx = list(peaks).index(peaks[j])
    #corr_amp = amp[idx]

    #print(peaks[j],"--",corr_amp)
    #plt.grid()
    #plt.show()
    #plot_imfs(seasonality,imfs)
    #plt.show()
    #stft = ShortTimeFourierTransform(seasonality)
    #print(stft)
    #stft.run()
    #stft.plot()

    #plt.plot(imf)
    decomp.plot()
    plt.show()

    #main(exp_numb, channel, dispersion_index)















    #main(exp_numb, channel, dispersion_index)
