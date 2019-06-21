import matplotlib.pyplot as plt
from heapq import nlargest
import statsmodels.api as sm
import numpy as np
from fft import *
from etl import *



#
def plot_frequency_spectrum(sample, output_file):
    l = 100
    x = [236.4 for x in range(l) ]
    yx = np.linspace(0,0.039,l)
    lowcut = 2000
    highcut = 9990
    sampling_freq = 20000
    period = 1
    sampling_interval = 1./20480
    y = butter_bandpass_filter(sample,lowcut,highcut,sampling_freq,order=5)
    envelop = get_envelop(y)
    low_pass = butter_lowpass_filter(envelop,2000,sampling_freq,order=5)
    freq, amplitude = get_fft(low_pass,period,sampling_interval)
    #return freq
    #new_list = nlargest(4, list(amplitude[:1000]))
    #indexes = list(map(lambda pt: list(amplitude[:1000]).index(pt),new_list))
    #new_freq = freq[:1000][indexes]

    #xharmonic1 = [new_freq[1] for x in range(l) ]
    #harmonic1 = np.linspace(0,new_list[1],l)

    #xharmonic2 = [new_freq[2] for x in range(l) ]
    #harmonic2 = np.linspace(0,new_list[2],l)

    #xharmonic3 = [new_freq[3] for x in range(l) ]
    #harmonic3 = np.linspace(0,new_list[3],l)
    #print(xharmonic1,harmonic1)

    #plt.plot(freq[:1000], amplitude[:1000])
    #plt.plot(x,yx)
    #plt.plot(xharmonic1,harmonic1)
    #plt.plot(xharmonic2,harmonic2)
    #plt.plot(xharmonic3,harmonic3)
    #plt.xlabel("Frequency")
    #plt.ylabel("Amplitude")
    #plt.ylim([0,0.04])
    #plt.legend(["Frequency spectrum","BPFO","2*BPFO","3*BPFO","4*BPFO"])
    #plt.savefig("{}.png".format(output_file))


def plot_signal(signal, output_file):
    t = np.linspace(0,1, len(signal))
    plt.plot(t, signal)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.savefig("{}.png".format(output_file))



if __name__ == '__main__':

    fault_frequencies =  {"bpfi":296.8, "bpfi":236.4}
    exp_numb = 2 # bpfo
    channel = 0
    file_name = "experiment{}_bearing{}_fft".format(exp_numb,channel+1)
    output_path = "/Users/yapiachou/UiBergen/UiB-master/fig/{}".format(file_name)
    samples = get_experiment_bearing_data(exp_numb, channel)

    m = len(samples)
    k = 900
    signal = samples[k]
    #plot_signal(signal, output_path)
    plot_frequency_spectrum(signal, output_path)
    #sm.graphics.tsa.plot_acf(spectrm, lags=40)
    plt.show()
