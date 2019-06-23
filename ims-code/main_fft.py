import matplotlib.pyplot as plt
from heapq import nlargest
from multiprocessing import Pool
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

    plt.plot(freq[:1000], amplitude[:1000])
    plt.plot(x,yx)
    #plt.plot(xharmonic1,harmonic1)
    #plt.plot(xharmonic2,harmonic2)
    #plt.plot(xharmonic3,harmonic3)
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    #plt.ylim([0,0.04])
    #plt.legend(["Frequency spectrum","BPFO","2*BPFO","3*BPFO","4*BPFO"])
    plt.savefig("{}.png".format(output_file))


def plot_signal(signal, output_file):
    t = np.linspace(0,1, len(signal))
    plt.plot(t, signal)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.savefig("{}.png".format(output_file))


def forcing_frequency(samples,fault_freq):
    with Pool() as p:
        tuples = [p.apply(get_fault_frequency,
            args=(signal, fault_freq)) for signal in samples]
        return tuples

def save_bearing_trend(exp_numb):
    fault_frequencies =  {"bpfi":296.8, "bpfo":236.4, "rdf":236.4}
    #exp_numb = 2 # bpfo
    file_name = "experiment{}_bearing_fft_trend.csv".format(exp_numb)
    #output_path = "/Users/yapiachou/UiBergen/UiB-master/fig/{}".format(file_name)

    samples1 = get_experiment_bearing_data(exp_numb, 1)
    print("processing bearing 1")
    samples2 = get_experiment_bearing_data(exp_numb, 3)
    print("processing bearing 2")
    samples3 = get_experiment_bearing_data(exp_numb, 5)
    print("processing bearing 3")
    samples4 = get_experiment_bearing_data(exp_numb, 7)
    print("processing bearing 4")
    fault_freq = fault_frequencies["rdf"]
    bearing1 = forcing_frequency(samples1,fault_freq)
    print("computing bearing 1 forcing frequency")
    bearing2 = forcing_frequency(samples2,fault_freq)
    print("computing bearing 2 forcing frequency")
    bearing3 = forcing_frequency(samples3,fault_freq)
    print("computing bearing 3 forcing frequency")
    bearing4 = forcing_frequency(samples4,fault_freq)
    print("computing bearing 4 forcing frequency")
    d = {"amp1":bearing1,"amp2":bearing2,"amp3":bearing3,"amp4":bearing4}
    df = pd.DataFrame(d)
    df.to_csv(file_name)


def plot_bearing_fft_trend(file_name,exp_numb,out):
    output_path = "/Users/yapiachou/UiBergen/UiB-master/fig/{}".format(out)
    df = pd.read_csv(file_name)
    plt.plot(df["amp1"].values)
    plt.plot(df["amp2"].values)
    plt.plot(df["amp3"].values)
    plt.plot(df["amp4"].values)
    #plt.show()
    #exit()
    #trend1 = decompose(df["amp1"].values).trend
    #trend2 = decompose(df["amp2"].values).trend
    #trend3 = decompose(df["amp3"].values).trend
    #trend4 = decompose(df["amp4"].values).trend
    #plt.plot(trend1)
    #plt.plot(trend2)
    #plt.plot(trend3)
    #plt.plot(trend4)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.title("BPFO amplitude for experiment {}".format(exp_numb))
    plt.legend(["Bearing1","Bearing2","Bearing3","Bearing4"])
    plt.savefig(output_path)
    #plt.show()



if __name__ == '__main__':
    exp_numb = 2
    #path = "../data/3rd_test"
    #all_files = get_all_files(path)
    #all_files.sort()
    #print(all_files)
    #print(get_all_dates(exp_numb)[321])
    #exit()
    #save_bearing_trend(exp_numb)
    out = "experiment{}_bearing_fft_amp.png".format(exp_numb)
    file_name = "experiment{}_bearing_fft_trend.csv".format(exp_numb)
    plot_bearing_fft_trend(file_name,exp_numb,out)
    #save_bearing_trend()
    exit()
    fault_frequencies =  {"bpfi":296.8, "bpfo":236.4}
    exp_numb = 2 # bpfo
    channel = 0
    file_name = "experiment{}_bearing{}_fft".format(exp_numb,channel+1)
    output_path = "/Users/yapiachou/UiBergen/UiB-master/fig/{}".format(file_name)
    samples = get_experiment_bearing_data(exp_numb, channel)
    fault_freq = fault_frequencies["bpfo"]
    bearing1 = forcing_frequency(samples,fault_freq)
    decomp = decompose(tuples)

    #trend = decomp.trend
    #plt.plot(trend)
    plt.show()
    exit()
    m = len(samples)
    k = 900
    signal = samples[k]
    #plot_signal(signal, output_path)
    plot_frequency_spectrum(signal, output_path)
    #sm.graphics.tsa.plot_acf(spectrm, lags=40)
    plt.show()
