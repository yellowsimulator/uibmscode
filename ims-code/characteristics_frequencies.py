import matplotlib.pyplot as plt
from collections import OrderedDict
from heapq import nlargest
import numpy as np
import seaborn as sns
#sns.set(); np.random.seed(0)
from fft import *
from etl import *
#
#
#
def variance(signal,date):
    v = np.var(signal)
    print(" variance: {} at {}".format(v, date))


def plot_distribution(signal_start,signal_end ,date_start,date_end):

    ax = sns.kdeplot(signal_start, shade=True,label="{}".format(date_start))
    ax = sns.kdeplot(signal_end,shade=True,label="{}".format(date_end))
    #plt.legend(["{}".format(date_start), "{}".format(date_end)])
    plt.title("Vibration time signal probability density per day")
    plt.xlabel("Amplitude")
    plt.ylabel("Frequency")
    plt.show()

def plot_signal(signal, date):
    m = len(signal)
    t = np.linspace(0,1,m)
    plt.plot(t, signal)
    #plt.title("Vibration accelaration recorded at {}".format(date))
    plt.xlabel("Time in second")
    plt.ylabel("Amplitude in G")
    plt.title("Acceleration signal in {}".format(date))
    plt.ylim([-0.6,0.62])
    plt.show()

def plot_envelope(signal,date):
    lowcut = 2000
    highcut = 9990
    sampling_freq = 20000
    period = 1
    sampling_interval = 1./20480
    y = butter_bandpass_filter(signal,lowcut,highcut,sampling_freq,order=5)
    envelop = get_envelop(y)

    m = len(envelop)
    t = np.linspace(0,1,m)
    plt.plot(t, envelop)
    plt.xlabel("Time in second")
    plt.ylabel("Amplitude in G")
    plt.title("Evelope signal in {}".format(date))
    plt.ylim([0,0.55])
    plt.show()





def plot_bpfi_spectrum(signal, channel,date,peak_number):
    l = 100

    lowcut = 2000
    highcut = 9990
    sampling_freq = 20000
    period = 1
    sampling_interval = 1./20480
    y = butter_bandpass_filter(signal,lowcut,highcut,sampling_freq,order=5)
    envelop = get_envelop(y)
    low_pass = butter_lowpass_filter(envelop,2000,sampling_freq,order=5)
    freq, amplitude = get_fft(low_pass,period,sampling_interval)
    new_list = nlargest(15, list(amplitude[:1000]))
    indexes = list(map(lambda pt: list(amplitude[:1000]).index(pt),new_list))
    new_freq = freq[:1000][indexes]
    print(new_freq)
    #exit()
    x = [new_freq[peak_number] for x in range(l) ]
    yx = np.linspace(0,new_list[peak_number],l)

    sb1_xx = [new_freq[3] for x in range(l) ]
    sb1_yx = np.linspace(0,new_list[3],l)

    sb2_xx = [new_freq[1] for x in range(l) ]
    sb2_yx = np.linspace(0,new_list[1],l)

    sb3_xx = [new_freq[6] for x in range(l) ]
    sb3_yx = np.linspace(0,new_list[6],l)

    sb4_xx = [new_freq[13] for x in range(l) ]
    sb4_yx = np.linspace(0,new_list[13],l)


    plt.plot(freq[:1000], amplitude[:1000])
    #if peak_number != 0:
    plt.plot(x,yx,color="red")
    plt.plot(sb1_xx,sb1_yx,color="green")
    plt.plot(sb2_xx,sb2_yx,color="green")
    plt.plot(sb3_xx,sb3_yx,color="green")
    plt.plot(sb4_xx,sb4_yx,color="green")
    plt.ylim([0,1.3])
    plt.xlabel("Frequency in Hz")
    plt.ylabel("Amplitude in G")
    plt.title("Envelope frequency spectrum in {}".format(date))
    plt.legend(["Frequency spectrum","BPFI","Sidebands"])
    plt.show()




def plot_bpfo_spectrum(signal, channel,date,peak_number):
    l = 100

    lowcut = 2000
    highcut = 9990
    sampling_freq = 20000
    period = 1
    sampling_interval = 1./20480
    y = butter_bandpass_filter(signal,lowcut,highcut,sampling_freq,order=5)
    envelop = get_envelop(y)
    low_pass = butter_lowpass_filter(envelop,2000,sampling_freq,order=5)
    freq, amplitude = get_fft(low_pass,period,sampling_interval)
    new_list = nlargest(7, list(amplitude[:1000]))
    indexes = list(map(lambda pt: list(amplitude[:1000]).index(pt),new_list))
    new_freq = freq[:1000][indexes]
    print(new_freq)
    #exit()

    x = [new_freq[peak_number] for x in range(l) ]
    yx = np.linspace(0,new_list[peak_number],l)

    x_bpfo = [new_freq[0] for x in range(l) ]
    y_bpfo = np.linspace(0,new_list[0],l)

    xh2_bpfo = [new_freq[1] for x in range(l) ]
    yh2_bpfo = np.linspace(0,new_list[1],l)

    xh3_bpfo = [new_freq[2] for x in range(l) ]
    yh3_bpfo = np.linspace(0,new_list[2],l)

    plt.plot(freq[:1000], amplitude[:1000])
    if peak_number != 0:
        plt.plot(x_bpfo, y_bpfo)
        plt.plot(xh2_bpfo, yh2_bpfo)
        plt.plot(xh3_bpfo, yh3_bpfo)

    plt.ylim([0,0.6])
    plt.xlabel("Frequency in Hz")
    plt.ylabel("Amplitude in G")
    plt.title("Envelope frequency spectrum in {}".format(date))
    plt.legend(["Frequency spectrum","BPFO", "Harmonic 2BPFO", "Harmonic 3BPFO"])
    plt.show()



def plot_rdf_spectrum(signal, channel,date,peak_number):
    l = 100

    lowcut = 2000
    highcut = 9990
    sampling_freq = 20000
    period = 1
    sampling_interval = 1./20480
    y = butter_bandpass_filter(signal,lowcut,highcut,sampling_freq,order=5)
    envelop = get_envelop(y)
    low_pass = butter_lowpass_filter(envelop,2000,sampling_freq,order=5)
    freq, amplitude = get_fft(low_pass,period,sampling_interval)
    new_list = nlargest(7, list(amplitude[:1000]))
    indexes = list(map(lambda pt: list(amplitude[:1000]).index(pt),new_list))
    new_freq = freq[:1000][indexes]
    print(new_freq)
    #exit()


    plt.plot(freq[:1000], amplitude[:1000])


    #plt.ylim([0,0.6])
    plt.xlabel("Frequency in Hz")
    plt.ylabel("Amplitude in G")
    plt.title("Envelope frequency spectrum in {}".format(date))
    plt.legend(["Frequency spectrum","BPFO", "Harmonic 2BPFO", "Harmonic 3BPFO"])
    plt.show()



def take_closest(num,collection):
   return min(collection,key=lambda x:abs(x-num))


def all_bpfi_amplitude(signal):
    lowcut = 2000
    highcut = 9990
    sampling_freq = 20000
    period = 1
    sampling_interval = 1./20480
    y = butter_bandpass_filter(signal,lowcut,highcut,sampling_freq,order=5)
    envelop = get_envelop(y)
    low_pass = butter_lowpass_filter(envelop,2000,sampling_freq,order=5)
    freq, amplitude = get_fft(low_pass,period,sampling_interval)
    new_list = nlargest(20, list(amplitude[:1000]))
    indexes = list(map(lambda pt: list(amplitude[:1000]).index(pt),new_list))
    new_freq = freq[:1000][indexes]
    new_amplitude = amplitude[:1000][indexes]
    return new_freq, new_amplitude

def get_all_defect(exp_numb,channel,defect_freq,percent):
    all_defect_frequencies = []
    all_signals = get_experiment_bearing_data(exp_numb, channel)
    lim = int(len(all_signals)/2)
    for signal in all_signals[lim:]:
        new_freq, new_amplitude = all_bpfi_amplitude(signal)
        #bpfi = take_closest(defect_freq,new_freq)
        bpfi = list(filter(lambda x: abs(x-296.8)<=296.8*(percent/100.), list(new_freq)))
        #print(bpfi)
        #exit()
        if len(bpfi) > 1:
            all_defect_frequencies.append(bpfi[0])
    return all_defect_frequencies

def save_defect_frequencies(exp_numb,channel,defect_freq,percent):
    #exp_numb = 1
    #channel = 4
    #defect_freq = 296.8
    path = "bpfi_freq/bpfi{}.csv".format(percent)
    all_defect_frequencies = get_all_defect(exp_numb,channel,defect_freq,percent)
    d = {"freq":all_defect_frequencies}
    df = pd.DataFrame(d)
    df.to_csv("{}".format(path))

def slip_bpfi(percent):
    path = "experiment1-pics/all_bpfi.csv"
    data = pd.read_csv(path)["freq"].values
    filtered = list(filter(lambda x: abs(x-296.8)<=296.8*(percent/100.), list(data)))
    rest = list(set(list(data))-set(filtered))
    #print(filtered)
    #exit()
    m = len(filtered)
    p = round((m/len(data))*100.,2)
    print(p,"%")
    #sns.distplot(filtered,label="")

    plt.vlines(296.8,0,250,color="orange")
    plt.hist(filtered,color="green")
    plt.xlabel("Frequency in Hz")
    plt.ylabel("Frequency count")
    plt.legend(["BPFI (296.8 Hz)","Within {}% of BPFI".format(percent)])
    plt.title("Distribution of All frequencies within {}% of BPFI".format(percent))
    plt.show()


def all_slip():
    path = "experiment1-pics/all_bpfi.csv"
    data = pd.read_csv(path)["freq"].values
    sns.distplot(data,label="")
    #sns.distplot([296.8],label="")
    #plt.hlines(296.8,0,1)
    plt.xlabel("Frequency in Hz")
    plt.ylabel("Frequency count")
    plt.legend(["Closest to BPFI","BPFI"])
    plt.title("Distribution of All frequencies closest to BPFI")
    plt.show()

from scipy import signal as s
if __name__ == '__main__':
    percent = 1
    exp_numb = 1
    channel = 4
    defect_freq = 296.8
    path = "bpfi_freq/bpfi.csv"
    for percent in [1,2,3,4,5]:
        print("processing {}".format(1))
        save_defect_frequencies(exp_numb,channel,defect_freq,percent)
    exit()
    percent = 5
    slip_bpfi(percent)
    #all_slip()
    #print(all_defect_frequencies)
    exit()
    path = "../data/1st_test"
    files = get_all_files(path)
    m = len(files)

    k = -1# -10, -11 , -15, -12 is good
    #numbers = {"{}".format(-11):-4,"{}".format(-10):-2,"{}".format(-1):0,"{}".format(-12):-5}
    peak_number = 0 #numbers["{}".format(k)]

    file = files[k]
    date = get_date_from_file(file).split('\\')[1]
    #print(date)
    #exit()
    data_frame = get_dataframe(file)
    channel = 4 # or 5
    signal = get_channel_data(channel,data_frame)
    plot_bpfi_spectrum(signal, channel,date,peak_number)
    #variance(signal,date)
    exit()
    date_start = get_date_from_file(files[0]).split('\\')[1]
    date_end = get_date_from_file(files[-1]).split('\\')[1]

    signal_start = get_channel_data(channel,get_dataframe(files[0]))
    signal_end = get_channel_data(channel,get_dataframe(files[-1]))
    plot_distribution(signal_start,signal_end,date_start,date_end)


    #plot_rdf_spectrum(signal, channel,date,peak_number)
    #plot_bpfo_spectrum(signal, channel,date,peak_number)
    #plot_signal(signal, date)
    #plot_envelope(signal,date)


    exit()






























##
##
