import matplotlib.pyplot as plt
from collections import OrderedDict
from heapq import nlargest
import numpy as np
from fft import *
from etl import *
#
#
#
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
    new_list = nlargest(7, list(amplitude[:1000]))
    indexes = list(map(lambda pt: list(amplitude[:1000]).index(pt),new_list))
    new_freq = freq[:1000][indexes]
    print(new_freq)
    x = [new_freq[peak_number] for x in range(l) ]
    yx = np.linspace(0,new_list[peak_number],l)

    rpm_xx = [new_freq[-2] for x in range(l) ]
    rpm_yx = np.linspace(0,new_list[-2],l)

    plt.plot(freq[:1000], amplitude[:1000])
    if peak_number != 0:
        plt.plot(x,yx,color="red")
    #plt.plot(rpm_xx,rpm_yx,color="green")
    plt.ylim([0,1.3])
    plt.xlabel("Frequency in Hz")
    plt.ylabel("Amplitude in G")
    plt.title("Envelope frequency spectrum in {}".format(date))
    plt.legend(["Frequency spectrum","BPFI"])
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
from scipy import signal as s
if __name__ == '__main__':
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
    #dt = 0.01
    #plt.psd(signal, 512, 1 / dt)
    fs = 20480
    f, Pxx_den = s.welch(signal, fs, nperseg=1024)

    plt.semilogy(f[:100], Pxx_den[:100])
    plt.show()
    #plot_rdf_spectrum(signal, channel,date,peak_number)
    #plot_bpfo_spectrum(signal, channel,date,peak_number)
    #plot_signal(signal, date)
    #plot_envelope(signal,date)


    exit()






























##
##
