import matplotlib.pyplot as plt
from heapq import nlargest
from multiprocessing import Pool
import statsmodels.api as sm
import numpy as np
from fft import *
from etl import *
from sklearn.svm import SVR
#from fbprophet import Prophet

#
def plot_frequency_spectrum(sample, output_file, channel):
    l = 100

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
    new_list = nlargest(7, list(amplitude[:1000]))
    indexes = list(map(lambda pt: list(amplitude[:1000]).index(pt),new_list))
    new_freq = freq[:1000][indexes]
    x = [236.4 for x in range(l) ]
    yx = np.linspace(0,new_list[0],l)

    xharmonic1 = [new_freq[1] for x in range(l) ]
    harmonic1 = np.linspace(0,new_list[1],l)

    xharmonic2 = [new_freq[2] for x in range(l) ]
    harmonic2 = np.linspace(0,new_list[2],l)

    #xharmonic3 = [new_freq[6] for x in range(l) ]
    #harmonic3 = np.linspace(0,new_list[6],l)
    #print(xharmonic1,harmonic1)
    #print(new_freq[0],new_freq[1],new_freq[2],new_freq[5])
    plt.plot(freq[:1000], amplitude[:1000])
    #plt.plot(x,yx)
    #plt.plot(xharmonic1,harmonic1)
    #plt.plot(xharmonic2,harmonic2)
    #plt.plot(xharmonic3,harmonic3)
    plt.xlabel("Frequency in Hz")
    plt.ylabel("Amplitude in G")
    plt.title("Frequency spectrum for bearing {}".format(channel+1))
    plt.ylim([0,0.55])
    #plt.legend(["Frequency spectrum","BPFO","2*BPFO harmonic","3*BPFO harmonic"])
    plt.savefig("{}.png".format(output_file))
    #plt.show()


def plot_signal(signal, output_file, channel):
    t = np.linspace(0,1, len(signal))
    plt.plot(t, signal)
    plt.xlabel("Time in second")
    plt.ylabel(r'Amplitude in $m/s^{2}$')
    plt.title("Vibration signal obtained from an accelerometer. Bearing {}".format(channel+1))
    plt.savefig("{}.png".format(output_file))


def forcing_frequency(samples,fault_freq):
    with Pool() as p:
        tuples = [p.apply(get_fault_frequency,
            args=(signal, fault_freq)) for signal in samples]
        return tuples

def save_bearing_trend(exp_numb):
    fault_frequencies =  {"bpfi":296.8, "rdf":236.4,"bpfo":236.4}

    #exp_numb = 2 # bpfo
    file_name = "experiment1_fft/experiment{}_bearing_fft_trend.csv".format(exp_numb)
    #output_path = "/Users/yapiachou/UiBergen/UiB-master/fig/{}".format(file_name)

    samples1 = get_experiment_bearing_data(exp_numb, 0)
    print("processing bearing 1")
    samples2 = get_experiment_bearing_data(exp_numb, 2)
    print("processing bearing 2")
    samples3 = get_experiment_bearing_data(exp_numb, 4)
    print("processing bearing 3")
    samples4 = get_experiment_bearing_data(exp_numb, 6)
    print("processing bearing 4")
    fault_freq = fault_frequencies["bpfi"]
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


def plot_fft_trend(path):
    df = pd.read_csv(path)
    amp1 = df["amp1"].values
    amp2 = df["amp2"].values
    amp3 = df["amp3"].values
    amp4 = df["amp4"].values
    plt.plot(amp1)
    plt.plot(amp2)
    plt.plot(amp3)
    plt.plot(amp4)
    plt.legend(["b1","b2","b3","b4"])
    plt.show()


def plot_bearing_fft_trend(file_name,exp_numb,out):
    output_path = "/Users/yapiachou/UiBergen/UiB-master/fig/{}".format(out)
    df = pd.read_csv(file_name)
    #data = df["amp1"].values
    #print(data[0]*9.81, max(data)*9.81)
    #exit()
    #derivative = np.gradient(data)
    #plt.subplot(211)
    #plt.plot(data)
    #plt.subplot(212)
    #plt.plot(derivative)
    #plt.show()
    #exit()
    #y = np.array(range(len(data)))
    #print(y)
    #exit()
    #m = int(0.8*len(data))
    #train = data[:m]; y_train = y[:m]
    #test = data[m:];y_yest = y[m:]
    #print(y_train)
    #print(y_yest)
    #exit()
    #svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
               #coef0=1)
    #svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    #res = svr_rbf.fit(train.reshape(-1,1), y_train).predict(test.reshape(-1,1))

    #print(res)
    #exit()
    #plt.plot(res)
    #plt.plot(test)
    #plt.legend(["predicted","real"])
    #plt.show()
    #exit()
    g = 9.81
    trend1 = decompose(df["amp1"].values).trend
    f = decompose(g*df["amp1"].values).trend
    deriv = np.gradient(f)
    t = np.linspace(0, 10*len(trend1),len(trend1) )
    plt.plot(t,deriv,color="red")
    ##plt.plot(decompose(g*df["amp2"].values).trend)
    #plt.plot(decompose(g*df["amp3"].values).trend)
    #plt.plot(decompose(g*df["amp4"].values).trend)
    plt.xlabel("Minutes")
    plt.ylabel("Gradient")
    plt.title("Gradient of the BPFO defect trend for bearing 1")
    file_name = "bearing1_bpfo_trend_grad.png"
    path = "/Users/yapiachou/UiBergen/UiB-master/fig/{}".format(file_name)
    #plt.legend(["Bearing1 with BPFO defect","Bearing2","Bearing3","Bearing4"])
    plt.savefig(path)
    #plt.show()
    exit()

    trend1 = decompose(df["amp1"].values).trend
    t = np.linspace(0, 10*len(trend1),len(trend1) )
    grand = np.gradient(trend1)
    plt.plot(t,df["amp1"].values,color="red")
    plt.plot(t,trend1)
    #plt.plot(t,grand)
    plt.xlabel("Minutes")
    #plt.ylabel("Gradient")
    plt.title("bearing 1 BPFO, trend, derivative")
    file_name = "bearing1_bpfo_trend_derivative.png"
    path = "/Users/yapiachou/UiBergen/UiB-master/fig/{}".format(file_name)
    plt.legend(["BPFO amplitude", "Trend", "derivative"])
    plt.savefig(path)
    plt.show()
    exit()
    trend2 = decompose(df["amp2"].values).trend
    trend3 = decompose(df["amp3"].values).trend
    trend4 = decompose(df["amp4"].values).trend
    t = np.linspace(0, 10*len(trend1),len(trend1) )
    plt.plot(t,trend1,color="red")
    plt.plot(t,trend2)
    plt.plot(t,trend3)
    plt.plot(t,trend4)
    plt.xlabel("Minutes")
    plt.ylabel("Amplitude in G")
    plt.title("BPFO amplitude trend for experiment {}, for all bearings".format(exp_numb))
    plt.legend(["Bearing1 with BPFO defect","Bearing2","Bearing3","Bearing4"])
    plt.savefig(output_path)
    plt.show()



if __name__ == '__main__':
    path = "experiment1_fft/experiment1_bearing_fft_trend.csv"
    plot_fft_trend(path)
    exit()
    exp_numb = 1
    #path = "../data/3rd_test"
    #all_files = get_all_files(path)
    #all_files.sort()
    #print(all_files)
    #print(get_all_dates(exp_numb)[533])
    #exit()
    save_bearing_trend(exp_numb)
    exit()
    out = "experiment{}_bearing_fft_trend.png".format(exp_numb)
    file_name = "experiment{}_bearing_fft_trend.csv".format(exp_numb)
    plot_bearing_fft_trend(file_name,exp_numb,out)
    exit()
    #save_bearing_trend()
    #exit()
    fault_frequencies =  {"bpfi":296.8, "bpfo":236.4}
    exp_numb = 2 # bpfo
    channel = 3
    file_name = "experiment{}_bearing{}".format(exp_numb,channel+1)
    output_path = "/Users/yapiachou/UiBergen/UiB-master/fig/{}".format(file_name)
    samples = get_experiment_bearing_data(exp_numb, channel)
    fault_freq = fault_frequencies["bpfo"]
    bearing1 = forcing_frequency(samples,fault_freq)
    #decomp = decompose(tuples)

    #trend = decomp.trend
    #plt.plot(trend)
    #plt.show()
    #exit()
    #m = len(samples)
    k = 900
    signal = samples[k]
    #plot_signal(signal, output_path,channel)
    plot_frequency_spectrum(signal, output_path,channel)
    #sm.graphics.tsa.plot_acf(spectrm, lags=40)
    plt.show()
