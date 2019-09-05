import pyhht
from pyhht import EMD
import numpy as np
from etl import *
from stldecompose import decompose, forecast
def get_imfs(signal):
    """"
    Return the imf of a signal
    """
    decomposer = EMD(signal)
    imf = decomposer.decompose()
    return imf

def save_imf(exp_numb, channel):
    samples = get_experiment_bearing_data(exp_numb, channel)
    for j, sample in enumerate(samples):
        print("processing sample {}".format(j+1))
        imfs = get_imfs(sample)
        m = len(imfs)
        names = ["imf{}".format(k+1) for k in range(m)]
        save_to_csv("../data/imf_bpfo/sample{}_imfs.csv".format(j+1),imfs, names )


def burst_amplitude():
    amplitudes = []
    path = "../data/imf_bpfo/"
    files = glob("{}/*".format(path))
    for sample_k, file in enumerate(files):
        print("processing file {}".format(sample_k+1))
        df = pd.read_csv(file)
        columns  = list(df.columns)
        temp_list = []
        temp_dict = {}
        for j, column_name in enumerate(columns):
            imf = list(df[column_name])
            decomp = decompose(imf)
            seasonality = decomp.seasonal
            col = "slt_of_imf{}".format(j+1)
            temp_list.append((col,seasonality))
        temp_dict = dict(temp_list)
        temp_df = pd.DataFrame(temp_dict)
        temp_df.to_csv("../data/stl_bpfo/sample{}_stl.csv".format(sample_k))



def plot_imf(signal):
    decomposer = EMD(signal)
    imfs = decomposer.decompose()
    m = len(signal)
    x = np.linspace(0,1,m)
    pyhht.plot_imfs(x, imfs)
    plt.show()



def get_imf(exp_numb,channel):
    samples = 100#728 #714, 724(4)
    path = "../data/imf_bpfo/sample{}_imfs.csv".format(samples)
    df = pd.read_csv(path)
    #new_df = df.loc[:,"imf1":"imf4"]
    #new_df.plot()
    #plt.show()
    #exit()
    k = 6
    #for k in range(1,9):
    s = df["imf{}".format(k)].values

    decomp = decompose(s)
    lim = 10000
    seasonality = decomp.seasonal[:lim]

    t = np.linspace(0,1, lim)
    plt.plot(t,seasonality)
    #data = get_experiment_bearing_data(exp_numb,channel)
    #k = 850
    #j = 8
    #signal = data[k]
    #imfs = get_imfs(signal)
    #imf = imfs[j]
    #decomp = decompose(imf)
    #pulse = decomp.seasonal
    #lim = 10000
    #plt.plot(pulse[:lim])
    plt.show()
    #plot_imf(signal)
    #plt.plot(data[k])

    #plt.show()


from scipy import signal
if __name__ == '__main__':
    samples = 983#728 #714, 724(4)
    path = "../data/stl_bpfo/sample{}_stl.csv".format(samples)
    df = pd.read_csv(path)


    k = 10 # at k=8, for 900 sample number we get 33.33Hz the machine ro
    data = df["slt_of_imf{}".format(k)]
    fs = 12600
    f,amp = signal.periodogram(data,fs=fs,window="hanning",scaling="spectrum")
    peaks = signal.find_peaks(amp[:1000])[0]
    #p = max(f[:300][peaks])
    #print(p)

    plt.plot(f[:1000],amp[:1000])
    plt.xlabel("Frequency in Hz")
    plt.ylabel("Amplitude")
    plt.title("Spectrum of the STL transform of imf number {}".format(k))
    plt.show()

    exit()
    #burst_amplitude()
    exit()
    exp_numb = 2
    channel = 0
    save_imf(exp_numb, channel)
    #get_imf(exp_numb,channel)
