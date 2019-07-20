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



def plot_imf(signal):
    decomposer = EMD(signal)
    imfs = decomposer.decompose()
    m = len(signal)
    x = np.linspace(0,1,m)
    pyhht.plot_imfs(x, imfs)
    plt.show()



def get_imf(exp_numb,channel):
    samples = 100#728 #714, 724(4)
    path = "../data/imfs/sample{}_imfs.csv".format(samples)
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



if __name__ == '__main__':
    exp_numb = 2
    channel = 0
    get_imf(exp_numb,channel)
