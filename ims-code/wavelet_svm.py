import numpy as np
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import iqr
from etl import get_experiment_bearing_data
from wavelet_transform import *




def compute_ca_cd_and_save(exp_numb,channel,k):
    signals = get_experiment_bearing_data(exp_numb,channel)
    cA_iqr_list = []
    cD_iqr_list = []
    for j,signal in enumerate(signals):
        print("performing wavelet transform for bearing{} sample {}".format(channel+1,j))
        cA, cD = get_wavelet(signal,k)
        print("computing iqr for bearing{} sample {}".format(channel+1,j))
        cA_iqr_list.append(iqr(cA))
        cD_iqr_list.append(iqr(cD))
    d = {"cA_iqr":list(cA_iqr_list),"cD_iqr":list(cD_iqr_list)}
    df = pd.DataFrame(d)
    print("saving data for bearing{} sample {}".format(channel+1,j))
    df.to_csv("../data/wavelet/bearing{}_debauchies_{}_iqr.csv".format(channel+1,k))
    cA_iqr_list = []
    cD_iqr_list = []


def process_all_bearing(k):
    exp_numb = 2
    for channel in [0,1,2,3]:
        compute_ca_cd_and_save(exp_numb,channel,k)


def plot_data(k,save=True):
    colors = ["red","blue","green","orange"]
    for j in [1,2,3,4]:
        path = "../data/wavelet/bearing{}_debauchies_{}_iqr.csv".format(j,k)
        df = pd.read_csv(path)
        cA_iqr = df["cA_iqr"].values
        cD_iqr = df["cD_iqr"].values
        plt.scatter(cA_iqr, cD_iqr,color=colors[j-1],s=20)
    plt.legend(["bearing1 severely damaged","bearing2","bearing3","bearing4"])
    plt.xlabel("Frequency feature health index")
    plt.ylabel("Temporal feature health index")
    plt.title("Bearings health trajectories in temporal-frequency feature space")
    if save:
        plt.savefig("/Users/yapiachou/UiBergen/UiB-master/fig/health_plot.png")
    else:
        plt.show()


def plot_frequency_feature(k,save=True):
    colors = ["red","blue","green","orange"]
    for j in [1,2,3,4]:
        path = "../data/wavelet/bearing{}_debauchies_{}_iqr.csv".format(j,k)
        df = pd.read_csv(path)
        cA_iqr = df["cA_iqr"].values
        #cD_iqr = df["cD_iqr"].values
        x = np.linspace(0,10*len(cA_iqr),len(cA_iqr))
        plt.plot(x, cA_iqr,color=colors[j-1])
    plt.legend(["bearing1 severely damaged","bearing2","bearing3","bearing4"])
    plt.xlabel("Time in minute")
    plt.ylabel("Frequency feature health index")
    plt.title("Bearings health in terms of Frequency feature")
    if save:
        plt.savefig("/Users/yapiachou/UiBergen/UiB-master/fig/frequency_feature_health.png")
    else:
        plt.show()


def plot_temporal_feature(k,save=True):
    colors = ["red","blue","green","orange"]
    for j in [1,2,3,4]:
        path = "../data/wavelet/bearing{}_debauchies_{}_iqr.csv".format(j,k)
        df = pd.read_csv(path)
        #cA_iqr = df["cA_iqr"].values
        cD_iqr = df["cD_iqr"].values
        x = np.linspace(0,10*len(cD_iqr),len(cD_iqr))
        plt.plot(x, cD_iqr,color=colors[j-1])
    plt.legend(["bearing1 severely damaged","bearing2","bearing3","bearing4"])
    plt.xlabel("Time in minute")
    plt.ylabel("Temporal feature health index")
    plt.title("Bearings health in terms of temporal feature")
    if save:
        plt.savefig("/Users/yapiachou/UiBergen/UiB-master/fig/temporal_feature_health.png")
    else:
        plt.show()


def create_svm_data(k):
    print("working")
    path1 = "../data/wavelet/bearing1_debauchies_{}_iqr.csv".format(k)
    path2 = "../data/wavelet/bearing2_debauchies_{}_iqr.csv".format(k)
    path3 = "../data/wavelet/bearing3_debauchies_{}_iqr.csv".format(k)
    path4 = "../data/wavelet/bearing4_debauchies_{}_iqr.csv".format(k)
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    df3 = pd.read_csv(path3)
    df4 = pd.read_csv(path4)

    df_temp1 = df1.append(df2)
    df_temp2 = df_temp1.append(df3)
    df = df_temp2.append(df4)
    path = "../data/wavelet/bearing_debauchies_{}_iqr.csv".format(k)
    df.to_csv(path,index=0)
    print("done")


def plot_tst(k):
    colors = ["red","blue","green","orange"]
    #for j in [1,2,3,4]:
    path = "../data/wavelet/bearing_debauchies_{}_iqr.csv".format(k)
    df = pd.read_csv(path)

    m = 983
    bearing1 = df.loc[:m,"cA_iqr":"cD_iqr"]
    bearing2 = df.loc[m+1:2*m-1,"cA_iqr":"cD_iqr"]
    #print(bearing2)
    #exit()
    bearing3 = df.loc[2*m+1:3*m-1,"cA_iqr":"cD_iqr"]
    bearing4 = df.loc[3*m+1:4*m-1,"cA_iqr":"cD_iqr"]
    for j,dataframe in enumerate([bearing1,bearing2,bearing3,bearing4]):
        cA_iqr = list(dataframe["cA_iqr"])
        cD_iqr = list(dataframe["cD_iqr"])
        plt.scatter(cD_iqr, cA_iqr,color=colors[j],s=10)
    plt.legend(["bearing1 severely damaged","bearing2","bearing3","bearing4"])
    plt.xlabel("Temporal feature health index")
    plt.ylabel("Frequency feature health index")
    plt.title("Bearings health trajectories in temporal-frequency feature space")
    plt.show()
    #To learn more about semantic completion, see https://tabnine.com/semantic.plt.show()

def d3plot():
    fig = plt.figure()
    dk = 20
    path = "../data/wavelet/bearing_debauchies_{}_iqr.csv".format(dk)
    df = pd.read_csv(path)
    m = 983
    bearing1 = df.loc[:m,"cA_iqr":"cD_iqr"].values
    bearing3 = df.loc[2*m+1:3*m-1,"cA_iqr":"cD_iqr"].values
    bearing4 = df.loc[3*m+1:4*m-1,"cA_iqr":"cD_iqr"].values
    bearing2 = df.loc[m+1:2*m-1,"cA_iqr":"cD_iqr"].values
    X_train = df.loc[m+1:4*m-1,"cA_iqr":"cD_iqr"].values
    xdata = bearing1[:, 0]
    ydata =  bearing1[:, 1]
    ax = plt.axes(projection='3d')
    zdata = np.linspace(0, len(bearing1[:, 0]),len(bearing1[:, 0]))

    xdata2 = bearing2[:, 0]
    ydata2 =  bearing2[:, 1]
    zdata2 = np.linspace(0, len(bearing2[:, 0]),len(bearing2[:, 0]))

    xdata3 = bearing3[:, 0]
    ydata3 =  bearing3[:, 1]
    zdata3 = np.linspace(0, len(bearing3[:, 0]),len(bearing3[:, 0]))

    xdata4 = bearing4[:, 0]
    ydata4 =  bearing4[:, 1]
    zdata4 = np.linspace(0, len(bearing4[:, 0]),len(bearing4[:, 0]))

    ax.scatter3D(xdata, ydata, zdata,color="red");
    ax.scatter3D(xdata2, ydata2, zdata2,color="green");
    ax.scatter3D(xdata3, ydata3, zdata3,color="blue");
    ax.scatter3D(xdata4, ydata4, zdata4,color="yellow");

    ax.set_xlim(0,max(xdata))
    ax.set_ylim(0,max(ydata))
    ax.set_zlim(0,max(zdata))
    ax.set_xlabel("High frequency feature")
    ax.set_ylabel("Low frequency feature")
    ax.set_zlabel("Time in minute")
    plt.show()


if __name__ == '__main__':
    d3plot()
    #k = 30
    K = [2,3,4,6,7,8,9,11,12,13,14,16,17,18,19,21,22,23,24,25,26,27,28,29]

    #process_all_bearing(k)
    #create_svm_data(k)
    #process_all_bearing()
    #plot_data(k)
