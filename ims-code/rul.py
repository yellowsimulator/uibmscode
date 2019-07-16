import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate
import scipy.optimize as optimization
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Flatten
from keras.layers import LSTM
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge





from etl import *
def main():
    exp_numb = 2
    out = "experiment{}_bearing_fft_trend.png".format(exp_numb)
    file_name = "experiment{}_bearing_fft_trend.csv".format(exp_numb)
    df = pd.read_csv(file_name)
    d = 9.81
    amp = d*df["amp1"].values
    trend1 = decompose(amp).trend
    t = np.linspace(0, 10*len(trend1),len(trend1) )
    plt.plot(t,trend1)
    plt.show()


def predict():
    df = pd.read_csv("data.csv")
    X = df["X"].values # trend amplitude
    y = df["y"].values # rul
    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


def get_trend(i):
    exp_numb = 2
    file_name = "experiment{}_bearing_fft_trend.csv".format(exp_numb)
    df = pd.read_csv(file_name)
    d = 9.81
    amp = d*df["amp{}".format(i)].values
    trend = decompose(amp).trend
    t = np.linspace(0, 10*len(trend),len(trend))
    return t,trend


def Ei(i):
    t,trend = get_trend(i)
    I = np.trapz(trend, t)
    return I


def rul(i):
    #a = 0.2
    A = least_square(i)
    a = np.abs(A[0])

    #print(A[0])
    #print(np.abs(A[0]))
    t,trend = get_trend(i)
    t1 = t[-1]
    if i == 1:
        b = trend[0]-a*t1
    else:
        b = trend[-1]-a*t1
    #print(b)

    #k = (1./3)*t1**3+b*t1
    k = a*(1./2)*t1**2+b*t1
    E = Ei(1)-Ei(i)

    a0 = -k-E
    #coeff = [1./3,0,b,a0]
    coeff = [a*(1./2),b,a0]

    #coeff = [A[0]*(1./2),b,a0]
    roots = np.roots(coeff)
    tf = roots[-1]
    print("E1",Ei(1))
    print("E2",Ei(2))
    print("k",k)
    print("a0",a0)
    print("Energy:",E)
    print("failing time:",t[-1])


    print(roots)

def plot_trend():
    fig, ax = plt.subplots()
    t,trend1 = get_trend(1)
    t,trend2 = get_trend(2)
    t,trend3 = get_trend(3)
    t,trend4 = get_trend(4)
    plt.plot(t,trend1,color="red")
    plt.plot(t,trend2,color="green")
    plt.plot(t,trend3,color="blue")
    plt.plot(t,trend4,color="orange")

    t1 = t[-1]
    plt.axvline(t1, linestyle='--', color='black')
    #plt.xlim([0,15000])
    plt.xlabel("Time in Minute")
    plt.ylabel("BPFO amplitude")
    plt.legend(["bearing1","bearing2","bearing3","bearing4","Failure time for bearing1"])
    plt.show()


def least_square(i):
    xdata, ydata = get_trend(i)
    lim = 0
    xdata, ydata = xdata[lim:], ydata[lim:]
    x0 = np.zeros(len(xdata))
    sigma = np.array([1. for _ in range(len(x0))])
    #coeff = optimization.leastsq(func,x0,args=(xdata,ydata))
    result = optimization.curve_fit(func,xdata,ydata)
    coeff = list(result[0])
    #print("before",coeff)
    coeff.reverse()
    print("after",coeff)

    return coeff

def func(t,a,b):
    return a*np.exp(t) + b


def plot_trendi(i):
    xdata, ydata = get_trend(i)
    grad = np.gradient(ydata)
    plt.plot(xdata,ydata)
    #plt.plot(xdata,grad)
    plt.xlabel("Time in minute")
    plt.ylabel("Trend amplitude")
    plt.title("BPFO amplitude's trend for bearing {}".format(i))
    plt.show()

def sklearn_linear_regression(i):
    X,Y = get_trend(i)
    X = X.reshape(-1,1)
    colors = ["red","green","orange"]
    poly = PolynomialFeatures(p).fit(X,Y)
    for count, degree in enumerate([2,3,4]):
        model = make_pipeline(PolynomialFeatures(degree), Ridge())
        model.fit(X,Y)
        y_plot = model.predict(X)
        score = round(model.score(X,Y),2)
        coeff = model.coef_
        print(coeff)
        plt.plot(X,y_plot,color=colors[count],label="degree {} score: {}".format(degree,score))
    plt.plot(X,Y,label="real",color="blue")
    plt.legend(loc="best")
    #score = reg.score(X,Y)
    #coeff = poly.get_params()
    #print("score: ",score)
    #print("coeff: ",coeff)
    #pred = poly.predict(X)
    #plt.plot(X,pred)
    #plt.plot(X,Y)
    #plt.legend(["pred","real"])
    plt.show()







if __name__ == '__main__':
    i = 1
    p = 2
    #sklearn_linear_regression(i)
    plot_trend()
    #t,trend = get_trend(i)
    #b = trend[-1]
    #t = 9840.0000357
    #t1 = 9840.
    #E = Ei(1)-Ei(4)
    #A = (1./3)*t**3 + b*t - (1./3)*t1**3 - b*t1 - E
    #print(A)
    #exit()
    #least_square(i)
    #rul(i)














#
