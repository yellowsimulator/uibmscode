"""
==========================================
One-class SVM with non-linear kernel (RBF)
==========================================

An example using a one-class SVM for novelty detection.

:ref:`One-class SVM <svm_outlier_detection>` is an unsupervised
algorithm that learns a decision function for novelty detection:
classifying new data as similar or different to the training set.
"""
print(__doc__)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
dk = 30
xx, yy = np.meshgrid(np.linspace(0, 1, 500), np.linspace(0, 0.5, 500))
# Generate train data
path = "../data/wavelet/bearing_debauchies_{}_iqr.csv".format(dk)
df = pd.read_csv(path)
m = 983
X_train = df.loc[m+1:4*m-1,"cA_iqr":"cD_iqr"].values
X_test = df.loc[:m,"cA_iqr":"cD_iqr"].values
bearing2 = df.loc[m+1:2*m-1,"cA_iqr":"cD_iqr"].values
bearing3 = df.loc[2*m+1:3*m-1,"cA_iqr":"cD_iqr"].values
bearing4 = df.loc[3*m+1:4*m-1,"cA_iqr":"cD_iqr"].values
#print(X_train.shape)
#exit()
# fit the model
nu_value = 0.0001
gamma_value = 0.1
clf = svm.OneClassSVM(nu=nu_value, kernel="rbf", gamma=gamma_value)
clf.fit(X_train)

Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("Bearing anomaly detection with one-class SVM")
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='green')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='white')

s = 30

b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='r', s=s,edgecolors='k')
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], s=s, edgecolors='k')

b3 = plt.scatter(bearing2[:, 0], bearing2[:, 1], c='blue', s=s)
b4 = plt.scatter(bearing3[:, 0], bearing3[:, 1], c='green', s=s)
b5 = plt.scatter(bearing4[:, 0], bearing4[:, 1], c='orange', s=s)

plt.axis('tight')
plt.xlim((0, 1))
plt.ylim((0, 0.5))
#plt.xlabel("Frequency feature health index")
#plt.ylabel("Temporal feature health index")
plt.xlabel("Low frequency feature health index")
plt.ylabel("High frequency feature health index")
plt.legend([a.collections[0], b2,b3,b4,b5],
           ["learned frontier", "Bearing1",
            "Bearing2", "Bearing3", "Bearing4"],
           loc="upper left",
           prop=matplotlib.font_manager.FontProperties(size=11))
##"error train: %d/200 ; errors novel regular: %d/40 ; "
    #"errors novel abnormal: %d/40"
    #% (n_error_train, n_error_test, n_error_outliers))
plt.show()
