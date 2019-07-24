from sklearn.datasets import make_gaussian_quantiles
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.svm import SVC
from sklearn.datasets.samples_generator import make_circles



def visualize_2d(X,y,algorithm="tsne",title="Data in 2D",figsize=(8,8)):
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    if algorithm=="tsne":
        reducer = TSNE(n_components=2,random_state=47,n_iter=400,angle=0.6)
    elif algorithm=="pca":
        reducer = PCA(n_components=2,random_state=47)
    else:
        raise ValueError("Unsupported dimensionality reduction algorithm given.")
    if X.shape[1]>2:
        X = reducer.fit_transform(X)
    else:
        if type(X)==pd.DataFrame:
        	X=X.values
    f, (ax1) = plt.subplots(nrows=1, ncols=1,figsize=figsize)
    sns.scatterplot(X[:,0],X[:,1],hue=y,ax=ax1);
    ax1.set_title(title);
    plt.show();


def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


from mpl_toolkits import mplot3d

from ipywidgets import interact, fixed
X, y = make_circles(1000, factor=.1, noise=.1)
def plot_3D(elev=30, azim=30, X=X, y=y):
    r = np.exp(-(X ** 2).sum(1))
    ax = plt.subplot(projection='3d')
    scatter = ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=10)
    legend1 = ax.legend(*scatter.legend_elements(),loc="lower left", title="Classes")
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('r')
    plt.title("Linear separable classes 0 and 1")
    ax.add_artist(legend1)
    plt.show()

plot_3D(elev=30, azim=30, X=X, y=y)
#plot_svc_decision_function(model, ax=None, plot_support=True)
exit()
#interact(plot_3D, elev=[-90, 90], azip=(-180, 180),X=fixed(X), y=fixed(y));
#plt.show()






clf = SVC(kernel='linear').fit(X, y)
#ax = plt.subplot()
fig, ax = plt.subplots()
scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=10)
legend1 = ax.legend(*scatter.legend_elements(),loc="lower left", title="Classes")
plt.title("Nonlinear separable classes 0 and 1")
plt.xlabel("x")
plt.ylabel("y")
ax.add_artist(legend1)
#plt.legend(["class A"])
#plt.legend(["class B"])
#sns.scatterplot(X[:,0],X[:,1],hue=y,ax=ax1)

#plot_svc_decision_function(clf, plot_support=False);
plt.show()
# Construct dataset
#X1, y1 = make_gaussian_quantiles(cov=3.,n_samples=10000, n_features=2,n_classes=2, random_state=1)
#X1 = pd.DataFrame(X1,columns=['x','y'])
#y1 = pd.Series(y1)
#visualize_2d(X1,y1)
