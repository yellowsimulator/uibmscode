import pyhht
from pyhht import EMD
import numpy as np

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
    x = np.linspace(-m,m,1000)
    pyhht.plot_imfs(x, imfs)
    plt.show()







if __name__ == '__main__':
    main()
