import pyhht
from pyhht import EMD


def get_imfs(signal):
    """"
    Return the imf of a signal
    """
    decomposer = EMD(signal)
    imf = decomposer.decompose()
    return imf










if __name__ == '__main__':
    main()
