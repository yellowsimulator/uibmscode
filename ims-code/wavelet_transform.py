import pywt


def get_high_low_freq(data):
    """
    Returns the low (cA) and hight (cD) frequency
    computed from the discrete wavelet Transform
    """
    cA, cD = pywt.dwt(data, "db10")
    return cA, cD

















if __name__ == '__main__':
    main()
