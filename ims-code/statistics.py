"""
Objective of this file:
----------------------
In this file we implement statistical methods.
In particular the health index that quantifies
the anomaly of a sample, as well as other
important statistics.
"""

from multiprocessing import Pool
from scipy.stats import iqr
import numpy as np


def get_health_index(dispersion_index, sample):
    """
    Compute the health index for a given
    sample.
    Arguments:
    ---------
    dispersion_index:
        its the statistical quantity
        quantifying the dispersion.
        example: iqr, variance,...
    sample:
        its the sample for which the dispersion
        index is computed.
    Return:
    ------
    health_index:
        the health index of the sample
    """
    if dispersion_index == "iqr":
        health_index = iqr(sample)
    elif dispersion_index == "variance":
        health_index = np.var(sample)
    return health_index


def get_all_health_index(dispersion_index, samples):
    """
    Return all index for a list of samples.
    Arguments:
    ---------
    dispersion_index:
        either the string "variance" for variance or
        the string "iqr" for the interquantile range.
    sample:
        an array containing the sample
    Return:
    ------
    list of all health index
    """
    with Pool() as p:
        health_indexes = [p.apply(get_health_index,
                 args=(dispersion_index, sample)) for sample in samples]
        return health_indexes


def get_health_index_function(health_indexes,n,a):
    """
    Implement the health index function. (not conclusive !!!!!)
    Arguments:
    ---------
    first_health_index:
        health index of the first sample.
    last_health_index:
        health index of the last sample.
    a:
        the coefficients a2,a3,...,an
    Return:
    ------
    health_function:
        the health index function
    """
    health_function = sum(np.log(a)) + np.log(health_indexes[0]/health_indexes[n])
    return health_function


def get_health_index_coefficient(health_index_array, j):
    """
    Compute the health index coefficient a
    Arguments:
    ---------
    health_index_j:
        health index of sample j
    health_index_j_minus:
        health index of sample j-1
    Return:
        coeficient a.
    """
    return float(health_index_array[j]/health_index_array[j-1])


def get_all_health_index_coefficients(health_index_array):
    """
    Return all health index coeficienta a2,...,an
    Arguments:
    ---------
    health_indexes:
        all health indexes
    Return:
        list of coefficients a2,..,an
    """
    indexes = range(len(health_index_array))
    with Pool() as p:
        coefficients = [p.apply(get_health_index_coefficient,
                args = (health_index_array, j)) for j in indexes]
        return coefficients






if __name__ == '__main__':
    print("ok")











    #
