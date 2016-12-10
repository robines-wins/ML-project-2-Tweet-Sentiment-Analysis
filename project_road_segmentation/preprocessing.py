# This file contains the method used to preprocess the data
import numpy as np

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    toReturn = np.power(x,1)
    for i in range(2,degree+1):
        toReturn = np.hstack([toReturn,np.power(x,i)])
    return standardize(toReturn)

def standardize(x, mean_x=None, std_x=None):
    """Standardize the original data set. And adds a 1 as the first column"""
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    x = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis=0)
    x[:, std_x>0] = x[:, std_x>0] / std_x[std_x>0]
    tx = np.hstack((np.ones((x.shape[0],1)), x))
    return tx, mean_x, std_x


def de_standardize(x, mean_x, std_x):
    """Reverse the procedure of standardization."""
    x = x * std_x
    x = x + mean_x
    return x


def build_model_data(output_data, input_data):
    """Form (y,tX) to get regression data in matrix form."""
    y = output_data
    x = input_data
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx