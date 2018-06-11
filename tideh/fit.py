# Author:   Sylvain Gauthier
# Author:   Sebastian Rühl
#
# For license information, see LICENSE.txt

"""
Provides an interface for fitting parameters of infectious rate.

Used for modeling infectious rate of a tweet. The estimated values of the instantaneous infectious rate are used here
for fitting the model of the infectious rate to them.

References
----------
.. Kobayashi, R. and Lambiotte, R., 2016, March. TiDeH: Time-Dependent Hawkes Process for Predicting Retweet Dynamics.
   In ICWSM (pp. 191-200).
"""

from scipy.optimize import leastsq
import numpy as np


def loss_function(params, estimates, fun, xval):
    """
    Loss function used by least squares.

    :param params: current values of parameters to fit
    :param estimates: estimates of function to fit
    :param fun: function to fit
    :param xval: x axis values to estimates (time points)
    :return: array of loss values for every estimation
    """
    return [(fun(xval[i], *params) - estimates[i]) for i in range(len(estimates))]


def fit_parameter(estimates, fun, start_values, xval):
    """
    Fitting any numbers of given infectious rate function using least squares.
    Used count of observed events in observation window as weights.

    :param estimates: estimated values of function
    :param fun: function to fit
    :param start_values: initial guesses of parameters, should be a ndarray
    :param xval: x axis values to estimates (time points)
    :return: fitted parameters
    """
    if start_values is None:
        start_values = np.array([0, 0, 0, 1.])

    return leastsq(func=loss_function, x0=start_values, args=(estimates, fun, xval))[0]


def error(estimated, fitted):
    """
    Calculates mean percentage error for fitted values to estimated values.

    :param estimated: estimated values
    :param fitted: fitted values
    :return: percent error
    """
    return sum([abs(e / f - 1) for e, f in zip(estimated, fitted)]) / len(estimated)
