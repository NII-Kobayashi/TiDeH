# Author:   Sylvain Gauthier
# Author:   Sebastian Rühl
#
# For license information, see LICENSE.txt

"""
Implements basic mathematical expression of different functions used for estimation, prediction and simulation.
This includes the memory kernel, integral of memory kernel and the infectious rate.

Provides implementations using native Python and optimized versions using nd-arrays and vectorization.

References
----------
.. Kobayashi, R. and Lambiotte, R., 2016, March. TiDeH: Time-Dependent Hawkes Process for Predicting Retweet Dynamics.
   In ICWSM (pp. 191-200).
.. Zhao, Q., Erdogdu, M.A., He, H.Y., Rajaraman, A. and Leskovec, J., 2015, August. Seismic: A self-exciting point
   process model for predicting tweet popularity. In Proceedings of the 21th ACM SIGKDD International Conference on
   Knowledge Discovery and Data Mining (pp. 1513-1522). ACM.
"""

from math import *
import numpy as np


def kernel_zhao(s, s0=0.08333, theta=0.242):
    """
    Calculates Zhao kernel for given value.

    :param s: time point to evaluate
    :param s0: initial reaction time
    :param theta: empirically determined constant
    :return: value at time point s
    """
    c0 = 1.0 / s0 / (1 - 1.0 / -theta)  # normalization constant
    if s >= 0:
        if s <= s0:
            return c0
        else:
            return c0 * (s / s0) ** (-(1. + theta))
    else:
        return 0


def kernel_zhao_vec(s, s0=0.08333, theta=0.242):
    """
    Calculates Zhao kernel for given value.
    Optimized using nd-arrays and vectorization.

    :param s: time points to evaluate, should be a nd-array
    :param s0: initial reaction time
    :param theta: empirically determined constant
    :return: values at given time points
    """
    c0 = 1.0 / s0 / (1 - 1.0 / -theta)  # normalization constant
    res = np.copy(s)
    res[s < 0] = 0
    res[(s <= s0) & (s >= 0)] = c0
    res[s > s0] = c0 * (res[s > s0] / s0) ** (-(1. + theta))
    return res


def kernel_primitive_zhao(x, s0=0.08333, theta=0.242):
    """
    Calculates the primitive of the Zhao kernel for given values.

    :param x: point to evaluate
    :param s0: initial reaction time
    :param theta: empirically determined constant
    :return: primitive evaluated at x
    """
    c0 = 1.0 / s0 / (1 - 1.0 / -theta)
    if x < 0:
        return 0
    elif x <= s0:
        return c0 * x
    else:
        return c0 * (s0 + (s0 * (1 - (x / s0) ** -theta)) / theta)


def kernel_primitive_zhao_vec(x, s0=0.08333, theta=0.242):
    """
    Calculates the primitive of the Zhao kernel for given values.
    Optimized using nd-arrays and vectorization.

    :param x: points to evaluate, should be a nd-array
    :param s0: initial reaction time
    :param theta: empirically determined constant
    :param c0: normalization constant
    :return: primitives evaluated at given points
    """
    c0 = 1.0 / s0 / (1 - 1.0 / -theta)
    res = np.copy(x)
    res[x < 0] = 0
    res[(x <= s0) & (x >= 0)] = c0 * res[(x <= s0) & (x >= 0)]
    res[x > s0] = c0 * (s0 + (s0 * (1 - (res[x > s0] / s0) ** -theta)) / theta)
    return res


def integral_zhao(x1, x2, s0=0.08333, theta=0.242):
    """
    Calculates definite integral of Zhao function.

    :param x1: start
    :param x2: end
    :param s0: initial reaction time
    :param theta: empirically determined constant
    :return: integral of Zhao function
    """
    return kernel_primitive_zhao(x2, s0, theta) - kernel_primitive_zhao(x1, s0, theta)


def integral_zhao_vec(x1, x2, s0=0.08333, theta=0.242):
    """
    Calculates definite integral of Zhao function.
    Optimized using nd-arrays and vectorization.

    x1 and x2 should be nd-arrays of same size.

    :param x1: start values
    :param x2: end values
    :param s0: initial reaction time
    :param theta: empirically determined constant
    :return: integrals of Zhao function
    """
    return kernel_primitive_zhao_vec(x2, s0, theta) - kernel_primitive_zhao_vec(x1, s0, theta)


def infectious_rate_tweets(t, p0=0.001, r0=0.424, phi0=0.125, taum=2., t0=0, tm=24, bounds=None):
    """
    Alternative form of infectious rate from paper. Supports bounds for r0 and taum. Bounds should be passed as an array
    in the form of [(lower r0, lower taum), (upper r0, upper taum)].
    Converted to hours.

    :param t: point to evaluate function at (in hours)
    :param p0: base rate
    :param r0: amplitude
    :param phi0: shift (in days)
    :param taum: decay/freshness (in days)
    :param t0: start time of observation (in hours)
    :param tm: cyclic property (after what time a full circle passed, in hours)
    :param bounds: bounds for r0 and taum
    :return: intensity for point t
    """
    if bounds is not None:
        if not (bounds[0][0] < r0 < bounds[1][0]):
            r0 = max(bounds[0][0], bounds[1][0] * sigmoid(taum / bounds[1][0]))
        if not (bounds[0][1] < taum < bounds[1][1]):
            taum = max(bounds[0][1], bounds[1][1] * sigmoid(taum / bounds[1][1]))
    return p0 * (1. - r0 * sin((48 / tm) * pi * ((t + t0) / 24 + phi0))) * exp(-t / (24 * taum))


def infectious_rate_tweets_vec(t, p0=0.001, r0=0.424, phi0=0.125, taum=2., t0=0, tm=24., bounds=None):
    """
    Alternative form of infectious rate from paper. Supports bounds for r0 and taum. Bound should be passed as an array
    in the form of [(lower r0, lower taum), (upper r0, upper taum)].
    Converted to hours.
    Vectorized version.

    :param t: points to evaluate function at, should be a nd-array (in hours)
    :param p0: base rate
    :param r0: amplitude
    :param phi0: shift (in days)
    :param taum: decay/freshness (in days)
    :param t0: start time of observation (in hours)
    :param tm: cyclic property (after what time a full circle passed, in hours)
    :param bounds: bounds for r0 and taum
    :return: intensities for given points
    """
    if bounds is not None:
        if not (bounds[0][0] < r0 < bounds[1][0]):
            r0 = max(bounds[0][0], bounds[1][0] * sigmoid(taum / bounds[1][0]))
        if not (bounds[0][1] < taum < bounds[1][1]):
            taum = max(bounds[0][1], bounds[1][1] * sigmoid(taum / bounds[1][1]))
    return p0 * (1. - r0 * np.sin((48. / tm) * np.pi * ((t + t0) / 24. + phi0))) * np.exp(-t / (24. * taum))


def sigmoid(x):
    """
    Calculates sigmoid function for value x.
    """
    return 1 / (1 + exp(-x))
