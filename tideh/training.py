# Author:   Sylvain Gauthier
# Author:   Sebastian Rühl
#
# For license information, see LICENSE.txt

"""
Implements functions for training parameters of infectious rate of the twitter model. Parameters are trained by
minimizing the prediction error.

Provides implementations using native Python and optimized versions using nd-arrays and vectorization.

References
----------
.. Kobayashi, R. and Lambiotte, R., 2016, March. TiDeH: Time-Dependent Hawkes Process for Predicting Retweet Dynamics.
   In ICWSM (pp. 191-200).
"""

from . import main
from . import functions

import numpy as np
from scipy.optimize import minimize


def to_minimize(x, events_data, obs_time, pred_time, e_window_size, e_window_stride, kernel_int, p, kernel,
                dt, p_window):
    """
    Calculates training error for every given data_training set for given parameters and calculates the median training
    error used for minimization.

    :param x: current parameters for infectious rate
    :param events_data: list of all event_data tuples (one for each file)
    :param obs_time: observation time (in hours)
    :param pred_time: prediction time, i.e. the function predicts the number of retweets from obs_time to pred_time
    (in hours)
    :param e_window_size: bin width for estimation (in hours)
    :param e_window_stride: interval (stride size) for moving windows (in hours)
    :param kernel_int: kernel integral function for estimation
    :param p: infectious rate function
    :param kernel: kernel function for prediction
    :param dt: interval width for numerical integral calculation
    :param p_window: bin width for prediction (in hours)
    :return: median of training errors
    """

    pred_errors = []
    for event_data in events_data:
        if event_data[1][1][0] > obs_time:  # no events in observation time -> skip
            continue
        err = main.training_error(x, event_data, obs_time, pred_time, e_window_size, e_window_stride,
                                  kernel_int, p, kernel, dt, p_window)
        pred_errors.append(err)

    err_med = np.median(pred_errors)
    return err_med


def train(start, events_data, obs_time=6, pred_time=168, e_window_size=4, e_window_stride=1,
          kernel_int=functions.integral_zhao_vec, p=functions.infectious_rate_tweets_vec,
          kernel=functions.kernel_zhao_vec, dt=0.1, p_window=4, simplex=None):
    """
    Trains parameters of infectious rate. Uses Nelder-Mead.

    If a simplex matrix array object is passed it will overwrite the start values.

    :param start: start values
    :param events_data: list of all event_data tuples (one for each file)
    :param obs_time: observation time (in hours)
    :param pred_time: prediction time, i.e. the function predicts the number of retweets from obs_time to pred_time
    (in hours)
    :param e_window_size: bin width for estimation (in hours)
    :param e_window_stride: interval (stride size) for moving windows (in hours)
    :param kernel_int: kernel integral function for estimation
    :param p: infectious rate function
    :param kernel: kernel function for prediction
    :param dt: interval width for numerical integral calculation
    :param p_window: bin width for prediction (in hours)
    :param simplex: array-matrix object holding initial simple matrix
    :return: see scipy.optimize.minimize documentation
    """
    options = {
        'initial_simplex': simplex
    }

    return minimize(to_minimize, start, method='Nelder-Mead', args=(
        events_data, obs_time, pred_time, e_window_size, e_window_stride, kernel_int, p, kernel, dt, p_window),
                    tol=1e-4, options=options)


def to_minimize_optimized(x, events_data, obs_time, pred_time, e_window_size, e_window_stride, kernel_int, p, kernel,
                          dt, p_window):
    """
    Calculates training error for every given data_training set for given parameters and calculates the median training error
    used for minimization.

    Optimized version using numpy. Passed function should expect numpy arrays as input.

    :param x: current parameters for infectious rate
    :param events_data: list of all event_data tuples, should be in nd-array format (one for each file)
    :param obs_time: observation time (in hours)
    :param pred_time: prediction time, i.e. the function predicts the numbers of retweets from obs_time to pred_time
    (in hours)
    :param e_window_size: bin width for estimation (\delta_{obs} in Kobayashi and Lambiotte 2016) (in hours)
    :param e_window_stride: interval (stride size) for moving windows (in hours)
    :param kernel_int: kernel integral function for estimation
    :param p: infectious rate function
    :param kernel: kernel function for prediction
    :param dt: interval width for numerical integral calculation
    :param p_window: bin width for prediction (\delta_{pred} in Kobayashi and Lambiotte 2016) (in hours)
    :return: median of training errors
    """
    pred_errors = []
    for event_data in events_data:
        if event_data[1][0][1] > obs_time:  # no events in observation time -> skip
            continue
        err = main.training_error_optimized(x, event_data, obs_time, pred_time, e_window_size, e_window_stride,
                                            kernel_int, p, kernel, dt, p_window)
        pred_errors.append(err)

    err_med = np.median(pred_errors)
    return err_med


def train_optimized(start, events_data, obs_time=6, pred_time=168, e_window_size=4, e_window_stride=1,
                    kernel_int=functions.integral_zhao_vec, p=functions.infectious_rate_tweets_vec,
                    kernel=functions.kernel_zhao_vec, dt=0.1, p_window=4, simplex=None):
    """
    Trains parameters of infectious rate. Uses Nelder-Mead.

    If a simplex matrix array object is passed it will overwrite the start values.

    Optimized version using numpy. Passed function should expect numpy arrays as input.

    :param start: start values
    :param events_data: list of all event_data tuples, should be in nd-array format (one for each file)
    :param obs_time: observation time (in hours)
    :param pred_time: prediction time, i.e. the function predicts the numbers of retweets from obs_time to pred_time
    (in hours)
    :param e_window_size: bin width for estimation (\delta_{obs} in Kobayashi and Lambiotte 2016) (in hours)
    :param e_window_stride: interval (stride size) for moving windows (in hours)
    :param kernel_int: kernel integral function for estimation
    :param p: infectious rate function
    :param kernel: kernel function for prediction
    :param dt: interval width for numerical integral calculation
    :param p_window: bin width for prediction (\delta_{pred} in Kobayashi and Lambiotte 2016) (in hours)
    :param simplex: array-matrix object holding initial simple matrix
    :return: see scipy.optimize.minimize documentation
    """
    options = {
        'initial_simplex': simplex
    }

    return minimize(to_minimize_optimized, start, method='Nelder-Mead',
                    args=(
                        events_data, obs_time, pred_time, e_window_size, e_window_stride, kernel_int, p, kernel, dt,
                        p_window),
                    tol=1e-4, options=options)
