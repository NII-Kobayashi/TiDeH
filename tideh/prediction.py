# Author:   Sylvain Gauthier
# Author:   Sebastian Rühl
#
# For license information, see LICENSE.txt

"""
Implements functions for predicting future retweets.

Provides implementations using native Python and optimized versions using nd-arrays and vectorization.

References
----------
.. Kobayashi, R. and Lambiotte, R., 2016, March. TiDeH: Time-Dependent Hawkes Process for Predicting Retweet Dynamics.
   In ICWSM (pp. 191-200).
"""

from . import functions as fct
import numpy as np


def predict_nb_retweets(events, obs_time, pred_time, p=fct.infectious_rate_tweets, p_max=0.001424,
                        kernel=fct.kernel_zhao, dt=0.1):
    """
    Predicts the future infectious for given events and infectious rate. Parameters of infectious rate should already be
    present in the function itself.

    Parameter p_max can be used to hinder the prediction to burst.

    :param events: array of event tuples (event_time, follower_count)
    :param obs_time: observation time (in hours)
    :param pred_time: prediction time, i.e. the function predicts the number of retweets from obs_time to pred_time
    (in hours)
    :param kernel: kernel function
    :param dt: interval width for numerical integral calculation
    :param p: infectious rate function
    :param p_max: maximum value of p
    :return: 3-tuple holding estimated integral values, total number of predicted retweets, and total prediction error
    """
    dp = sum([fol_count for _, fol_count in events[1:]]) / len(events[1:])
    if p_max is not None:
        dp = min(dp, 1. / p_max)
    t = obs_time
    lambda_t = [p(t) * sum([fol_count * kernel(t - event_time) for event_time, fol_count in events])]
    times = [t]
    i = 1

    while t < pred_time - dt:
        t += dt
        integral = kernel(i * dt) * lambda_t[0] / 2 + sum([lambda_t[i] * kernel(t - (i * dt + obs_time))
                                                           for i in range(1, len(lambda_t))])
        ft = p(t) * sum([fol_count * kernel(t - event_time) for event_time, fol_count in events])
        alpha = dp * p(t) * dt
        ctk = 1. / (1 - alpha * kernel(0) / 2.)
        lambda_t.append(ctk * (ft + alpha * integral))
        times.append(t)
        i += 1

    return lambda_t, dt * sum(lambda_t), times


def predict_nb_retweets_vec(event_times, follower, obs_time, pred_time, p=fct.infectious_rate_tweets_vec,
                            p_max=0.001424, kernel=fct.kernel_zhao_vec, dt=0.1):
    """
    Predicts the future infectious for given events and infectious rate. Parameters of infectious rate should already be
    present in the function itself.

    Parameter p_max can be used to hinder the prediction to burst.

    Optimized version using numpy. Passed functions should expect numpy arrays as input.

    :param event_times: nd-array of event_times
    :param follower: nd-array of follower counts
    :param obs_time: observation time (in hours)
    :param pred_time: prediction time, i.e. the function predicts the number of retweets from obs_time to pred_time
    (in hours)
    :param kernel: kernel function
    :param dt: interval width for numerical integral calculation
    :param p: infectious rate function
    :param p_max: maximum value of p
    :return: 3-tuple holding estimated integral values, total number of predicted retweets, and total prediction error
    """
    dp = follower[1:].mean()  # exclude initial tweet for better results
    if p_max is not None:
        dp = min(dp, 1. / p_max)  # non bursting condition

    dts = np.arange(obs_time, pred_time, dt)  # intervals to calculate integral on
    p_t = p(dts)  # infectious rate for given intervals
    # sum of memory kernel for every interval
    k_sum = np.array([np.sum(follower * kernel(dtt - event_times)) for dtt in dts])
    ft = p_t * k_sum  # retweet probability for every interval
    ker = kernel(np.maximum(dts[-1] - dts, 0))  # kernel values for every interval, will be ordered from last to first
    alpha = dp * p_t * dt
    c_tk = 1. / (1 - alpha * kernel(np.array([0.])) / 2.)
    lambda_t = np.array([ft[0]])  # nd-array to store calculated integral values

    for i in range(1, dts.size):
        integral = ker[-(i + 1)] * lambda_t[0] / 2. + np.sum((ker[-i:][:-1] * lambda_t[1:]))
        lambda_t = np.append(lambda_t, c_tk[i] * (ft[i] + alpha[i] * integral))

    return lambda_t, dt * sum(lambda_t)
