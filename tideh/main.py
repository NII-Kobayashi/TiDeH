# Author:   Sylvain Gauthier
# Author:   Sebastian Rühl
#
# For license information, see LICENSE.txt

"""
Front-end module that offers high-level functions for easy access to parameter estimation, prediction and training.
Also provides functions to load data from files.

Provides implementations using native Python and optimized versions using nd-arrays and vectorization.

References
----------
.. Kobayashi, R. and Lambiotte, R., 2016, March. TiDeH: Time-Dependent Hawkes Process for Predicting Retweet Dynamics.
   In ICWSM (pp. 191-200).
"""

from . import functions
from . import estimate
from . import fit
from . import prediction
from . import training
import numpy as np


def estimate_parameters(events, obs_time=None, window_size=4, window_stride=1, kernel_int=functions.integral_zhao,
                        p=functions.infectious_rate_tweets, values=None, **p_params):
    """
    Estimate parameters of infectious rate function.
    The given infectious rate function should contain the parameters to estimate before any other parameters besides the
    evaluation parameters in its function signature.

    :param events: array of event tuples containing (event_time, follower_cnt)
    :param obs_time: observation time (in hours)
    :param kernel_int: integral function of kernel function
    :param window_size: bin width for estimation (\delta_{obs} in Kobayashi and Lambiotte 2016) (in hours)
    :param window_stride: interval (stride size) for moving windows (in hours)
    :param p: infectious rate function
    :param values: initial guesses of parameters to estimate
    :param p_params: additional named parameters (dict) passed to infectious rate function
    :return: 3-tuple holding array of estimated parameters, error, and 3-tuple holding return values of estimation
    function
    """
    obs_time = int(min(obs_time, events[len(events) - 1][0]))
    events = [e for e in events if e[0] < obs_time]  # e[0] holds time of the event

    estimations, window_event_count, window_middle = \
        estimate.estimate_infectious_rate(events, kernel_int, obs_time, window_size, window_stride)

    fitted = fit.fit_parameter(estimates=estimations, fun=lambda x, *args: p(x, *args, **p_params),
                               start_values=values, xval=window_middle)

    err = fit.error(estimations, [p(i, *fitted, **p_params) for i in window_middle])

    return fitted, err, (window_middle, estimations, window_event_count)


def estimate_parameters_optimized(event_times, follower, obs_time=6, window_size=4, window_stride=1,
                                  kernel_int=functions.integral_zhao_vec, p=functions.infectious_rate_tweets_vec,
                                  values=None, **p_params):
    """
    Estimate parameters of infectious rate function.
    The given infectious rate function should contain the parameters to estimate before any other parameters besides the
    evaluation parameters in its function signature.
    Optimized using nd-arrays and vectorization.

    :param event_times: nd-array of event times
    :param follower: nd-array of follower counts
    :param obs_time: observation time (in hours)
    :param kernel_int: integral function of kernel function
    :param window_size: bin width for estimation (\delta_{obs} in Kobayashi and Lambiotte 2016) (in hours)
    :param window_stride: interval (stride size) for moving windows (in hours)
    :param p: infectious rate function
    :param values: initial guesses of parameters to estimate
    :param p_params: additional named parameters (dict) passed to infectious rate function
    :return: 3-tuple holding array of estimated parameters, error, and 3-tuple holding return values of estimation
    function
    """
    filt_obs = event_times < obs_time

    estimations, window_event_count, window_middle = \
        estimate.estimate_infectious_rate_vec(event_times=event_times[filt_obs], follower=follower[filt_obs],
                                              kernel_integral=kernel_int, obs_time=obs_time, window_size=window_size,
                                              window_stride=window_stride)

    fitted = fit.fit_parameter(estimates=estimations, fun=lambda x, *args: p(x, *args, **p_params),
                               start_values=values, xval=window_middle)

    err = fit.error(estimations, [p(i, *fitted, **p_params) for i in window_middle])

    return fitted, err, (window_middle, estimations, window_event_count)


def predict(events, obs_time=6, pred_time=168, window=4, kernel=functions.kernel_zhao, dt=0.1,
            p=functions.infectious_rate_tweets, p_max=0.001424, params=None, **p_params):
    """
    Predicts infectious rate for given prediction period and infectious rate. Also calculates prediction error for given
    prediction window size if there are events available for the prediction period.

    Parameter p_max can be used to hinder the prediction to burst.

    :param events: array of event tuples (event_time, follower_count)
    :param obs_time: observation time (in hours)
    :param pred_time: prediction time, i.e. the function predicts the number of retweets from obs_time to pred_time
    (in hours)
    :param window: bin width for prediction (\delta_{pred} in Kobayashi and Lambiotte 2016) (in hours)
    :param kernel: kernel function
    :param dt: interval width for numerical integral calculation
    :param p: infectious rate function
    :param p_max: maximum value of p
    :param params: possibility to pass parameters in array format to infectious rate, should follow correct parameter
    ordering, alternative way is to pass all parameters within p_params
    :param p_params: additional named parameters (dict) passed to infectious rate function
    :return: 3-tuple holding estimated integral values, total number of predicted retweets, and total prediction error
    """
    events_filt = [ev for ev in events if ev[0] <= obs_time]

    if params is None:
        params = []

    lambda_t, total, _ = prediction.predict_nb_retweets(events=events_filt, obs_time=obs_time, pred_time=pred_time,
                                                        p=lambda x: p(x, *params, **p_params), p_max=p_max, dt=dt,
                                                        kernel=kernel)

    events_time_pred = np.array([ev for ev, _ in events if ev > obs_time])
    win_int = int(window / dt)
    tp = np.arange(obs_time, pred_time, window)
    l_t = np.array(lambda_t)

    err = 0
    for i, t_cur in enumerate(tp):
        t_end = t_cur + window
        if t_end > pred_time:
            break
        count_current = estimate.get_event_count(events_time_pred, t_cur, t_end)
        pred_count = dt * l_t[(i * win_int):((i + 1) * win_int)].sum()
        err += abs(count_current - pred_count)

    return lambda_t, total, err


def predict_optimized(event_times, follower, obs_time=6, pred_time=168, window=4, kernel=functions.kernel_zhao_vec,
                      dt=0.1, p=functions.infectious_rate_tweets_vec, p_max=0.001424, params=None, **p_params):
    """
    Predicts infectious rate for given prediction period and infectious rate. Also calculates prediction error for given
    prediction window size if there are events available for the prediction period.

    Parameter p_max can be used to hinder the prediction to burst.

    Optimized version using numpy. Passed functions should expect numpy arrays as input.

    :param event_times: nd-array of event times
    :param follower: nd-array of follower counts
    :param obs_time: observation time (in hours)
    :param pred_time: prediction time, i.e. the function predicts the number of retweets from obs_time to pred_time
    (in hours)
    :param window: bin width for prediction (\delta_{pred} in Kobayashi and Lambiotte 2016) (in hours)
    :param kernel: kernel function
    :param dt: interval width for numerical integral calculation
    :param p: infectious rate function
    :param p_max: maximum value of p
    :param params: possibility to pass parameters in array format to infectious rate, should follow correct parameter
    ordering, alternative way is to pass all parameters within p_params
    :param p_params: additional named parameters (dict) passed to infectious rate function
    :return: 3-tuple holding estimated integral values, total number of predicted retweets, and total prediction error
    """
    filt_obs = event_times < obs_time

    if params is None:
        params = []

    lambda_t, total = prediction.predict_nb_retweets_vec(event_times=event_times[filt_obs], follower=follower[filt_obs],
                                                         obs_time=obs_time, pred_time=pred_time,
                                                         p=lambda y: p(y, *params, **p_params), kernel=kernel, dt=dt,
                                                         p_max=p_max)

    events_time_pred = event_times[event_times >= obs_time]

    win_int = int(window / dt)
    tp = np.arange(obs_time, pred_time, window)
    err = 0
    for i, t_cur in enumerate(tp):
        t_end = t_cur + window
        if t_end > pred_time:
            break
        count_current = estimate.get_event_count(events_time_pred, t_cur, t_end)
        pred_count = dt * lambda_t[(i * win_int):((i + 1) * win_int)].sum()
        err += abs(count_current - pred_count)

    return lambda_t, total, err


def load_events(filename, time_factor=1, start_factor=1):
    """
    Loads events form given file path.
    :param filename: path to file
    :param time_factor: factor to multiply time with, useful to convert time unit
    :param start_factor: factor to multiply start_time with
    :return: tuple, first element contains tuple of number of events and start time of observation, second elements
    holds event as array of tuple (event_time, number_of_followers)
    """
    res = []

    with open(filename, "r") as in_file:
        first = next(in_file)
        values_first = first.split(" ")
        for line in in_file:
            values = line.split(" ")
            res.append((float(values[0]) * time_factor, int(values[1])))

    return (float(values_first[0]), float(values_first[1]) * start_factor), res


def load_events_vec(filename, time_factor=1, start_factor=1):
    """
    Loads events in shape of a tuple of nd-arrays.

    The returned values can be used as input for all optimized functions.

    :param filename: path to file
    :param time_factor: factor to multiply time with, useful to convert time unit
    :param start_factor: factor to multiply start_time with
    :return: 2-tuple, first element holding tuple of number of events and start time, second holding tuple of event
    times and follower nd-arrays
    """
    (nb_events, start_time), events = load_events(filename, time_factor, start_factor)
    event_times = np.array([e[0] for e in events])
    followers = np.array([e[1] for e in events])
    return (nb_events, start_time), (event_times, followers)


def training_error(param, event_data, obs_time=6, pred_time=168, e_window_size=4, e_window_stride=1,
                   kernel_int=functions.integral_zhao_vec, p=functions.infectious_rate_tweets_vec,
                   kernel=functions.kernel_zhao_vec, dt=0.1, p_window=4):
    """
    Calculates training error for a given parameter set and event_data. First estimates p0 and then calculates
    prediction error for given parameters.

    Params is an array with current values for [r0, phi0, taum].

    :param param: parameters of infectious rate
    :param event_data: events as array of tuples (event_time, follower count)
    :param obs_time: observation time (in hours)
    :param pred_time: prediction time, i.e. the function predicts the number of retweets from obs_time to pred_time
    (in hours)
    :param e_window_size: bin width for estimation (in hours)
    :param e_window_stride: interval for moving windows (in hours)
    :param kernel_int: kernel integral function for estimation
    :param p: infectious rate function
    :param kernel: kernel function for prediction
    :param dt: interval width for numerical integral calculation
    :param p_window: bin width for prediction (in hours)
    :return: prediction error for given parameters
    """
    if param[2] < 0:  # taum parameter should never be negative
        return 10000
    else:
        (_, start_time), events = event_data

        add_params = {
            'r0': param[0],
            'phi0': param[1],
            'taum': 1 / param[2],  # training algorithm trains 1/taum -> convert back here
            't0': start_time}

        params, _, _ = estimate_parameters(events=events, obs_time=obs_time, window_size=e_window_size,
                                           window_stride=e_window_stride, kernel_int=kernel_int, p=p,
                                           values=np.array([0]), **add_params)

        add_params['p0'] = params[0]
        p_max = add_params['p0'] * (1 + add_params['r0'])

        _, _, err = predict(events=events, obs_time=obs_time, pred_time=pred_time, window=p_window, p=p, p_max=p_max,
                            kernel=kernel, dt=dt, **add_params)

        return err


def training_error_optimized(param, event_data, obs_time=6, pred_time=168, e_window_size=4, e_window_stride=1,
                             kernel_int=functions.integral_zhao_vec, p=functions.infectious_rate_tweets_vec,
                             kernel=functions.kernel_zhao_vec, dt=0.1, p_window=4):
    """
    Calculates training error for a given parameter set and event_data. First estimates p0 and then calculates
    prediction error for given parameters.

    Params is an array with current values for [r0, phi0, taum].

    Optimized version using numpy. Passed function should expect numpy arrays as input.

    :param param: parameters of infectious rate
    :param event_data: event data_training as tuple of nd-arrays (event_times, follower)
    :param obs_time: observation time (in hours)
    :param pred_time: prediction time, i.e. the function predicts the number of retweets from obs_time to pred_time
    (in hours)
    :param e_window_size: bin width for estimation (in hours)
    :param e_window_stride: interval for moving windows (in hours)
    :param kernel_int: kernel integral function for estimation
    :param p: infectious rate function
    :param kernel: kernel function for prediction
    :param dt: interval width for numerical integral calculation
    :param p_window: bin width for prediction (in hours)
    :return: prediction error for given parameters
    """
    if param[2] < 0:
        return 10000  # taum parameter should never be negative
    else:
        (_, start_time), (event_times, follower) = event_data

        add_params = {
            'r0': param[0],
            'phi0': param[1],
            'taum': 1 / param[2],  # training algorithm trains 1/taum -> convert back here
            't0': start_time
        }

        params, _, _ = estimate_parameters_optimized(event_times=event_times, follower=follower, obs_time=obs_time,
                                                     window_size=e_window_size, window_stride=e_window_stride,
                                                     kernel_int=kernel_int, p=p, values=np.array([0]), **add_params)

        add_params['p0'] = params[0]
        p_max = add_params['p0'] * (1 + add_params['r0'])

        _, _, err = predict_optimized(event_times=event_times, follower=follower, obs_time=obs_time,
                                      pred_time=pred_time, window=p_window, p=p, p_max=p_max, kernel=kernel, dt=dt,
                                      **add_params)

        return err


def training_cross_validation(events_data, iterations, start_values, simplex=None, obs_time=6, pred_time=168,
                              e_window_size=4, e_window_stride=1, kernel_int=functions.integral_zhao_vec,
                              p=functions.infectious_rate_tweets_vec, kernel=functions.kernel_zhao_vec,
                              dt=0.1, p_window=4):
    """
    Trains parameters of infectious rate with a cross validation approach.

    Should use optimized functions (all passed functions should expect np-arrays as input).

    For training taum use 1/taum!

    :param events_data: list of all event_data tuples, should be in nd-array format (one for each file)
    :param iterations: iteration count for cross validation
    :param start_values: start values
    :param simplex: array-matrix object holding initial simple matrix
    :param obs_time: observation time (in hours)
    :param pred_time: prediction time, i.e. the function predicts the number of retweets from obs_time to pred_time
    (in hours)
    :param e_window_size: bin width for estimation (in hours)
    :param e_window_stride: interval for moving windows (in hours)
    :param kernel_int: kernel integral function for estimation
    :param p: infectious rate function
    :param kernel: kernel function for prediction
    :param dt: interval width for numerical integral calculation
    :param p_window: bin width for prediction (in hours)
    :return: tuple, first element holding 3-tuple of mean training error, median training error and array of trained
    parameters, second element holds tuple of array of training for each iteration and array of prediction errors
    """

    error = []  # training errors for not trained files
    res = []  # training result for each iteration
    param_res = []  # final parameters for each iteration

    for run in range(iterations):
        print("Currently at %i. iteration of CV..." % (run+1))
        # files to keep for current training
        filter_in = np.delete(np.arange(len(events_data)), slice(run, None, iterations))
        cur_events = [events_data[i] for i in filter_in]  # filter out the event_data objects
        train_res = training.train_optimized(start_values, cur_events, obs_time, pred_time, e_window_size,
                                             e_window_stride, kernel_int, p, kernel, dt, p_window, simplex)
        param = train_res['x']

        # calculate training error on filtered files
        cur_events_check = [events_data[i] for i in np.arange(run, len(events_data), iterations)]
        for data in cur_events_check:
            err = training_error_optimized(param, data, obs_time, pred_time, e_window_size, e_window_stride,
                                           kernel_int, p, kernel, dt, p_window)
            error.append(err)

        res.append(train_res)
        param_res.append(param)
        print("Final parameters of %i. CV: %s" % ((run+1), param))

    mean_err = np.mean(error)
    median_err = np.median(error)

    return (mean_err, median_err, param_res), (res, error)
