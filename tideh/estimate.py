# Author:   Sylvain Gauthier
# Author:   Sebastian Rühl
#
# For license information, see LICENSE.txt

"""
Implements functions for estimating instantaneous infectious rate. Used for modeling the infectious rate of a tweet.

It is possible to only estimate the infectious rate on a given interval or using a moving time window approach. See
function comments for more details.

Provides implementations using native Python and optimized versions using nd-arrays and vectorization.

References
----------
.. Kobayashi, R. and Lambiotte, R., 2016, March. TiDeH: Time-Dependent Hawkes Process for Predicting Retweet Dynamics.
   In ICWSM (pp. 191-200).
"""

from . import functions


def estimate_infectious_rate_constant(events, t_start, t_end, kernel_integral, count_events=None):
    """
    Returns estimation of infectious rate for given events on defined interval.
    The infectious is expected to be constant on given interval.

    :param events: array of event tuples containing (event_time, follower_cnt)
    :param t_start: time interval start
    :param t_end: time interval end
    :param kernel_integral: integral function of kernel function
    :param count_events: count of observed events in interval (used for time window approach)
    :return: estimated value for infectious rate
    """
    kernel_int = [fol_cnt * kernel_integral(t_start - event_time, t_end - event_time) for event_time, fol_cnt in events]
    if count_events is not None:
        return count_events / sum(kernel_int)
    else:
        return (len(events)) / sum(kernel_int)


def estimate_infectious_rate_constant_vec(event_times, follower, t_start, t_end, kernel_integral, count_events=None):
    """
    Returns estimation of infectious rate for given event time and followers on defined interval.
    Optimized using numpy.

    :param event_times: nd-array of event times
    :param follower: nd-array of follower counts
    :param t_start: time interval start
    :param t_end: time interval end
    :param kernel_integral: integral function of kernel function
    :param count_events: count of observed events in interval (used for time window approach)
    :return: estimated values for infectious rate
    """
    kernel_int = follower * kernel_integral(t_start - event_times, t_end - event_times)
    if count_events is not None:
        return count_events / kernel_int.sum()
    else:
        return event_times.size / kernel_int.sum()


def estimate_infectious_rate(events, kernel_integral=functions.integral_zhao, obs_time=24, window_size=4,
                             window_stride=1):
    """
    Estimates infectious rate using moving time window approach.

    :param events: observed events as array of tuples (event_time, follower_count)
    :param kernel_integral: function for calculating the integral of the kernel
    :param obs_time: observation time
    :param window_size: bin width for estimation (in hours)
    :param window_stride: interval for moving windows (in hours)
    :return: 3-tuple holding array of estimated infectious rate for every moving time window, event counts of every
    window and the time in the middle of every window
    """
    events_tmp = []  # contains all events until some time point
    events_iterator = 0
    events_iterator_counting = 1  # do not count the first event (initial tweet)

    estimations = []
    window_middle = []  # hold the time points of the intervals (time point in the middle of every window)
    window_event_count = []
    count_current = 0  # current count of events in a window
    count_diff = 1  # count of events in current stride window; do not count the first event (initial tweet)

    for start in range(0, obs_time - window_size + window_stride, window_stride):
        end = start + window_size

        # count up current events in window and add them to to events array
        while events_iterator < len(events) and events[events_iterator][0] < end:
            events_tmp.append(events[events_iterator])
            events_iterator += 1
            count_current += 1

        # subtract count of events for stride interval
        count_current = count_current - count_diff

        est = estimate_infectious_rate_constant(events=events_tmp,
                                                t_start=start,
                                                t_end=end,
                                                kernel_integral=kernel_integral,
                                                count_events=count_current)

        # count up events for length of stride window
        count_diff = 0
        while events_iterator_counting < len(events) and events[events_iterator_counting][0] < (start + window_stride):
            events_iterator_counting += 1
            count_diff += 1

        window_middle.append(start + window_size / 2)
        window_event_count.append(count_current)
        estimations.append(est)

    return estimations, window_event_count, window_middle


def estimate_infectious_rate_vec(event_times, follower, kernel_integral=functions.integral_zhao_vec, obs_time=24,
                                 window_size=4, window_stride=1):
    """
    Estimates infectious rate using moving time window approach.
    Optimized using numpy and vectorized approach.

    :param event_times: nd-array of event times
    :param follower: nd-array of follower counts
    :param kernel_integral: function for calculating the integral of the kernel
    :param obs_time: observation time
    :param window_size: bin width for estimation (in hours)
    :param window_stride: interval for moving windows (in hours)
    :return: 3-tuple holding list of estimated infectious rate for every moving time window, event counts for every
    window and the time in the middle of every window
    """
    estimations = []
    window_middle = []
    window_event_count = []

    for start in range(0, obs_time - window_size + window_stride, window_stride):
        end = start + window_size

        mask = event_times < end  # all events up until end of current interval
        count_current = get_event_count(event_times, start, end)

        est = estimate_infectious_rate_constant_vec(event_times[mask],
                                                    follower[mask],
                                                    t_start=start,
                                                    t_end=end,
                                                    kernel_integral=kernel_integral,
                                                    count_events=count_current)

        window_middle.append(start + window_size / 2)
        window_event_count.append(count_current)
        estimations.append(est)

    return estimations, window_event_count, window_middle


def get_event_count(event_times, start, end):
    """
    Count of events in given interval.

    :param event_times: nd-array of event times
    :param start: interval start
    :param end: interval end
    :return: count of events in interval
    """
    mask = (event_times > start) & (event_times <= end)
    return event_times[mask].size
