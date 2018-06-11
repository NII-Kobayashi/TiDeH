"""
Example of predicting future retweet activity using functions implemented in native Python.
First the parameters of the infectious rate are estimated for the given observation time. Then, the number of retweets
before a given time (= pred_time) is predicted.

Inputs are
  1) Data file that includes the retweet times and the number of followers
     Here, this code reads 'data/example/sample_file.txt' (= filename).
  2) Observation time (= obs_time).
  3) Final time of prediction (= pred_time).

Outputs are
  1) Estimate of model parameters of TiDeH (p_0, r_0, phi_0, t_m).
  2) Number of retweets from obs_time (h) to pred_time (h).

You can change the window size for estimation (\delta_obs) by calling the function "estimate_parameters" as following:
params, err, _ = estimate_parameters(events=events, obs_time=obs_time, window_size=window_size, **add_params)
where the variable "window_size" represents \delta_obs.

You can also change the window size for prediction (\delta_pred) by calling the function "predict" as following:
_, total, pred_error = predict(events=events, obs_time=obs_time, pred_time=pred_time, window=window, p_max=None,
                               params=params, **add_params)
where the variable "window" represents \delta_pred.

This code is developed by Sylvain Gauthier and Sebastian Rühl under the supervision of Ryota Kobayashi.
"""
from tideh import estimate_parameters
from tideh import load_events
from tideh import predict

filename = 'data/example/sample_file.txt'
obs_time = 48  # observation time of 2 days
pred_time = 168  # predict for one week

# the number of retweets is not necessary for the further steps
# make sure that all times are loaded in the correct time unit (hours)
(_, start_time), events = load_events(filename)

# additional parameters passed to infectious rate function
add_params = {'t0': start_time, 'bounds': [(-1, 0.5), (1, 20.)]}

params, err, _ = estimate_parameters(events=events, obs_time=obs_time, **add_params)

print("Estimated parameters are:")
print("p0:   %.10f" % params[0])
print("r0:   %.10f" % params[1])
print("phi0: %.10f" % params[2])
print("tm:   %.10f" % params[3])
print("Average %% error (estimated to fitted): %.3f" % (err * 100))

# predict future retweets
_, total, pred_error = predict(events=events, obs_time=obs_time, pred_time=pred_time, p_max=None, params=params,
                               **add_params)

print("Predicted number of retweets from %s to %s hours: %i" % (obs_time, pred_time, total))
print("Predicted number of retweets at hour %s: %i" % (pred_time,
                                                       len([e for e, _ in events if e <= obs_time]) + total))
print("Prediction error (absolute): %f" % pred_error)
