"""
This code trains the TiDeH parameters (r0, phi0, and taum) based on a retweet dataset (data/training/RT*.txt), assuming
the parameters are same in the dataset.
Please replace file paths according to your local directory structure.

Inputs are
1) Data file that includes the retweet times and the number of followers
Here, this code reads 'data/training/RT*.txt' (= filename).
2) Observation time (= obs_time).
3) Final time of prediction (= pred_time).

Outputs are
1) Model parameters of TiDeH (r_0, phi_0, 1/t_m).
2) Errors evaluated via Cross-Validation.

This code is developed by Sylvain Gauthier and Sebastian Rühl under the supervision of Ryota Kobayashi.
"""
from tideh import load_events_vec
from tideh import training_cross_validation

number_of_files = 100  # number of files to train on
file_name_prefix = 'data/training/RT'  # file names prefix of files used for training
iterations = 5   # number of cross validation iterations
pred_time = 168  # prediction time (hours)

# get file paths of files to use for training
file_names = [file_name_prefix + str(i) + '.txt' for i in range(1, number_of_files + 1)]

# load events for optimized training
events_data = []
for file in file_names:
    events_data.append(load_events_vec(file, 1 / 3600, 24))  # convert event_times and start_time to hours


# initial simplex matrix passed to simplex algorithm
# ordering is r0, phi, taum; taum is trained by 1/taum
simplex = [
    [0.2, 0, 0.25],
    [0, -0.1, 0.25],
    [0.2, 0.1, 0.15],
    [0.4, 0.0, 0.15],
]
start_values = [0.2, 0, 0.25]

(mean_err, median_err, param), _ = training_cross_validation(events_data, iterations, start_values, simplex,
                                                             pred_time=pred_time)

print("Final mean training error: %0.3f" % mean_err)
print("Final median training error: %0.3f" % median_err)
print("Final infectious rate parameters: " + str(param))
