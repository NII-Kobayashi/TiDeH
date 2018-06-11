# Prediction of Twitter retweet dynamic using time dependant Hawkes Process

This is the code illustrating the *Kobayashi & Lambiotte 2016* paper and aims at
providing a framework for experimenting with time dependant Hawkes processes in
the context of twitter retweet prediction. An arXiv.org version is available 
[here](https://arxiv.org/abs/1603.09449).

## Requirements

 - Python 3
 - Scipy >= 0.19
 - Numpy >= 1.10.4

## Getting started

The git repository can be cloned by simply using:

    git clone https://github.com/NII-Kobayashi/TiDeH.git

Once the repository is cloned, the folder should contain two different
subfolders and this README file.

The **data** folder contains some twitter data that can be used for testing.

The **tideh** folder contains all the core python code.

## Running example files

There are three example codes in this directory, i.e. example_native.py, example_optimized.py, example_simulation.py, 
and example_training.py.

- *example_native.py* : This code estimates all the model parameters (p0, r0, phi0, and tau_m) from observation data 
and predicts future retweet activity. As shown in Kobayashi and Lambiotte 2016 (Fig. 7), we recommend to use the code 
when you have enough observation time (more than 12 hours).
- *example_optimized.py* : An optimized version of "example_native.py". This code is more efficient than the native code
"example_native.py". We recommend to use this code for practical use.
- *example_simulation.py*:  This code generates a sample retweet dataset based on Time-Dependent Hawakes (TiDeH) 
process, and estimates all the model parameters.
- *example_training.py*: This code trains the model parameters (r0, phi0 and tau_m) from a retweet dataset (all the 
retweet files in directory 'data/training/RT').

You can just run:

    python3 <name of the example file>
    
for example
    
    python3 example_native.py

## Description of each module

 - *main.py* : this is a front-end module that offers high-level functions to
   estimate the parameters of the model and compute prediction on a tweet
   sequence.
 - *estimate.py* : this is the file that implements the estimation algorithm, to
   estimate the value of the infectious rate function according to a given tweet
   sequence at different points in time.
 - *fit.py* : given estimated values for the infectious rate function, this
   module permit to find the parameters for a given model that fit the
   estimation best. By default, the infectious rate model described in the
   article is used.
 - *prediction.py* : this module implements the prediction algorithm using the
   self-consistent integral equation described in the article, but with
   arbitrary kernel and infectious rate functions. By default, the ones in the
   paper are used.
 - *training.py* : implements the training for best parameters of the infectious
   rate function, with given tweet sequences as input. Will find the parameters
   for the infectious rate function described in the paper that minimize the
   global prediction error for all the input data.
 - *simulate.py* : implements a simulation of a random tweet sequence according
   to the model described in the paper. Very useful to generate artificial
   twitter data.
 - *functions.py* : implements the basic mathematical expression of the
   different functions used, mainly the kernel memory function and the
   infectious rate function as they are described in the paper.

## Note on the modularity of the code

Except for the training, all functions have been written in the goal to keep the
codebase as general as possible. It is therefore possible to pass arbitrary
kernel functions and infectious rate functions to all the algorithms to
experiment on their influence for the final result.

## Data source

The provided samples are extracted from the data set used by Zhao et al., 
KDD '15, in the [SEISMIC](http://snap.stanford.edu/seismic/seismic.pdf) paper. 
You can find more information about the data [here] 
(http://snap.stanford.edu/seismic/#data).

For this work the data (used for training) was slightly aggregated to the 
following format:
- one file per tweet
- space separated
- first row: \<number of total retweets\> \<start time of tweet in days\>
- every other row: \<relative time of tweet/retweet in seconds\> \<number of followers\>
- only tweets with more than 2000 retweets were used

## License

This project is licensed under the terms of the MIT license.

Please contact Ryota Kobayashi if you want to use the code for commercial purposes.
