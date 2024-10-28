"""
Functions utilize for the reconstruction procedure should be placed here.

Author: Matthew DeVerna
"""
import random

import powerlaw

import numpy as np
import pandas as pd
import scipy.integrate as integrate


def power_law(z, alpha, xmin=1):
    """
    Power law function.
        Form: (alpha-1)/xmin)*((xmin/x)**alpha)
    Parameters:
    ----------
    - z (float): value at which P(z | alpha, xmin) is required. In this script, it will
        represent the time difference in seconds between two tweets/retweets.
    - alpha (float): power law exponent.
        Ref: based on https://en.wikipedia.org/wiki/Power_law
    - xmin (float): minimum value from where power law behavior exists. Default = 1 (second).
        Ref:  https://en.wikipedia.org/wiki/Power_law

    Returns:
    ----------
    (odeint, ode) (tuple) where:
        - odeint: General integration of ordinary differential equations. We use this.
        - ode: Integrate ODE using VODE and ZVODE routines.
        Reference: https://docs.scipy.org/doc/scipy/reference/tutorial/integrate.html
    """

    epsilon = 0.0001
    return integrate.quad(
        lambda x: ((alpha - 1) / xmin) * ((xmin / x) ** alpha), z - epsilon, z + epsilon
    )


def get_who_rtd_whom(
    poten_edge_users,
    poten_edge_tstamps,
    poten_edge_fcounts,
    curr_tstamp,
    gamma,
    alpha,
    xmin,
):
    """
    Probabilistically infer which user in `poten_edge_users` was retweeted.
    For this doc string, the tweet we are inferring this retweet for will be
    called "tweet X".
    Probability of each user is weighted based on how many followers a user
    has, as well as how much time has passed between when the users in
    `poten_edge_users` were involved in the cascade and when tweet X was sent.

    Parameters:
    -----------
    - poten_edge_users (list) : temporally ordered list of user ID strings
        corresponding to when these users interacted with this cascade
    - poten_edge_tstamps (list) : temporally ordered list of timestamps
        corresponding to when the users in `poten_edge_users` interacted
        with this cascade
    - poten_edge_fcounts (array) : temporally ordered array of mean follower
        counts corresponding to the users in `poten_edge_users`
    - curr_tstamp (int) : the timestamp of tweet X
    - gamma (float) : weight given to the follower-count probability distribition.
        Must be between 0 and 1. 1-gamma is given to the time-difference probability
        distribution.
    - alpha (float) : alpha value for power-law function
    - xmin (float) : xmin value for power-law function

    Returns:
    ----------
    - retweeted_uid (str) : the user that tweet X retweeted
    """

    # Create an array of probabilities (temporally ordered) for the
    # probability of the user at position `idx` in the cascade, of
    # retweeting everyone prior to position `idx`. Probabilities
    # are proportional to the number of followers a user has relative
    # all others considered
    total_fcounts = sum(poten_edge_fcounts)
    fcount_probabilities = poten_edge_fcounts / total_fcounts

    # Now we create an array of probabilities (temporally ordered) based
    # on the number of seconds that have passed since the retweet at
    # position `idx` and all tweets prior to position `idx`. Probability
    # is drawn from a power law function
    time_diffs = curr_tstamp - poten_edge_tstamps
    tdiff_probs = np.array(
        [power_law(tdiff + 0.1, alpha, xmin)[0] for tdiff in time_diffs]
    )
    norm_tstamp_prob_array = tdiff_probs / tdiff_probs.sum()

    # Weighted probabilities based on gamma
    weighted_fcount = gamma * fcount_probabilities
    weighted_tstamp = (1 - gamma) * norm_tstamp_prob_array

    # Combine the weighted arrays to get the final individual probabilities
    indiv_probs = weighted_fcount + weighted_tstamp

    # Sample a user based on the integration of these two probabilities
    retweeted_uid = np.random.choice(
        poten_edge_users,
        p=indiv_probs,
    )

    return retweeted_uid


def simulate_plaw_fits(arr, sample_size, num_sims, xmin):
    """
    Fit sampled data from an array to a power-law function many times
    and estimate the parameter alpha each time.

    Parameters:
    -----------
    - arr (array/sequence) : an array of data to be fit to the power-law function
    - sample_size (int) : the size of the sample to draw from arr
    - num_sims (int) : the number of simulations to run
    - xmin (float) : the minimum xvalue for the the scaling relationship

    Return:
    ----------
    est_params (pandas.DataFrame) : a dataframe containing the following columns:
        - run (int) : the run number
        - xmin (float) : the estimated xmin value
        - alpha (float) : the estimated alpha value
    """

    estimated_parameters = []

    for run in range(1, num_sims + 1):
        sample_seconds = random.choices(arr, k=sample_size)

        fit = powerlaw.Fit(sample_seconds, xmin=xmin)

        estimated_parameters.append((run, fit.power_law.xmin, fit.power_law.alpha))

    est_params_df = pd.DataFrame(estimated_parameters, columns=["run", "xmin", "alpha"])
    return est_params_df
