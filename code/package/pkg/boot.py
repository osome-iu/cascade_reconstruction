"""
Bootstrapping functions.

Author: Matthew DeVerna
"""

import numpy as np


def bootstrap_ci(array, confidence=0.95, n_samples=1_000, d_only=False):
    """
    Calculate a confidence interval for sample data via bootstrapping

    Parameters:
    -----------
    - array (of ints/floats): the sample data
    - confidence (float): the desired level of confidence (default=0.95)
    - n_samples (float): number of bootstrap samples
    - distance_only (bool): If True, return only half the distance between the
        upper and lower bounds

    Returns:
    -----------
    - if d_only = False (tuple): The lower and upper bounds of the confidence interval.
    - if d_only = True (float, Default): Distance from low to high divided by 2
    """
    # Ignore NaN values
    n = len(array)
    bootstrapped_means = []
    for _ in range(n_samples):
        bootstrap_sample = np.random.choice(a=array, size=n, replace=True)
        bootstrapped_means.append(np.mean(bootstrap_sample))
    alpha = (1 - confidence) / 2
    lower_bound = np.percentile(bootstrapped_means, alpha * 100)
    upper_bound = np.percentile(bootstrapped_means, (1 - alpha) * 100)
    if d_only:
        h = (upper_bound - lower_bound) / 2
        return h
    return (lower_bound, upper_bound)
