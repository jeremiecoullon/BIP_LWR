# -*- coding: utf-8 -*-

import numpy as np
from copy import deepcopy
from scipy.interpolate import interp1d

def uniform_log_prob(theta, lower, upper):
    if lower < theta < upper:
        return np.log(1/(upper - lower))
    else:
        return -1000000

# lower and upper bounds for Del Castillo FD parameters
prior_bounds = {'u': (1, 10), 'z': (100, 400), 'rho_j': (300, 800), 'w': (0.004, 10)}

prior_bounds_exp_FD = {'alpha': (1, 50), 'beta': (0.001, 10)}


def cut_prior_means(prior_mean, data_array_str):
    """
    Returns prior mean cut based on dataset

    Parameters
    ----------
    prior_mean: ndarray
        array of prior mean of length 150
    data_array_str: str
        Name of data_array (CSV file) for flow data

    Returns
    -------
    new_prior_mean: ndarray
        Cut prior mean
    """
    data_IC_dict = {'data_array_70108_flow_shorter2.csv': (20, 80), # DS1,60t
            'test_data/Simulated_LWR_Nov2018/data_array_DelCast_flow.csv': (20, 80), # DelCastillo simulated data, DS1,60t
            'test_data/Simulated_LWR_Nov2018/data_array_DelCast_flow_RT_40.csv': (20, 80), # DelCastillo simulated data for high resolution BCs, DS1,60t
            'test_data/Simulated_LWR_Nov2018/data_array_Exp_flow.csv': (20, 80), # Exp simulated data, DS1,60t
            'test_data/Simulated_LWR_Nov2018/data_array_Exp_flow_RT40.csv': (20, 80), # Exp simulated data; high resolution BCs, DS1,60t
            'data_array_70108_flow_52t.csv': (28, 80), # DS1, 52t.
            'data_array_70108_flow_49t.csv': (31, 80), # DS1, 49t.
            'data_array_70108_flow_48t.csv': (32, 80), # DS1, 48t.
            'data_array_70108_flow_47t.csv': (33, 80), # DS1, 47t.
            'data_array_70108_flow_46t.csv': (34, 80), # DS1, 46t.
            'data_array_flow_70108_longer.csv': (0, 150), # DS1, 150t
            'data_array_70108_flow_shorter.csv': (0, 90),
            'test_data/artificial_data_array_flow_prior_BC_poisson_short.csv': (0, 20),

            'Sim_data_array_Exp_flow_40t.csv': (31, 71), # Exp, Simulated cut to 40t
            'Sim_data_array_DelCast_flow_40t.csv': (31, 71), # Exp, Simulated cut to 40t
            }
    new_prior_mean = deepcopy(prior_mean)
    if data_array_str in data_IC_dict.keys():
        lower_b, upper_b = data_IC_dict[data_array_str]
        new_prior_mean = new_prior_mean[lower_b:upper_b]
    else:
        raise ValueError("need to cut prior mean for this dataset")
    return new_prior_mean


def create_interpolate_prior_mean_fun(final_time, prior_mean_raw):
    """
    Create function to interpolate prior mean. Use cubic splines
    """
    f_outlet = interp1d(np.arange(0, final_time+1), prior_mean_raw, kind='cubic')
    return f_outlet
