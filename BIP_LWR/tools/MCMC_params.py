# -*- coding: utf-8 -*-

import numpy as np
from BIP_LWR.tools.util import scale_cov_directions


# manually increase u variance to mix in tempered likelihood
unscaled_cov_inv = np.array([[ 0.53855822, -0.18681024, -0.00724823,  0.00129741],
                   [-0.18681024,  3.60189924,  0.03020478, -0.00365225],
                   [-0.00724823,  0.03020478,  0.004, -0.00002783],
                   [ 0.00129741, -0.00365225, -0.00002783,  0.00003305]])
# Cov fitted to joint FD and BC sampling (in April3_2018-tune_real_data-Run_5: on real data ('data_array_70108_density_longer.csv'))
unscaled_cov_inv_joint = np.array([[ 3.81896926,  8.89477701, -0.05738674,  0.01151146],
           [ 8.89477701, 30.50346597, -0.07976908,  0.02867397],
           [-0.05738674, -0.07976908,  0.00422877,  0.00016535],
           [ 0.01151146,  0.02867397,  0.00016535,  0.00012827]])
cov_inv = scale_cov_directions(cov=unscaled_cov_inv, alpha_list=[1, 1, 1, 1])
cov_inv_joint = scale_cov_directions(cov=unscaled_cov_inv_joint, alpha_list=[1, 0.01, 0.01, 0.01])

# ==========================================================================================
# Fit to Feb4; Run_10, process 3 (beta=1): Del Castillo on DS1 (PT with 4 temperatures)
global_cov_delCast_DS1 = np.array([[18.81002999, 35.91820071, -1.55164297,  0.02254338],
       [35.91820071, 83.3302258 , -2.96591058,  0.03870085],
       [-1.55164297, -2.96591058,  0.28228428,  0.01163059],
       [ 0.02254338,  0.03870085,  0.01163059,  0.00143503]])
cov_inv_joint_DelCast_DS1 = scale_cov_directions(cov=global_cov_delCast_DS1, alpha_list=[1, 0.01, 0.01, 0.01])
# ==========================================================================================

# DS1, 49t for Del Castillo
section_dict_70108_49t = {
    'section_1': {'param': 'BC_inlet', 'cut1': 0, 'cut2': 49, 'omega': 0.07},
    'section_2': {'param': 'BC_inlet', 'cut1': 0, 'cut2': 15, 'omega':0.15},
    'section_3': {'param': 'BC_inlet', 'cut1': 7, 'cut2': 35, 'omega':0.13},
    'section_4': {'param': 'BC_inlet', 'cut1': 33, 'cut2': 49, 'omega':0.4},

    'section_5': {'param': 'BC_outlet', 'cut1':0, 'cut2': 49, 'omega':0.04},
    'section_6': {'param': 'BC_outlet', 'cut1': 0, 'cut2': 15, 'omega':0.25},
    'section_7': {'param': 'BC_outlet', 'cut1': 12, 'cut2': 24, 'omega':0.21},
    'section_8': {'param': 'BC_outlet', 'cut1': 25, 'cut2': 39, 'omega':0.16},
    'section_9': {'param': 'BC_outlet', 'cut1':29, 'cut2': 49, 'omega':0.12},
    'section_10': {'param': 'BC_outlet', 'cut1': 6, 'cut2': 18, 'omega':0.4},
    'section_11': {'param': 'BC_outlet', 'cut1': 39, 'cut2': 49, 'omega':0.4},
    'section_12': {'param': 'BC_inlet', 'cut1': 0, 'cut2': 12, 'omega':0.27},

    'section_13': {'param': 'BC_inlet', 'cut1': 20, 'cut2': 36, 'omega':0.25},
    # outlet bimodality
    'section_14': {'param': 'BC_outlet', 'cut1': 2, 'cut2': 12, 'omega': 0.9},   
}

section_dict_70108_49t_delCast_temp076 = {
    'section_1': {'param': 'BC_inlet', 'cut1': 0, 'cut2': 49, 'omega': 0.1},
    'section_2': {'param': 'BC_inlet', 'cut1': 0, 'cut2': 15, 'omega':0.2},
    'section_3': {'param': 'BC_inlet', 'cut1': 7, 'cut2': 34, 'omega':0.18},
    'section_4': {'param': 'BC_inlet', 'cut1': 34, 'cut2': 49, 'omega':0.44},

    'section_5': {'param': 'BC_outlet', 'cut1':0, 'cut2': 49, 'omega':0.055},
    'section_6': {'param': 'BC_outlet', 'cut1': 0, 'cut2': 15, 'omega':0.3},
    'section_7': {'param': 'BC_outlet', 'cut1': 12, 'cut2': 24, 'omega':0.21},
    'section_8': {'param': 'BC_outlet', 'cut1': 25, 'cut2': 39, 'omega':0.2},
    'section_9': {'param': 'BC_outlet', 'cut1':29, 'cut2': 49, 'omega':0.14},
    'section_10': {'param': 'BC_outlet', 'cut1': 6, 'cut2': 18, 'omega':0.45},
    'section_11': {'param': 'BC_outlet', 'cut1': 38, 'cut2': 49, 'omega':0.48},
    'section_12': {'param': 'BC_inlet', 'cut1': 0, 'cut2': 12, 'omega':0.3},
    'section_13': {'param': 'BC_inlet', 'cut1': 20, 'cut2': 35, 'omega':0.31},
    # for bimodality
    'section_14': {'param': 'BC_inlet', 'cut1': 5, 'cut2': 15, 'omega': 0.9},
    # outlet bimodality
    'section_15': {'param': 'BC_outlet', 'cut1': 2, 'cut2': 12, 'omega': 1},   
}
section_dict_DS1_test_block = {
 'section_100': {'param': 'BC_outlet', 'cut1': 2, 'cut2': 12, 'omega': 0.9},    
}

section_dict_70108_49t_delCast_temp058 = {
    'section_1': {'param': 'BC_inlet', 'cut1': 0, 'cut2': 49, 'omega': 0.11},
    'section_2': {'param': 'BC_inlet', 'cut1': 0, 'cut2': 15, 'omega':0.21},
    'section_3': {'param': 'BC_inlet', 'cut1': 7, 'cut2': 34, 'omega':0.2},
    'section_4': {'param': 'BC_inlet', 'cut1': 34, 'cut2': 49, 'omega':0.47},

    'section_5': {'param': 'BC_outlet', 'cut1':0, 'cut2': 49, 'omega':0.06},
    'section_6': {'param': 'BC_outlet', 'cut1': 0, 'cut2': 15, 'omega':0.39},
    'section_7': {'param': 'BC_outlet', 'cut1': 12, 'cut2': 24, 'omega':0.26},
    'section_8': {'param': 'BC_outlet', 'cut1': 25, 'cut2': 39, 'omega':0.2},
    'section_9': {'param': 'BC_outlet', 'cut1': 29, 'cut2': 49, 'omega':0.14},
    'section_10': {'param': 'BC_outlet', 'cut1': 6, 'cut2': 18, 'omega':0.5},
    'section_11': {'param': 'BC_outlet', 'cut1': 38, 'cut2': 49, 'omega':0.51},
    'section_12': {'param': 'BC_inlet', 'cut1': 0, 'cut2': 12, 'omega': 0.4},
    'section_13': {'param': 'BC_inlet', 'cut1': 20, 'cut2': 35, 'omega': 0.36},
    # for bimodality
    'section_14': {'param': 'BC_inlet', 'cut1': 5, 'cut2': 15, 'omega': 0.9},
    # outlet bimodality
    'section_15': {'param': 'BC_outlet', 'cut1': 3, 'cut2': 12, 'omega': 1},   
}




# ====================
# Exp FD
# ====================

# DS1, 49t for Exp FD (for BC|FD)
section_dict_70108_49t_exp = {
    'section_1': {'param': 'BC_inlet', 'cut1': 0, 'cut2': 49, 'omega': 0.125},
    'section_2': {'param': 'BC_inlet', 'cut1': 0, 'cut2': 15, 'omega':0.48},
    'section_3': {'param': 'BC_inlet', 'cut1': 7, 'cut2': 34, 'omega':0.28},
    'section_4': {'param': 'BC_inlet', 'cut1': 30, 'cut2': 40, 'omega':0.6},
    'section_5': {'param': 'BC_inlet', 'cut1': 40, 'cut2': 49, 'omega':0.4},

    'section_6': {'param': 'BC_outlet', 'cut1':0, 'cut2': 49, 'omega':0.06},
    'section_7': {'param': 'BC_outlet', 'cut1': 0, 'cut2': 20, 'omega':0.44},
    'section_8': {'param': 'BC_outlet', 'cut1': 12, 'cut2': 24, 'omega':0.38},
    'section_9': {'param': 'BC_outlet', 'cut1': 29, 'cut2': 41, 'omega':0.1},
    'section_10': {'param': 'BC_outlet', 'cut1': 25, 'cut2': 35, 'omega':0.11},
    'section_11': {'param': 'BC_outlet', 'cut1': 33, 'cut2': 39, 'omega':0.15},
    'section_12': {'param': 'BC_outlet', 'cut1': 22, 'cut2': 30, 'omega':0.19},
    # for inlet bimodality
    'section_13': {'param': 'BC_inlet', 'cut1': 0, 'cut2': 10, 'omega':0.8},
    # in case you can jump
    'section_14': {'param': 'BC_outlet', 'cut1': 18, 'cut2': 31, 'omega':0.9},
    'section_15': {'param': 'BC_outlet', 'cut1': 37, 'cut2': 49, 'omega':0.7},
}
section_dict_70108_49t_exp_temp076 = {
    'section_1': {'param': 'BC_inlet', 'cut1': 0, 'cut2': 49, 'omega': 0.14},
    'section_2': {'param': 'BC_inlet', 'cut1': 0, 'cut2': 15, 'omega':0.5},
    'section_3': {'param': 'BC_inlet', 'cut1': 7, 'cut2': 34, 'omega':0.32},
    'section_4': {'param': 'BC_inlet', 'cut1': 30, 'cut2': 44, 'omega':0.64},
    'section_5': {'param': 'BC_inlet', 'cut1': 40, 'cut2': 49, 'omega':0.44},

    'section_6': {'param': 'BC_outlet', 'cut1':0, 'cut2': 49, 'omega':0.067},
    'section_7': {'param': 'BC_outlet', 'cut1': 0, 'cut2': 20, 'omega':0.5},
    'section_8': {'param': 'BC_outlet', 'cut1': 12, 'cut2': 24, 'omega':0.45},
    'section_9': {'param': 'BC_outlet', 'cut1': 28, 'cut2': 39, 'omega':0.14},
    'section_10': {'param': 'BC_outlet', 'cut1': 25, 'cut2': 35, 'omega':0.14},
    'section_11': {'param': 'BC_outlet', 'cut1': 33, 'cut2': 39, 'omega':0.2},
    'section_12': {'param': 'BC_outlet', 'cut1': 20, 'cut2': 30, 'omega':0.26},
    # for inlet bimodality
    'section_13': {'param': 'BC_inlet', 'cut1': 0, 'cut2': 10, 'omega':0.8},
    # for other inlet bimodality
    'section_14': {'param': 'BC_inlet', 'cut1': 42, 'cut2': 49, 'omega': 0.7},
    # in case you can jump
    'section_15': {'param': 'BC_outlet', 'cut1': 18, 'cut2': 31, 'omega':0.9},
    'section_16': {'param': 'BC_outlet', 'cut1': 37, 'cut2': 49, 'omega':0.79},
}

section_dict_70108_49t_exp_temp058 = {
    'section_1': {'param': 'BC_inlet', 'cut1': 0, 'cut2': 49, 'omega': 0.18},
    'section_2': {'param': 'BC_inlet', 'cut1': 0, 'cut2': 15, 'omega':0.53},
    'section_3': {'param': 'BC_inlet', 'cut1': 7, 'cut2': 34, 'omega':0.39},
    'section_4': {'param': 'BC_inlet', 'cut1': 30, 'cut2': 44, 'omega':0.73},
    'section_5': {'param': 'BC_inlet', 'cut1': 40, 'cut2': 49, 'omega':0.5},

    'section_6': {'param': 'BC_outlet', 'cut1':0, 'cut2': 49, 'omega':0.073},
    'section_7': {'param': 'BC_outlet', 'cut1': 0, 'cut2': 20, 'omega':0.59},
    'section_8': {'param': 'BC_outlet', 'cut1': 8, 'cut2': 25, 'omega':0.45},
    'section_9': {'param': 'BC_outlet', 'cut1': 23, 'cut2': 39, 'omega':0.09},
    'section_10': {'param': 'BC_outlet', 'cut1': 25, 'cut2': 35, 'omega':0.18},
    'section_11': {'param': 'BC_outlet', 'cut1': 33, 'cut2': 39, 'omega':0.26},
    'section_12': {'param': 'BC_outlet', 'cut1': 20, 'cut2': 30, 'omega':0.28},
    # for inlet bimodality
    'section_13': {'param': 'BC_inlet', 'cut1': 0, 'cut2': 10, 'omega':0.8},
    # for other inlet bimodality
    'section_14': {'param': 'BC_inlet', 'cut1': 42, 'cut2': 49, 'omega': 0.7},
    # in case you can jump
    'section_15': {'param': 'BC_outlet', 'cut1': 18, 'cut2': 31, 'omega':0.9},
    'section_16': {'param': 'BC_outlet', 'cut1': 37, 'cut2': 49, 'omega':0.85},

}