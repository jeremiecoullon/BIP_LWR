#!/usr/bin/env python
"""
Runs MCMC for FD parameters given fixed BCs. BCs are from interpolated density from data.

Saves the output to a hdf5 file in 'Analysis' folder which will be created in the
parent directory of the root directory.

To save results to S3:
- set `upload_to_S3=True`
- in `tools/util.py`, set the default bucket name of `upload_chain()` function to your bucket name
- add your AWS config parameters as environment variables (boto will read them)

"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import os
from copy import deepcopy
from collections import OrderedDict
from BIP_LWR.lwr.lwr_solver import LWR_Solver
from BIP_LWR.samplers.lwr.delcast_sampler import DelCastSampler
from BIP_LWR.tools.util import scale_gibbs_blocks, del_Cast_random_IC, apply_FD_noise
from BIP_LWR.tools.MCMC_params import cov_inv, cov_inv_joint_DelCast_DS1, section_dict_DS1_test_block, section_dict_70108_49t_delCast_temp058, section_dict_70108_49t
from BIP_LWR.tools.MCMC_FDBC_params import section_dict_70108_49t_delCast_temp044, section_dict_70108_49t_delCast_temp02

from BIP_LWR.traffic_data.chain_data.DelCast_DS1_FDBC.FD_DelCast_DS1_FDBC import FD_delcast_ds1_FDBC
# --------------
# --------------

# temperature to temper the likelihood with: set to 1 to get target posterior
alpha_temp_cst = 1

# For FD only sampling (March4)
cov = np.array([[ 1.46565905, -0.43494473, -0.04680924,  0.00179531],
 [-0.43494473,  5.61913019,  0.01936478, -0.0128512],
 [-0.04680924,  0.01936478,  0.00575797,  0.00036424],
 [ 0.00179531, -0.0128512,   0.00036424,  0.00008159]])

move_probs = [1, 0, 0] # move probabilities: FD, joint move, BC
le_section_dict = {'section_1': {'param': 'BC_inlet', 'cut1': 0, 'cut2': 1, 'omega': 1},}
upload_to_S3 = False # whether or not to upload the results to S3 rather than locally
comments = "FD only sampling"

config_dict = {'my_analysis_dir': 'FD_only_sampling',
                'run_num': 1,
                'data_array_dict':
                        # Data: 49 times, all spaces
                        {'flow': 'data_array_70108_flow_49t.csv',
                        'density': 'data_array_70108_density_49t.csv'},

                'upload_to_S3': upload_to_S3,
                'save_chain': True,
                'w_transf_type': 'inv',
                'comments': comments,
                # resolution for BCs
                'ratio_times_BCs': 40,
                # on which steps to save MCMC output
                'step_save': 1,
                      }

# to get raw high dimensional BCs
lwr = LWR_Solver(config_dict=config_dict)


FD_mean = {'rho_j': 451.61634886887066, 'u': 2.8720913517939604, 'w': 0.13347762896487803, 'z': 179.43205285224118}

FD = OrderedDict()
for k in ['z','rho_j','u','w']:
    FD[k] = FD_mean[k]
FD['BC_outlet'] = lwr.high_res_BCs("BC_outlet")
FD['BC_inlet'] = lwr.high_res_BCs("BC_inlet")

# FD = deepcopy(FD_delcast_ds1_FDBC)
# mcmc = DelCastSampler(ICs=FD, cov=cov, cov_joint=cov_joint, section_dict=le_section_dict,
#             move_probs=move_probs, config_dict=config_dict)
# FD_noise_dict = {'z': 20, 'rho_j': 20, 'u': 2, 'w':0.2}
# FD = del_Cast_random_IC(FD_mean=FD, mcmc_obj=mcmc, omega=0.7, FD_prior=False, FD_noise_dict=FD_noise_dict)
# del mcmc
# ================
# FD = deepcopy(FD_weird_mode)

FD['rho_j'] = np.random.normal(FD['rho_j'], scale=1e-6)


mcmc = DelCastSampler(ICs=FD, cov=cov, section_dict=le_section_dict,
            move_probs=move_probs, config_dict=config_dict)
mcmc.alpha_temp = alpha_temp_cst

mcmc.run(n_iter=5, print_rate=1)
