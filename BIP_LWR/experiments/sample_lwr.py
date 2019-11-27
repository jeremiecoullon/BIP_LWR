#!/usr/bin/env python

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
w_transf_type = 'inv'

alpha_temp_cst = 1

# FD|BC for beta=1
# cov = np.array([[ 0.66878389,  0.89393652, -0.00067186,  0.00210403],
#        [ 0.89393652,  3.33640027,  0.00612946, -0.00136162],
#        [-0.00067186,  0.00612946,  0.00484103,  0.00047524],
       # [ 0.00210403, -0.00136162,  0.00047524,  0.00010494]])

# FD|BC for beta=0.76
# cov = np.array([[ 0.78746895,  1.27333014,  0.00141434,  0.00308789],
#        [ 1.27333014,  5.9597071 ,  0.01655901, -0.00077557],
#        [ 0.00141434,  0.01655901,  0.00573808,  0.00047403],
#        [ 0.00308789, -0.00077557,  0.00047403,  0.00011784]])

# FD|BC for beta=0.58
# cov = np.array([[ 0.91032747,  0.97317003,  0.00399942,  0.00339584],
#        [ 0.97317003,  6.24639295,  0.02674442, -0.00593384],
#        [ 0.00399942,  0.02674442,  0.0056371 ,  0.00051024],
#        [ 0.00339584, -0.00593384,  0.00051024,  0.00012784]])

# for beta=0.44
# cov = np.array([[ 1.69715527,  2.72433505, -0.03295926,  0.00081907],
#        [ 2.72433505, 10.64697739, -0.06830345, -0.01433845],
#        [-0.03295926, -0.06830345,  0.01229563,  0.00125811],
#        [ 0.00081907, -0.01433845,  0.00125811,  0.00026881]])
# for beta=0.15 (re-use for beta=0.2)
# cov = np.array([[ 4.20024635,  4.24006613, -0.02955102,  0.00685512],
#        [ 4.24006613, 23.06947014, -0.02415083, -0.0439088 ],
#        [-0.02955102, -0.02415083,  0.03938422,  0.00430554],
#        [ 0.00685512, -0.0439088 ,  0.00430554,  0.00093482]])
# ===============
# beta=1
# cov_joint = 1.3*cov_inv_joint*(1/alpha_temp_cst)
# # beta=0.58
# cov_joint = 3*cov_inv_joint*(1/alpha_temp_cst)
# beta=0.44
cov_joint = cov_inv_joint_DelCast_DS1*(1/alpha_temp_cst)


# For FD only sampling (March4)
cov = np.array([[ 1.46565905, -0.43494473, -0.04680924,  0.00179531],
 [-0.43494473,  5.61913019,  0.01936478, -0.0128512],
 [-0.04680924,  0.01936478,  0.00575797,  0.00036424],
 [ 0.00179531, -0.0128512,   0.00036424,  0.00008159]])

# move_probs = [0.1, 0.3, 0.6] # moves: [FD, joint, BCs]
move_probs = [1, 0, 0]
# le_section_dict = section_dict_70108_49t_delCast_temp044
le_section_dict = {'section_1': {'param': 'BC_inlet', 'cut1': 0, 'cut2': 1, 'omega': 1},}
upload_to_S3 = True
# comments = "FD and BC for alpha_temp_cst={}. With joint move and shift".format(alpha_temp_cst)
# comments = "tune FD for alpha_temp_cst={}".format(alpha_temp_cst)
comments = "FD only sampling"

# Jan6_2019-delCast-DS1_PT
# Feb4_2019-DelCast_DS1-FDBC
# March4_2019-DelCast_DS1-FDonly
config_dict = {'my_analysis_dir': '2019/March4_2019-DelCast_DS1-FDonly',
                'run_num': 1,
                'data_array_dict':
                        # DATASET 1, 49 times, all spaces
                        {'flow': 'data_array_70108_flow_49t.csv',
                        'density': 'data_array_70108_density_49t.csv'},

                'upload_to_S3': upload_to_S3,
                'save_chain': True,
                'w_transf_type': w_transf_type,
                'comments': comments,
                'ratio_times_BCs': 40,
                'step_save': 1,
                # 'FD_only': True,
                      }

# to get raw high dimensional BCs
lwr = LWR_Solver(config_dict=config_dict)



# Mean of FD | BC_raw (from Dec2; run_6; local)
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


mcmc = DelCastSampler(ICs=FD, cov=cov, cov_joint=cov_joint, section_dict=le_section_dict,
            move_probs=move_probs, config_dict=config_dict)
mcmc.alpha_temp = alpha_temp_cst
