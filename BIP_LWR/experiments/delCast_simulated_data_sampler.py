#!/usr/bin/env python

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from BIP_LWR.tools.MCMC_simulated_params import section_dict_delCast_simulated_temp1, cov_joint_delcast, section_dict_delCast_simulated_temp05
# from BIP_LWR.tools.MCMC_simulated_params import section_dict_delCast_simulated_temp04
from BIP_LWR.samplers.lwr.delcast_sampler import DelCastSampler
from BIP_LWR.tools.util import del_Cast_random_IC, scale_gibbs_blocks
from BIP_LWR.traffic_data.test_data.Simulated_LWR_Nov2018.del_Cast_true_params import FD_delCast_simulated
from BIP_LWR.traffic_data.test_data.Simulated_LWR_Nov2018.del_Cast_BC_20training_data.delCast_simulated_20training_FDs import FD_delCastsim_list


from BIP_LWR.tools.MCMC_FDBC_params import section_dict_delCast_simulated_temp04, section_dict_delCast_simulated_temp05, section_dict_delCast_simulated_temp063 
from BIP_LWR.tools.MCMC_FDBC_params import section_dict_delCast_simulated_temp08, section_dict_delCast_simulated_temp1, section_dict_delCast_simulated_temp025

from BIP_LWR.tools.MCMC_sim_cut40t_params import section_dict_delCast_simulated_40t_temp042, cov_cut_042_delcast

from BIP_LWR.traffic_data.chain_data.DelCast_Sim_FDBC.FD_DelCast_Sim_FDBC import FD_delcast_sim_FDBC

alpha_temp_cst = 1


# ====================================
# fit to Nov4; run1; local (but multiply it by 1.3)
# 2.5 is for beta_temp=0.5
# cov = np.array([[ 1.87894051,  1.39963161, -0.02410411,  0.00099765],
#        [ 1.39963161, 12.17206861,  0.05803593, -0.00761352],
#        [-0.02410411,  0.05803593,  0.00177518,  0.00002993],
#        [ 0.00099765, -0.00761352,  0.00002993,  0.00005551]])


# for beta=0.4
# cov = 1.8*np.array([[ 1.87894051,  1.39963161, -0.02410411,  0.00099765],
#        [ 1.39963161, 12.17206861,  0.05803593, -0.00761352],
#        [-0.02410411,  0.05803593,  0.00177518,  0.00002993],
#        [ 0.00099765, -0.00761352,  0.00002993,  0.00005551]])

# for beta=0.2
# cov = np.array([[ 3.76981695,  0.4520113 , -0.0727871 ,  0.00823682],
#        [ 0.4520113 , 29.47393605,  0.23799159, -0.03223826],
#        [-0.0727871 ,  0.23799159,  0.00700883,  0.00008671],
#        [ 0.00823682, -0.03223826,  0.00008671,  0.0003117 ]])

# Beta=1 (tuned in Feb 2019)
cov = np.array([[ 0.62772279,  0.36479498, -0.01274923,  0.00033383],
       [ 0.36479498,  4.38784781,  0.02543674, -0.0049363 ],
       [-0.01274923,  0.02543674,  0.00167817,  0.00006093],
       [ 0.00033383, -0.0049363 ,  0.00006093,  0.00003466]])

# ====================================
# cov = np.diag([10000, 20000, 4, 4])

# cov = 2*deepcopy(cov_cut_042_delcast)

cov_joint = (1/alpha_temp_cst)*deepcopy(cov_joint_delcast)
# move_probs = [0.1, 0.3, 0.6] # moves: [FD, joint, BCs]
# move_probs = [0.4, 0, 0.6]
move_probs = [1,0,0]

# le_section_dict = section_dict_delCast_simulated_40t_temp042
le_section_dict = {'section_1': {'param': 'BC_inlet', 'cut1': 0, 'cut2': 1, 'omega': 1},}

upload_to_S3 = True
# comments = "FD and BC for simulated cut (40t) for alpha_temp_cst={}. Retuned cov and BCs".format(alpha_temp_cst)
comments = "FD sampling only"

# Nov1_2018-Delcast_ArtificialData
# Nov4_2018-Higher_res-Delcast_ArtificialData
# Jan3_2019-DelCast_sim-Replica_exchange
# Feb6_2019-DelCast_Sim-FDBC-cut_time
# March3_2019-DelCast_Sim_FDonly
config_dict = {'my_analysis_dir': '2019/March3_2019-DelCast_Sim_FDonly',
                'run_num': 1,
                'data_array_dict':
                        # Del Castillo, 60t.
                        # True FD: {'rho_j': 410, 'u': 3.2, 'w': 0.1, 'z': 180}
                        # High dimensional BCs: with ratio_times_BCs = 40 (so number of BC time points is 2400)
                        {'flow': 'test_data/Simulated_LWR_Nov2018/data_array_DelCast_flow_RT_40.csv',
                        'density': 'test_data/Simulated_LWR_Nov2018/data_array_DelCast_density_RT_40.csv'},
                       #   {'flow': 'Sim_data_array_DelCast_flow_40t.csv',
                       # 'density': 'Sim_data_array_DelCast_density_40t.csv'},

                'upload_to_S3': upload_to_S3,
                'save_chain': True,
                'w_transf_type': 'inv',
                'comments': comments,
                'ratio_times_BCs': 40,
                'step_save': 1,
                      }

# ===========
# start = 11*40
# end = -9*40
# FD = deepcopy(FD_delcast_sim_FDBC)
FD = deepcopy(FD_delCast_simulated)
FD['BC_outlet'] = FD['BC_outlet']#[start:end]
FD['BC_inlet'] = FD['BC_inlet']#[start:end]
# FD = FD_delCastsim_list[0]
# mcmc = DelCastSampler(ICs=FD, cov=cov, cov_joint=cov_joint, section_dict=le_section_dict,
#             move_probs=move_probs, config_dict=config_dict)
# FD_noise_dict = {'z': 20, 'rho_j': 20, 'u': 2, 'w':0.3}
# # FD_noise_dict = {'z': 1, 'rho_j': 1, 'u': 0.1, 'w':0.1}
# FD = del_Cast_random_IC(FD_mean=FD, mcmc_obj=mcmc, omega=0.5, FD_prior=False, FD_noise_dict=FD_noise_dict)
# del mcmc
# ================

FD['rho_j'] = np.random.normal(FD['rho_j'], scale=1e-6)

# FD['z'] = np.random.normal(FD['z'], scale=1)
mcmc = DelCastSampler(ICs=FD, cov=cov, cov_joint=cov_joint, section_dict=le_section_dict,
            move_probs=move_probs, config_dict=config_dict)
mcmc.alpha_temp = alpha_temp_cst
