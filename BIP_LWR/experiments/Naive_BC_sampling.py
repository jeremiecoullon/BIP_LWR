#!/usr/bin/env python

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
from copy import deepcopy
from BIP_LWR.tools.MCMC_simulated_params import section_dict_Exp_simulated_temp01, section_dict_Exp_simulated_temp1
from BIP_LWR.tools.MCMC_simulated_params import section_dict_Exp_simulated_temp05, section_dict_Exp_simulated_temp063, section_dict_Exp_simulated_temp08
from BIP_LWR.tools.util import scale_gibbs_blocks
from collections import OrderedDict
from BIP_LWR.traffic_data.test_data.Simulated_LWR_Nov2018.Exp_true_params import FD_Exp_simulated
from BIP_LWR.samplers.lwr.expsampler import ExpSampler

from BIP_LWR.moves.lwr_moves.Naive_BC_move import NaiveBCMove

alpha_temp_cst = 1

# move_probs = [0, 0, 0.8, 0.2] # moves: [FD, joint, BCs, beta_temp_idx]
move_probs = [0, 0, 1]
le_section_dict = section_dict_Exp_simulated_temp1


cov = 2*np.array([[0.00025767, 0.00000059],
       [0.00000059, 2.18e-9]])

cov_joint = deepcopy(cov)*8

comments = "Tune outlet bimodality block"
upload_to_S3 = False

config_dict = {'my_analysis_dir': '2019/Jan7_2019-Thesis-BC_Exp-naive_try/',
                'run_num': 2,
                'data_array_dict':
                        {'flow': 'test_data/Simulated_LWR_Nov2018/data_array_Exp_flow_RT40.csv',
                        'density': 'test_data/Simulated_LWR_Nov2018/data_array_Exp_density_RT40.csv'},

                        'upload_to_S3': upload_to_S3,
                        'save_chain': True,
                        'comments': comments,
                        'ratio_times_BCs': 40,
                        'step_save': 5,
                        # 'FD_only': True,
                      }

ICs = deepcopy(FD_Exp_simulated)

ICs['alpha'] = np.random.normal(ICs['alpha'], scale=1e-6)


mcmc = ExpSampler(ICs=ICs, cov=cov, cov_joint=cov_joint, section_dict=le_section_dict,
                move_probs=move_probs, config_dict=config_dict)
                # beta_temp_list=beta_temp_list, beta_temp_margs=beta_temp_margs, dict_move_covs_tempering=dict_move_covs_tempering)
mcmc.alpha_temp = alpha_temp_cst
mcmc.move = NaiveBCMove(omega=0.02, config_dict=config_dict)