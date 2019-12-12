#!/usr/bin/env python

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from BIP_LWR.lwr.lwr_solver import LWR_Solver
from BIP_LWR.samplers.replica_exchange.population_pt_sampler import PopulationPTSampler
from BIP_LWR.tools.MCMC_FDBC_params import section_dict_70108_49t_delCast_temp044, section_dict_70108_49t, section_dict_70108_49t_delCast_temp076, section_dict_70108_49t_delCast_temp058
from BIP_LWR.tools.MCMC_params import cov_inv_joint_DelCast_DS1
from BIP_LWR.samplers.lwr.delcast_sampler import DelCastSampler
from BIP_LWR.tools.util import BC_noise, apply_FD_noise
from BIP_LWR.traffic_data.chain_data.DelCast_DS1_FDBC.FD_DelCast_DS1_FDBC import FD_delcast_ds1_FDBC


def build_sampler_fun(alpha_temp_cst, section_dict, cov, cov_joint):
	"""
	Higher-order function: retuns a function that builds the sampler
	"""
	def define_delCastsampler_temp():
		move_probs = [0, 0, 1]
		le_section_dict = section_dict
		comments = "BConly sampler. 3 temperatures: [0.58, 0.76, 1]. CorrPT with width=5"
		upload_to_S3 = False
		config_dict = {'my_analysis_dir': 'BConly_PopulationPT_sampler',
					'run_num': 1,
					'data_array_dict':
					{'flow': 'data_array_70108_flow_49t.csv',
                    'density': 'data_array_70108_density_49t.csv'},
					'upload_to_S3': upload_to_S3,
					'save_chain': True,
					'comments': comments,
					'ratio_times_BCs': 40,
					# save MCMC sample every `save_step` number of steps
					'step_save': 31,}
		lwr = LWR_Solver(config_dict=config_dict)

		FD_initial = deepcopy(FD_delcast_ds1_FDBC)

		# Mean of FD | BC_raw (from Dec2; run_6; local)
		FD_mean = {'rho_j': 451.61634886887066, 'u': 2.8720913517939604, 'w': 0.13347762896487803, 'z': 179.43205285224118}
		FD = OrderedDict()
		for k in ['z','rho_j','u','w']:
		    FD[k] = FD_mean[k]
		FD['BC_outlet'] = FD_initial['BC_outlet']
		FD['BC_inlet'] = FD_initial['BC_inlet']

		# ===== BC noise =====
		# omega = 0.6
		# mcmc = DelCastSampler(ICs=FD, cov=cov, cov_joint=cov_joint, section_dict=le_section_dict,
		# 	move_probs=move_probs, config_dict=config_dict)
		# FD['BC_outlet'] = BC_noise(BC=FD['BC_outlet'], BC_mean=mcmc.move.BC_prior_mean['BC_outlet'],
		#             omega=omega, chol_precision=mcmc.move.BC_move.OU.cholPrecision)
		# FD['BC_inlet'] = BC_noise(BC=FD['BC_inlet'], BC_mean=mcmc.move.BC_prior_mean['BC_inlet'],
		#             omega=omega, chol_precision=mcmc.move.BC_move.OU.cholPrecision)
		# del mcmc
		# FD_noise_dict = {'z': 20, 'rho_j': 20, 'u': 2, 'w':0.3}
		# for FD_par, FD_sigma in FD_noise_dict.items():
		# 	FD = apply_FD_noise(FD=FD, FD_par=FD_par, FD_sigma=FD_sigma)
		# ===== End of  BC noise =====
		FD['rho_j'] = np.random.normal(FD['rho_j'], scale=1e-5)
		mcmc = DelCastSampler(ICs=FD, cov=cov, cov_joint=cov_joint, section_dict=le_section_dict,
			move_probs=move_probs, config_dict=config_dict)
		mcmc.alpha_temp = alpha_temp_cst
		return mcmc
	return define_delCastsampler_temp



# beta=1
cov3 = np.array([[ 0.66878389,  0.89393652, -0.00067186,  0.00210403],
       [ 0.89393652,  3.33640027,  0.00612946, -0.00136162],
       [-0.00067186,  0.00612946,  0.00484103,  0.00047524],
       [ 0.00210403, -0.00136162,  0.00047524,  0.00010494]])
cov_joint3 = deepcopy(cov_inv_joint_DelCast_DS1)

# beta=0.76
cov2 = np.array([[ 0.78746895,  1.27333014,  0.00141434,  0.00308789],
       [ 1.27333014,  5.9597071 ,  0.01655901, -0.00077557],
       [ 0.00141434,  0.01655901,  0.00573808,  0.00047403],
       [ 0.00308789, -0.00077557,  0.00047403,  0.00011784]])
cov_joint2 = deepcopy(cov_inv_joint_DelCast_DS1)*(1/0.76)

# beta=0.58
cov1 = np.array([[ 0.91032747,  0.97317003,  0.00399942,  0.00339584],
       [ 0.97317003,  6.24639295,  0.02674442, -0.00593384],
       [ 0.00399942,  0.02674442,  0.0056371 ,  0.00051024],
       [ 0.00339584, -0.00593384,  0.00051024,  0.00012784]])
cov_joint1 = deepcopy(cov_inv_joint_DelCast_DS1)*(1/0.58)

# for beta=0.44
cov0 = np.array([[ 1.69715527,  2.72433505, -0.03295926,  0.00081907],
       [ 2.72433505, 10.64697739, -0.06830345, -0.01433845],
       [-0.03295926, -0.06830345,  0.01229563,  0.00125811],
       [ 0.00081907, -0.01433845,  0.00125811,  0.00026881]])
cov_joint0 = deepcopy(cov_inv_joint_DelCast_DS1)*(1/0.44)

# these are irrelevant as we're only sampling BCs
cov_list = [cov0, cov1, cov2, cov3]
cov_joint_list = [cov_joint0, cov_joint1, cov_joint2, cov_joint3]


# For population PT: index 0 is the untempered distribution
mcmc_fun2 = build_sampler_fun(alpha_temp_cst=0.58, section_dict=section_dict_70108_49t_delCast_temp058,
	cov=cov_list[1], cov_joint=cov_joint_list[1])

mcmc_fun1 = build_sampler_fun(alpha_temp_cst=0.76, section_dict=section_dict_70108_49t_delCast_temp076,
	cov=cov_list[2], cov_joint=cov_joint_list[2])

mcmc_fun0 = build_sampler_fun(alpha_temp_cst=1, section_dict=section_dict_70108_49t,
	cov=cov_list[3], cov_joint=cov_joint_list[3])

dict_samplers = {0: mcmc_fun0, 1: mcmc_fun1, 2: mcmc_fun2}


# numer of within temperature moves
within_temp_iter = 3
lesam = PopulationPTSampler(dict_samplers=dict_samplers, within_temp_iter=within_temp_iter, PT_width=5)

lesam.run(n_iter=3, print_rate=1)
