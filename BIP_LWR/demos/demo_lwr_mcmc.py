#!/usr/bin/env python

import numpy as np
from collections import OrderedDict
from BIP_LWR.samplers.lwr.delcast_sampler import DelCastSampler
from BIP_LWR.lwr.lwr_solver import LWR_Solver
from BIP_LWR.tools.BC_ICs import BC_inlet_real_data_3, BC_outlet_real_data_3
from BIP_LWR.tools.util import transf_w
from BIP_LWR.tools.MCMC_params import section_dict, cov_inv, cov_inv_joint
from BIP_LWR.tools.util import scale_cov_directions

# --------------
# Set MCMC parameters
# --------------
w_transf_type = 'inv'
# covariance matrices
unscaled_cov_inv = np.array([[ 0.53855822, -0.18681024, -0.00724823,  0.00129741],
                   [-0.18681024,  3.60189924,  0.03020478, -0.00365225],
                   [-0.00724823,  0.03020478,  0.00064711, -0.00002783],
                   [ 0.00129741, -0.00365225, -0.00002783,  0.00003305]])
# Cov fitted to joint FD and BC sampling (in April3_2018-tune_real_data-Run_5: on real data ('data_array_70108_density_longer.csv'))
unscaled_cov_inv_joint = np.array([[ 3.81896926,  8.89477701, -0.05738674,  0.01151146],
           [ 8.89477701, 30.50346597, -0.07976908,  0.02867397],
           [-0.05738674, -0.07976908,  0.00422877,  0.00016535],
           [ 0.01151146,  0.02867397,  0.00016535,  0.00012827]])
cov = scale_cov_directions(cov=unscaled_cov_inv, alpha_list=[2, 1, 1, 1])
cov_joint = scale_cov_directions(cov=unscaled_cov_inv_joint, alpha_list=[1, 0.01, 0.01, 0.01])

BC_inlet = BC_inlet_real_data_3
BC_outlet = BC_outlet_real_data_3
move_probs = [0.1, 0.1, 0.8] # moves: [FD, joint, BCs]
le_section_dict = section_dict
upload_to_S3 = False
comments = """FD cov (fitted to pi(FD|BC) from real data) with alpha_list=[2,1,1,1]. Joint FD/BC move (fitted to pi(FD,BC) from real data) with alpha_list=[1, 0.01, 0.01, 0.01]."""

config_dict = {'my_analysis_dir': '2018/Demo_del_Castillo_MCMC',
                'run_num': 1,
                'data_array_dict':
                        {'flow': 'data_array_flow_70108_longer.csv',
                        'density': 'data_array_70108_density_longer.csv'},
                'upload_to_S3': upload_to_S3,
                'save_chain': True,
                'w_transf_type': w_transf_type,
                'comments': comments,
                'root_analysis_folder': ""
                      }


# --------------
# --------------

def random_ICs():
    """
    Initialise FD initial conditions randomly
    """
    IC_cov = np.diag([2, 2, 0.01, 0.01])
    mean_dict = {'IC_mean_1': [172, 440, 3.2, 9], 'IC_mean_2': [168, 450, 3.3, 20]}
    mean_key = np.random.choice(['IC_mean_1', 'IC_mean_2'], p=[0.4,0.6])
    z, rho_j, u, w = np.random.multivariate_normal(mean=mean_dict[mean_key], cov=IC_cov)
    FD = OrderedDict([('z', z), ('rho_j', rho_j), ('u', u), ('w', transf_w(w=w, w_transf_type=w_transf_type)),
            ('BC_outlet', BC_outlet), ('BC_inlet', BC_inlet)])

    return FD

if __name__ == '__main__':
    """
    Run MCMC for Del Castillo's FD and save the ouptut to hdf5 file
    """
    FD = random_ICs()
    mcmc = DelCastSampler(ICs=FD, cov=cov, cov_joint=cov_joint, section_dict=le_section_dict,
                        move_probs=move_probs, config_dict=config_dict)

    mcmc.run(n_iter=2, print_rate=1)
