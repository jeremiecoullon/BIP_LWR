# -*- coding: utf-8 -*-

import numpy as np
from copy import deepcopy

from .lwrsampler import LWRSampler
from BIP_LWR.tools.BC_ICs import BC_outlet_IC_1, BC_inlet_IC_1
from BIP_LWR.moves.lwr_moves.del_Cast_move import DelCastMove
from BIP_LWR.distributions.priors import uniform_log_prob, prior_bounds
from BIP_LWR.config import DelCastConfigHolder
from BIP_LWR.tools.util import build_schedule_array, scale_gibbs_blocks
from .util_sampler import check_sampler_param_order

class DelCastSampler(LWRSampler):
    """
    MCMC sampler for del Castillo FD

    Parameters
    ----------
    ICs: dict
        Dictionary of parameters (FDs and BCs) to use as initial conditions
    config_dict: dict
        Dictionary of configuration parameters
    cov: ndarray
        Covariance matrix for FD proposal.
        Default is None, which results in an idenity matrix
        Should be 4D if use a Gaussian proposal for all parameters, and 3D
        if the 'inv' w parametrisation is used
    cov_joint: ndarray
        Covariance matrix for the joint FD/BC proposal. This usually has higher
        variance than 'cov' as you want to make global moves
        Default is None, which results in an idenity matrix
        Should be 4D if use a Gaussian proposal for all parameters, and 3D
        if the 'inv' w parametrisation is used
    verbose: int
        0: print nothing
        1: print basic info when running.
        2: print basic info when running along with acceptance rate
    section_dict: dict
        Format: `section_dict = {'section_1': {'param': 'BC_inlet', 'cut1':0, 'cut2':150, 'omega':0.3},
                                'section_2': {'param': 'BC_inlet', 'cut1':80, 'cut2':100, 'omega':0.3},
                                'section_2': {'param': 'BC_outlet', 'cut1':0, 'cut2':150, 'omega':0.2},
                                }`
    data_array : ndarray or None
        array of raw data (3 columns) to use to compute the loss
        If None: use data_array in config.py
    move_probs: list
        List of probabilities to choose the following moves:
            - FD (random walk)
            - joint FD/BC
            - BCs (using Gibbs sampling)
        Must add up to 1.
    """

    def __init__(self, ICs, config_dict, cov=None, cov_joint=None, verbose=1, section_dict=None,
                    data_array=None, move_probs=[0.1, 0.2, 0.7]):

        check_sampler_param_order(ICs=ICs, FD_type="del_Cast")

        super(DelCastSampler, self).__init__(ICs=ICs, cov=cov, verbose=verbose,
                    data_array=data_array, config_dict=config_dict)

        self.config = DelCastConfigHolder(**config_dict)

        # self.TEMP_SA_DICT_MOVE = {'param_info': deepcopy(ICs), 'cov': deepcopy(cov),
        #         'cov_joint': deepcopy(cov_joint), 'section_dict': deepcopy(section_dict),
        #         'move_probs': deepcopy(move_probs), 'config_dict': deepcopy(config_dict),
        #         'w_transf_type': deepcopy(self.config.w_transf_type)}

        self.solver = 'lwr_del_Cast'
        self.move = DelCastMove(param_info=ICs, cov=cov, cov_joint=cov_joint,
                        section_dict=section_dict, move_probs=move_probs,
                        config_dict=config_dict, w_transf_type=self.config.w_transf_type)

        # simulated annealing schedule array
        self.len_schedule = 10
        self.num_plateaux = 2
        self.initial_temp = 1
        self.SA_schedule_array = build_schedule_array(len_schedule=self.len_schedule, num_plateaux=self.num_plateaux, initial_temp=self.initial_temp)
        self.alpha_temp = 1

        # TEMP: SIMULATED ANNEALING TEST
        # alpha_val, alpha_changed = self.alpha_schedule()
        # move_dict = deepcopy(self.TEMP_SA_DICT_MOVE)
        # move_dict['cov'] = (1/alpha_val) * move_dict['cov']
        # move_dict['cov_joint'] = (1/alpha_val) * move_dict['cov_joint']
        # move_dict['section_dict'] = scale_gibbs_blocks(alpha=alpha_val, section_dict=move_dict['section_dict'])
        # self.move = LWR_FD_BCGibbs(**move_dict)
        # TEMP: ============

    def log_prior_w(self, w):
        if self.config.w_transf_type == 'inv':
            w = 1/w
            return uniform_log_prob(theta=w, lower=prior_bounds['w'][0], upper=prior_bounds['w'][1])
        elif self.config.w_transf_type == 'log_inv':
            return 0
        elif self.config.w_transf_type == 'nat':
            if w < 0:
                return -1000000
            else:
                return 0

    def log_prior_z(self, z):
        """
        Uniform prior on z
        """
        return uniform_log_prob(theta=z, lower=prior_bounds['z'][0], upper=prior_bounds['z'][1])

    def log_prior_rho_j(self, rho_j, BC_inlet, BC_outlet):
        """
        Uniform prior on rho_j
        """
        lower = prior_bounds['rho_j'][0]
        return uniform_log_prob(theta=rho_j, lower=lower, upper=prior_bounds['rho_j'][1])

    def log_prior_u(self, u):
        """
        Uniform prior on rho_j
        """
        return uniform_log_prob(theta=u, lower=prior_bounds['u'][0], upper=prior_bounds['u'][1])

    def log_prior(self, z, rho_j, u, w, BC_inlet, BC_outlet):
        return self.log_prior_w(w) + self.log_prior_z(z) + self.log_prior_rho_j(rho_j, BC_inlet=BC_inlet, BC_outlet=BC_outlet) + self.log_prior_u(u)

    def alpha_schedule(self):
        """
        Schedule for Simulated Annealing.
        Note that the last element of self.SA_schedule_array must be 1 (so that you calculate
        the log-posterior for the last change)

        Returns
        -------
        alpha_temp: float
            Parameter to multiplty log-likelihood by
        alpha_changed: Bool
            Whether or not alpha_temp has just changed
        """
        alpha_temp = 1
        alpha_changed = False
        iter_num = len(self.backend.log_post_list)
        if iter_num < len(self.SA_schedule_array):
            alpha_temp = self.SA_schedule_array[iter_num]
            if iter_num>0 and (self.SA_schedule_array[iter_num] != self.SA_schedule_array[iter_num-1]):
                alpha_changed = True
            else:
                pass
        else:
            pass
        return alpha_temp, alpha_changed

    def log_posterior(self, BC_outlet=BC_outlet_IC_1, BC_inlet=BC_inlet_IC_1, *args, **kwargs):
        """
        Note: pass in transformed w (ex: inverted). This will be inverted back to the
        natural parametrisation (self.lwr.loss_function() takes in the natural parametrisation).
        """
        if self.config.w_transf_type == 'inv':
            kwargs['w'] = 1/kwargs.pop('w')
        elif self.config.w_transf_type == 'log_inv':
            kwargs['w'] = 1/np.exp(kwargs.pop('w'))
        elif self.config.w_transf_type == 'nat':
            pass
        alpha_temp, _ = self.alpha_schedule()
        log_lik = -self.lwr.loss_function(solver=self.solver, BC_outlet=BC_outlet, BC_inlet=BC_inlet, *args, **kwargs)
        # log_lik = 0
        log_p_FD = self.log_prior(BC_outlet=BC_outlet, BC_inlet=BC_inlet, *args, **kwargs)
        log_p_BC_inlet = self.log_prior_BC(BC=BC_inlet, mean=self.move.BC_prior_mean['BC_inlet'])
        log_p_BC_outlet = self.log_prior_BC(BC=BC_outlet, mean=self.move.BC_prior_mean['BC_outlet'])
        # return log_lik + log_p_FD + log_p_BC_inlet + log_p_BC_outlet
        return self.alpha_temp*log_lik + log_p_FD + log_p_BC_inlet + log_p_BC_outlet

    def __str__(self):
        return "Del Castillo sampler"
