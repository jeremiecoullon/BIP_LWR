# -*- coding: utf-8 -*-

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from copy import deepcopy
from collections import OrderedDict

from BIP_LWR.moves.mh import MHMove
from BIP_LWR.moves.nonparametric_moves.OU_gibbs import OUGibbsMove
from BIP_LWR.gp.ou_process import OUProcess
from BIP_LWR.lwr.lwr_solver import LWR_Solver
from BIP_LWR.distributions.priors import cut_prior_means, create_interpolate_prior_mean_fun
from BIP_LWR.tools.util import load_CSV_data
from BIP_LWR.moves.nonparametric_moves.utils import transform_section_dict

class LWR_FD_BCGibbs(MHMove):
    """
    Base class for proposal in LWR. Need to subclass to add `propose_FD()`
    This should take as parameters: `new_samples, current_FD, current_BCs, FD_move, joint_BC_move=False`
    and return `new_samples, log_hastings`

    Gibbs proposal for the FD and BC:
    - Gaussian move for FD
    - exp(OU) move for BCs using Gibbs update

    Parameters
    ----------
    FD_move: move class for FD
        Move for a specific FD. Subclass of Gaussian move.
        Must have param_info and cov as arguments
    param_info: dict
        Initial conditions
    section_dict: dict
        Gibbs sections for boundary conditions
    cov: ndarray
        Covariance matrix for the standard FD proposal
    cov_joint: ndarray
        Covariance matrix for the joint FD/BC proposal
    move_probs: list
        List of probabilities to choose the following moves:
            - FD (random walk)
            - joint FD/BC
            - BCs (using Gibbs sampling)
        Must add up to 1.
    joint_prob: float
        If a FD move is chosen this is the probability of also doing a joint move for the BCs
    """

    def __init__(self, FDMove, param_info, section_dict, cov, cov_joint, config_dict,
            move_probs=[0.1, 0.2, 0.7], beta_temp_list=[1], beta_temp_margs=[1],
            dict_move_covs_tempering={0:{'cov':np.eye(2), 'cov_joint':np.eye(2)}}):
        super(LWR_FD_BCGibbs, self).__init__(self.get_proposal)

        FD_params = OrderedDict([(k, v) for k,v in param_info.items() if k not in ['BC_outlet', 'BC_inlet', 'beta_temp_idx']])
        BC_params = OrderedDict([(k, v) for k,v in param_info.items() if k in ['BC_outlet', 'BC_inlet']])

        # OU parameters
        self.lwr = LWR_Solver(config_dict=config_dict)
        self.N = self.lwr.final_time * self.lwr.config.ratio_times_BCs + 1
        self.OU_params = {'beta': 0.227812, 'dt': 1/self.lwr.config.ratio_times_BCs,'sigma': 0.255809}
        self.OU = OUProcess(N=self.N, **self.OU_params)

        outlet_prior_mean_150 = load_CSV_data(path='prior_means/mean_outlet_150.csv')
        inlet_prior_mean_150 = load_CSV_data(path='prior_means/mean_inlet_150.csv')
        outlet_prior_mean = cut_prior_means(prior_mean=outlet_prior_mean_150, data_array_str=self.lwr.config.data_array_dict['flow'])
        inlet_prior_mean = cut_prior_means(prior_mean=inlet_prior_mean_150, data_array_str=self.lwr.config.data_array_dict['flow'])
        self.BC_prior_mean = {'BC_outlet': np.exp(outlet_prior_mean),
                            'BC_inlet': np.exp(inlet_prior_mean),
                            }

        # define the FD and BC moves
        self.FDMove = FDMove
        self.FD_move = self.FDMove(param_info=FD_params, cov=cov)
        # self.w_transf_type = w_transf_type
        # so that the backend can access the covariance matrix
        self.cov = self.FD_move.cov

        self.FD_joint_move = self.FDMove(param_info=FD_params, cov=cov_joint)
        self.cov_joint = self.FD_joint_move.cov

        # transform section_dict based on BC resolution
        self.section_dict = transform_section_dict(section_dict=section_dict, ratio_times=self.lwr.config.ratio_times_BCs)
        # pass in an OU mean to OUGibbsMove (BC_prior_mean is for an exp(OU) process)
        BC_prior_mean_OU = {k: np.log(v) for k,v in self.BC_prior_mean.items()}
        # interpolate BCs when using a resolution higher than 1min
        f_outlet = create_interpolate_prior_mean_fun(final_time=self.lwr.final_time, prior_mean_raw=BC_prior_mean_OU['BC_outlet'])
        f_inlet = create_interpolate_prior_mean_fun(final_time=self.lwr.final_time, prior_mean_raw=BC_prior_mean_OU['BC_inlet'])

        outlet_mean_highres = f_outlet(x=np.linspace(0, self.lwr.final_time, self.lwr.final_time * self.lwr.config.ratio_times_BCs+1))
        inlet_mean_highres = f_inlet(x=np.linspace(0, self.lwr.final_time, self.lwr.final_time * self.lwr.config.ratio_times_BCs+1))
        BC_prior_mean_OU_highres = OrderedDict()
        BC_prior_mean_OU_highres['BC_outlet'] = outlet_mean_highres
        BC_prior_mean_OU_highres['BC_inlet'] = inlet_mean_highres
        self.BC_prior_mean['BC_outlet'] = np.exp(outlet_mean_highres)
        self.BC_prior_mean['BC_inlet'] = np.exp(inlet_mean_highres)
        self.BC_move = OUGibbsMove(self.OU, section_dict=deepcopy(self.section_dict), prior_mean_dict=BC_prior_mean_OU_highres)

        self.move_probs = move_probs
        if len(move_probs) == 3:
            self.FD_prob, self.joint_prob, self.BC_prob = move_probs
            self.beta_prob = 0
            if 'beta_temp_idx' in param_info.keys():
                raise ValueError("Need to define the move probability of beta_temp")
        elif len(move_probs) == 4:
            self.FD_prob, self.joint_prob, self.BC_prob, self.beta_prob = move_probs
            self.dict_move_covs_tempering = dict_move_covs_tempering
            if 'beta_temp_idx' not in param_info.keys():
                raise ValueError("Need to add beta_temp_idx to dictionary of parameters")
        else:
            raise ValueError("length of move_probs must be either 3 or 4")
        if len(beta_temp_list) != len(beta_temp_margs):
            raise ValueError("Length of 'beta_temp_list' and 'beta_temp_margs' must be the same")
        self.beta_temp_list = beta_temp_list
        self.beta_temp_margs = beta_temp_margs


    def get_proposal(self, current_samples):
        """
        Propose FD and BC as a Gibbs proposal:
        - FD: Gaussian proposal
        - BCs: exp(OU) proposal using Gibbs proposal


        Parameters
        ----------
        current_samples: dict
            Dictionary of current samples. Should have both BCs and FDs

        Returns
        -------
        new_samples: dict
            Dictionary of updated FD and BC samples
        log_hastings: float
            Value of log-hastings for the proposal
        """
        current_FD = OrderedDict([(k, v) for k, v in current_samples.items() if k not in ['BC_outlet', 'BC_inlet', 'beta_temp_idx']])
        current_BCs = OrderedDict([(k, v) for k, v in current_samples.items() if k in ['BC_outlet', 'BC_inlet']])
        new_samples = OrderedDict([(k, v) for k, v in current_samples.items()])

        self.current_gibbs_param = np.random.choice(['FD', 'BC', 'global', 'beta_temp'], p=[self.FD_prob, self.BC_prob, self.joint_prob, self.beta_prob])

        if self.current_gibbs_param == 'FD':
            new_samples, log_hastings = self.propose_FD(new_samples=new_samples, current_FD=current_FD,
                            current_BCs=current_BCs, FD_move=self.FD_move)

        elif self.current_gibbs_param == 'global':
            new_samples, log_hastings = self.propose_FD(new_samples=new_samples, current_FD=current_FD,
                        current_BCs=current_BCs, FD_move=self.FD_joint_move, joint_BC_move=True)

        elif self.current_gibbs_param == 'BC':
            # print("Running BC Gibbs move")
            # propose BC parameters
            log_current_BCs = OrderedDict([(k, np.log(v)) for k, v in current_BCs.items()])
            log_new_BC_samples, log_hastings = self.BC_move.get_proposal(log_current_BCs)
            new_BC_samples = OrderedDict([(k, np.exp(v)) for k, v in log_new_BC_samples.items()])
            # update new_samples with the proposed BCs
            for k, v in new_BC_samples.items():
                new_samples[k] = v

        return new_samples, log_hastings

    def propose_FD(self, new_samples, current_FD, current_BCs, FD_move, joint_BC_move=False):
        raise NotImplementedError("Must implement propose_FD() in subclass")

    def __str__(self):
        section_dict_str = "\n          ".join(["{}: {}".format(k,v) for k,v in self.section_dict.items()])
        move_description = """Random scan Gibbs sampler for FD and BCs:
-----------------------------------------
    FD move:
        probability of selecting move: {0}
        {1}

    Joint FD & BC move:
        probability of selecting move: {2}
        Propose a new FD and update BCs using a deterministic function

    BC move:
        probability of selecting move: {3}
        Gibbs sampling using blocks:
            {4}
        """.format(self.FD_prob, self.FD_move.__str__(), self.joint_prob, self.BC_prob, section_dict_str)
        return move_description

    def update_cov_tempering(self, current_samples):
        """
        Update covariance matrix of FD_move and FD_joint_move for the current beta_temp_idx
        """
        if 'beta_temp_idx' not in current_samples.keys():
            raise ValueError("Must be running simulated tempering to update covariance matrices")
        FD_params = OrderedDict([(k, v) for k, v in current_samples.items() if k not in ['BC_outlet', 'BC_inlet', 'beta_temp_idx']])
        cov = self.dict_move_covs_tempering[current_samples['beta_temp_idx']]['cov']
        cov_joint = self.dict_move_covs_tempering[current_samples['beta_temp_idx']]['cov_joint']

        self.FD_move = self.FDMove(param_info=FD_params, cov=cov)
        self.FD_joint_move = self.FDMove(param_info=FD_params, cov=cov_joint)


    def update_cov(self, cov, params):
        """
        Run __init__() to update the covariance matrix for the FD move

        Parameters
        ----------
        cov: ndarray
            New covariance matrix
        params: list
            List of parameters for covariance matrix

        """
        param_info = OrderedDict([(k, 1) for k in params])
        self.__init__(param_info=param_info, section_dict=self.section_dict, cov=cov, cov_joint=cov, move_probs=self.move_probs)
