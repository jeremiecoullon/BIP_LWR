# -*- coding: utf-8 -*-

import numpy as np
from collections import OrderedDict
from BIP_LWR.moves.mh import MHMove
from .utils import create_N_y, build_section_info, build_cond_mean, Hastings_term_nonparam


class OUGibbsMove(MHMove):
    """
    RW on function space using OU proposals with Gibbs sampling
    On __init__(), builds 'self.section_info' which has the information for each Gibbs
    section (ie: size of blocks, prior mean etc..)

    Parameters
    ----------

    OU: Ornstein Uhlenbeck object

    section_dict: dict
        Format: `section_dict = {'section_1': {'param': 'x', 'cut1':0, 'cut2':100, 'omega':0.3},
                                'section_2': {'param': 'x', 'cut1':80, 'cut2':200, 'omega':0.3},
                                'section_3': {'param': 'y', 'cut1':0, 'cut2':200, 'omega':0.2},
                                }`
    prior_mean_dict: dict
        Dictionary of prior means for each parameter. Default is an empty dictionary
        which gives a prior mean of zeros
    """

    def __init__(self, OU, section_dict, prior_mean_dict={}):
        self.OU = OU
        self.prior_mean_dict = prior_mean_dict
        self.current_section = 'IC'
        self.section_dict = create_N_y(section_dict=section_dict, N=self.OU.N)

        for key, val in section_dict.items():
            new_dict = build_section_info(N_y1=val['N_y1'], N_y2=val['N_y2'], N_y3=val['N_y3'],
                                        prior_mean=self.prior_mean_dict.get(val['param'], np.zeros(self.OU.N)),
                                        OU_precision=self.OU.Precision)
            section_dict[key].update(new_dict)

        super(OUGibbsMove, self).__init__(self.get_proposal)

    def BC_Gibbs_propose(self, current_sample):
        """
        Cuts current sample based on the blocks in the current section and propose a new
        GP

        Parameters:
        current_sample: ndarray
            Current GP

        Returns:
        The proposed sample
        """
        y_1 = current_sample[0:self.N_y1]
        y_2 = current_sample[self.N_y1:(self.N_y1+self.N_y2)]
        y_3 = current_sample[(self.N_y1+self.N_y2)::]
        self.mean_y2 = build_cond_mean(y1=y_1,y3=y_3, mean1=self.mean1, mean2=self.mean2, mean3=self.mean3,
                                       block_22_21=self.block_22_21, block_22_23=self.block_22_23)
        proposal_y_2 = np.linalg.solve(self.chol_cond, np.random.normal(loc=0, scale=1, size=self.N_y2))
        y_2_sample = (proposal_y_2)*self.omega + np.sqrt(1-self.omega**2)*(y_2 - self.mean_y2) + self.mean_y2
        return np.concatenate([y_1, y_2_sample, y_3])


    def get_proposal(self, current_samples):
        """
        Given a dictionary of functional parameters, choose one them and update
        a subset of it. Return both parameters and the log-hastings correction

        Parameters
        ----------
        current_samples: dict
            Dictionary of parameter

        Returns
        -------
        new_samples: dict
            Dictionary of updated samples using an OU Gibbs move
        log_hastings: float
            Value of log-hastings for the proposal
        """
        # choose which section of a parameter to update
        if 'beta_temp_idx' not in current_samples.keys():
            self.current_section = np.random.choice(list(self.section_dict.keys()))
        else:
            # if running simulated tempering, choose only the relevant sections
            current_beta_temp_idx = current_samples['beta_temp_idx']
            list_sections_for_beta_temp = list({k:v for k,v in self.section_dict.items() if v['beta_temp_idx']==current_beta_temp_idx}.keys())
            self.current_section = np.random.choice(list_sections_for_beta_temp)
        for key, val in self.section_dict[self.current_section].items():
            setattr(self, key, val)

        # get the parameter to update
        fun_current_name = self.section_dict[self.current_section]['param']
        fun_current = current_samples[fun_current_name]

        # Do a Gibbs update on the chosen parameter
        fun_new = self.BC_Gibbs_propose(current_sample=fun_current)

        new_samples = OrderedDict({k:v for k,v in current_samples.items()})
        new_samples[fun_current_name] = fun_new

        # Hastings correction
        y_2_current = fun_current[self.N_y1:(self.N_y1+self.N_y2)]
        y_2_new = fun_new[self.N_y1:(self.N_y1+self.N_y2)]
        log_hastings = Hastings_term_nonparam(fun_current=y_2_current, fun_new=y_2_new,
                                    proposal_mean=self.mean_y2, precision=self.blocks['P_22'])

        return new_samples, log_hastings

    def __str__(self):
        return "OU move with Gibbs blocks"
